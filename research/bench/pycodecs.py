"""In-process Python codecs, run in a child process by the harness.

    python -I pycodecs.py c|d <codec> <input> <output>

Prints one JSON line: {"ms": elapsed, ...extra size breakdown}.
"""
import bz2
import gzip
import heapq
import json
import lzma
import struct
import sys
import time

import numpy as np


# ---------------------------------------------------------------- general-purpose libraries

def _zstd_c(b):
    import zstandard as zs
    p = zs.ZstdCompressionParameters.from_level(19, enable_ldm=True, window_log=27, source_size=len(b))
    return zs.ZstdCompressor(compression_params=p).compress(b)


def _zstd_d(b):
    import zstandard as zs
    return zs.ZstdDecompressor(max_window_size=1 << 27).decompress(b)


def _brotli_c(b):
    import brotli
    return brotli.compress(b, quality=11, lgwin=24)


def _brotli_d(b):
    import brotli
    return brotli.decompress(b)


def _ppmd_c(b):
    import pyppmd
    # PPMd variant H (as in 7-Zip), order 6 (7-Zip's default; pyppmd mis-decodes some inputs at order 8),
    # 256 MiB model; the original length is stored in front.
    enc = pyppmd.Ppmd7Encoder(6, 1 << 28)
    return struct.pack('<Q', len(b)) + enc.encode(b) + enc.flush()


def _ppmd_d(b):
    import pyppmd
    (n,) = struct.unpack('<Q', b[:8])
    return pyppmd.Ppmd7Decoder(6, 1 << 28).decode(b[8:], n) if n else b''


LIB = {
    'gzip': (lambda b: gzip.compress(b, 9, mtime=0), gzip.decompress),
    'bzip2': (lambda b: bz2.compress(b, 9), bz2.decompress),
    'xz': (lambda b: lzma.compress(b, preset=9 | lzma.PRESET_EXTREME), lzma.decompress),
    'brotli': (_brotli_c, _brotli_d),
    'zstd': (_zstd_c, _zstd_d),
    'ppmd': (_ppmd_c, _ppmd_d),
}


# ---------------------------------------------------------------- Huffman helpers

def code_lengths(freq):
    """Binary Huffman code lengths for symbols with nonzero frequency."""
    syms = [s for s in range(len(freq)) if freq[s] > 0]
    lengths = np.zeros(len(freq), dtype=np.int64)
    if len(syms) == 1:
        lengths[syms[0]] = 1
        return lengths
    heap = [(int(freq[s]), i, [s]) for i, s in enumerate(syms)]
    heapq.heapify(heap)
    tie = len(heap)
    while len(heap) > 1:
        f1, _, a = heapq.heappop(heap)
        f2, _, b = heapq.heappop(heap)
        for s in a + b:
            lengths[s] += 1
        heapq.heappush(heap, (f1 + f2, tie, a + b))
        tie += 1
    return lengths


def canonical(lengths):
    """Canonical codes (as integers) for the given lengths."""
    codes = np.zeros(len(lengths), dtype=np.int64)
    order = sorted((int(l), s) for s, l in enumerate(lengths) if l > 0)
    code, prev = 0, order[0][0]
    for i, (l, s) in enumerate(order):
        if i:
            code = (code + 1) << (l - prev)
        codes[s] = code
        prev = l
    return codes


def pack_codes(symbols, lengths, codes):
    """Concatenate the codes of `symbols` MSB-first into a numpy bit array."""
    L = lengths[symbols]
    C = codes[symbols]
    maxl = int(L.max())
    shifts = np.arange(maxl - 1, -1, -1)
    # bit j of a code of length l sits at position (maxl - l + j) of a right-aligned row
    bits = ((C[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    mask = shifts[None, :] < L[:, None]
    return bits[mask]


class BitReader:
    def __init__(self, data, start_bit=0):
        self.bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        self.pos = start_bit


def decode_block(reader, n, lengths, codes):
    """Decode n symbols with a lookup table indexed by the next maxl bits."""
    maxl = int(lengths.max())
    table_sym = np.zeros(1 << maxl, dtype=np.int64)
    table_len = np.zeros(1 << maxl, dtype=np.int64)
    for s in np.nonzero(lengths)[0]:
        l = int(lengths[s])
        lo = int(codes[s]) << (maxl - l)
        table_sym[lo:lo + (1 << (maxl - l))] = s
        table_len[lo:lo + (1 << (maxl - l))] = l
    bits = reader.bits
    # Pre-compute the maxl-bit window value at every bit position of the region we may read.
    need = reader.pos + n * maxl + maxl
    seg = np.zeros(need - reader.pos, dtype=np.int64)
    avail = bits[reader.pos:need].astype(np.int64)
    seg[:len(avail)] = avail
    w = np.zeros(len(seg) - maxl + 1, dtype=np.int64)
    for j in range(maxl):
        w = (w << 1) | seg[j:j + len(w)]
    out = np.empty(n, dtype=np.uint8 if len(lengths) <= 256 else np.int64)
    p = 0
    ts, tl = table_sym.tolist(), table_len.tolist()
    wl = w.tolist()
    for i in range(n):
        v = wl[p]
        out[i] = ts[v]
        p += tl[v]
    reader.pos += p
    return out


def limit_lengths(freq, maxlen=24):
    """Huffman lengths; counts are halved (keeping every used symbol at least 1) until the longest
    code fits in maxlen bits, the usual way to length-limit a Huffman code."""
    f = np.asarray(freq, dtype=np.int64).copy()
    while True:
        L = code_lengths(f)
        if L.max() <= maxlen:
            return L
        f = np.where(f > 0, np.maximum(f >> 1, 1), 0)


# ---------------------------------------------------------------- blockwise dynamic Huffman (Klein et al. 2024)

DYN_BLOCK = 1 << 12


def dyn_huffman_c(b):
    """Every 4 KiB block is coded with a Huffman tree built from all bytes seen so far
    (counts start at one). The decoder rebuilds the same tree, so no table is stored."""
    data = np.frombuffer(b, dtype=np.uint8)
    counts = np.ones(256, dtype=np.int64)
    pieces = []
    for start in range(0, len(data), DYN_BLOCK):
        blk = data[start:start + DYN_BLOCK]
        L = limit_lengths(counts, 16)
        pieces.append(pack_codes(blk.astype(np.int64), L, canonical(L)))
        counts += np.bincount(blk, minlength=256)
    bits = np.concatenate(pieces) if pieces else np.zeros(0, np.uint8)
    return struct.pack('<Q', len(data)) + np.packbits(bits).tobytes()


def dyn_huffman_d(b):
    (n,) = struct.unpack('<Q', b[:8])
    reader = BitReader(b[8:])
    counts = np.ones(256, dtype=np.int64)
    out = []
    for start in range(0, n, DYN_BLOCK):
        m = min(DYN_BLOCK, n - start)
        L = limit_lengths(counts, 16)
        blk = decode_block(reader, m, L, canonical(L))
        out.append(blk)
        counts += np.bincount(blk, minlength=256)
    return np.concatenate(out).tobytes() if out else b''


def varints(values):
    out = bytearray()
    for v in values:
        v = int(v)
        while v >= 128:
            out.append((v & 127) | 128)
            v >>= 7
        out.append(v)
    return bytes(out)


def read_varints(b, pos, n):
    vals = []
    for _ in range(n):
        v = shift = 0
        while True:
            c = b[pos]
            pos += 1
            v |= (c & 127) << shift
            shift += 7
            if c < 128:
                break
        vals.append(v)
    return np.array(vals, dtype=np.uint32), pos


# ---------------------------------------------------------------- static order-0 rANS (Duda's ANS, via constriction)

def rans0_c(b):
    import constriction
    data = np.frombuffer(b, dtype=np.uint8).astype(np.int32)
    freq = np.bincount(data, minlength=256).astype(np.uint32)
    if len(data) == 0:
        return struct.pack('<Q', 0)
    model = constriction.stream.model.Categorical(freq.astype(np.float64) / freq.sum(), perfect=False)
    enc = constriction.stream.stack.AnsCoder()
    enc.encode_reverse(data, model)
    words = enc.get_compressed()
    # header: length + the 256 symbol counts as varints (the decoder needs them to rebuild the model)
    return struct.pack('<Q', len(data)) + varints(freq) + words.tobytes()


def rans0_d(b):
    import constriction
    (n,) = struct.unpack('<Q', b[:8])
    if n == 0:
        return b''
    freq, pos = read_varints(b, 8, 256)
    words = np.frombuffer(b[pos:], dtype=np.uint32)
    model = constriction.stream.model.Categorical(freq.astype(np.float64) / freq.sum(), perfect=False)
    dec = constriction.stream.stack.AnsCoder(words)
    return dec.decode(model, n).astype(np.uint8).tobytes()


CUSTOM = {
    'dyn-huffman': (dyn_huffman_c, dyn_huffman_d),
    'rans0': (rans0_c, rans0_d),
}


def main():
    cmd, codec, src, dst = sys.argv[1:5]
    with open(src, 'rb') as f:
        data = f.read()
    fn = (LIB.get(codec) or CUSTOM[codec])[0 if cmd == 'c' else 1]
    t = time.perf_counter()
    out = fn(data)
    ms = (time.perf_counter() - t) * 1000
    with open(dst, 'wb') as f:
        f.write(out)
    print(json.dumps({'ms': ms}))


if __name__ == '__main__':
    main()
