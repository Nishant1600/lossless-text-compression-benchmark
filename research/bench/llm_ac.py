"""Lossless compression with a pretrained language model and range coding.

    python -I llm_ac.py c|d MODEL INPUT OUTPUT [--batch 32] [--window 1023]

This follows the setup of LLMZip (Valmeekam et al., 2023) and "Language Modeling Is Compression"
(Deletang et al., ICLR 2024): the model's next-token distribution drives an arithmetic (here: range)
coder, and the text is cut into independent windows so long inputs fit the context.

Design choices that keep decoding exact:
* The input is cut at line breaks into pieces whose tokenisation round-trips exactly and fits in
  `window` tokens. A piece that is not valid UTF-8, or whose tokens do not decode back to the same
  bytes, is stored as raw bytes (zstd) and flagged; its share is reported.
* Encoding and decoding run the *same* incremental computation (one token per step with a key/value
  cache, the same batch of pieces in the same order), so the GPU produces bit-identical probabilities
  on both sides. Probabilities are quantised on the GPU to integer frequencies (every token gets at
  least 1 of a total below 2^30), and an exact integer arithmetic coder (Witten-Neal-Cleary style,
  32-bit state) codes the symbol from its cumulative-frequency interval. Only three integers per
  row and step cross to the CPU, which keeps the GPU busy.

Container: magic, model name, piece table (varints), raw-piece bytes (zstd), range-coded words.
Prints a JSON line with sizes, token counts and timings.
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np

os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch  # noqa: E402
import zstandard  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MAGIC = b'LLMAC1'


def varint(v):
    out = bytearray()
    while v >= 128:
        out.append((v & 127) | 128)
        v >>= 7
    out.append(v)
    return bytes(out)


def read_varint(b, pos):
    v = shift = 0
    while True:
        c = b[pos]
        pos += 1
        v |= (c & 127) << shift
        shift += 7
        if c < 128:
            return v, pos


def load(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).cuda().eval()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    return tok, model, bos


def split_pieces(data, tok, window, target_bytes):
    """Cut bytes into pieces at line breaks. Returns list of (raw_bytes, token_ids or None)."""
    lines = data.splitlines(keepends=True)
    pieces, cur = [], b''
    for line in lines:
        if cur and len(cur) + len(line) > target_bytes:
            pieces.append(cur)
            cur = b''
        cur += line
    if cur:
        pieces.append(cur)
    out = []
    stack = list(reversed(pieces))
    while stack:
        p = stack.pop()
        try:
            text = p.decode('utf-8')
        except UnicodeDecodeError:
            out.append((p, None))
            continue
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) > window:
            # too long: split near the middle at a line break (or a character boundary) and retry
            mid = len(text) // 2
            cut = text.rfind('\n', 0, mid) + 1 or mid
            a, b = text[:cut].encode('utf-8'), text[cut:].encode('utf-8')
            stack.extend([b, a])
            continue
        if tok.decode(ids, clean_up_tokenization_spaces=False).encode('utf-8') != p or not ids:
            out.append((p, None))
            continue
        out.append((p, ids))
    return out


STATE = 32
FULL, HALF, QUARTER = 1 << STATE, 1 << (STATE - 1), 1 << (STATE - 2)
MASK = FULL - 1


class ArithEncoder:
    def __init__(self):
        self.low, self.high, self.pending, self.bits = 0, MASK, 0, bytearray()

    def _emit(self, bit):
        self.bits.append(bit)
        self.bits.extend([bit ^ 1] * self.pending)
        self.pending = 0

    def encode(self, cl, ch, tot):
        rng = self.high - self.low + 1
        self.high = self.low + rng * ch // tot - 1
        self.low = self.low + rng * cl // tot
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < HALF + QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self):
        self.pending += 1
        self._emit(0 if self.low < QUARTER else 1)
        return np.packbits(np.frombuffer(bytes(self.bits), dtype=np.uint8)).tobytes()


class ArithDecoder:
    def __init__(self, data):
        self.bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8)).tolist()
        self.pos = 0
        self.low, self.high, self.code = 0, MASK, 0
        for _ in range(STATE):
            self.code = (self.code << 1) | self._bit()

    def _bit(self):
        b = self.bits[self.pos] if self.pos < len(self.bits) else 0
        self.pos += 1
        return b

    def target(self, tot):
        return ((self.code - self.low + 1) * tot - 1) // (self.high - self.low + 1)

    def update(self, cl, ch, tot):
        rng = self.high - self.low + 1
        self.high = self.low + rng * ch // tot - 1
        self.low = self.low + rng * cl // tot
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= QUARTER and self.high < HALF + QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.code -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.code = (self.code << 1) | self._bit()


def quantised_cdf(logits, vocab):
    """Integer cumulative frequencies (B, V) on the GPU; every token gets frequency >= 1, total < 2^30."""
    p = torch.softmax(logits[:, :vocab].float(), dim=-1).double()
    scale = float(QUARTER - vocab - 1)
    freq = torch.floor(p * scale).long() + 1
    return torch.cumsum(freq, dim=-1)


@torch.inference_mode()
def run_batches(model, bos, lengths, batch, vocab, step_fn):
    """Drive the model over pieces of the given token lengths, `batch` pieces at a time.
    step_fn(rows, t, cum) -> next tokens (LongTensor on the GPU, one per row)."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])  # deterministic from the header
    for s in range(0, len(order), batch):
        rows = order[s:s + batch]
        n = max(lengths[i] for i in rows)
        inp = torch.full((len(rows), 1), bos, dtype=torch.long, device='cuda')
        past = None
        for t in range(n):
            out = model(input_ids=inp, past_key_values=past, use_cache=True)
            past = out.past_key_values
            cum = quantised_cdf(out.logits[:, -1, :], vocab)
            inp = step_fn(rows, t, cum).view(-1, 1)


def compress(args):
    t0 = time.perf_counter()
    data = open(args.input, 'rb').read()
    tok, model, bos = load(args.model)
    vocab = model.config.vocab_size
    pieces = split_pieces(data, tok, args.window, args.target_bytes)
    coded = [ids for _, ids in pieces if ids is not None]
    lengths = [len(ids) for ids in coded]
    encs = [ArithEncoder() for _ in coded]  # one stream per piece, so pieces in a batch stay independent

    def step(rows, t, cum):
        sym = torch.tensor([coded[i][t] if t < lengths[i] else 0 for i in rows], dtype=torch.long, device='cuda')
        hi = cum.gather(1, sym.view(-1, 1)).squeeze(1)
        lo = torch.where(sym > 0, cum.gather(1, (sym - 1).clamp(min=0).view(-1, 1)).squeeze(1), torch.zeros_like(hi))
        tot = cum[:, -1]
        trip = torch.stack([lo, hi, tot], 1).cpu().tolist()
        for k, i in enumerate(rows):
            if t < lengths[i]:
                encs[i].encode(*trip[k])
        return sym

    t1 = time.perf_counter()
    run_batches(model, bos, lengths, args.batch, vocab, step)
    t2 = time.perf_counter()
    streams = [e.finish() for e in encs]
    words = b''.join(varint(len(x)) + x for x in streams)
    table = bytearray(varint(len(pieces)))
    raw = b''
    for p, ids in pieces:
        if ids is None:
            table += varint(2 * len(p) + 1)
            raw += p
        else:
            table += varint(2 * len(ids))
    raw_z = zstandard.ZstdCompressor(level=19).compress(raw) if raw else b''
    name = args.model.encode()
    blob = (MAGIC + bytes([len(name)]) + name + struct.pack('<III', len(table), len(raw_z), len(words))
            + bytes(table) + raw_z + words)
    open(args.output, 'wb').write(blob)
    print(json.dumps(dict(
        ms=(time.perf_counter() - t0) * 1000, model_ms=(t2 - t1) * 1000,
        payload=len(words), side=len(blob) - len(words), raw_bytes=len(raw), raw_pieces=len(pieces) - len(coded),
        pieces=len(pieces), tokens=sum(lengths), input_bytes=len(data),
    )))


def decompress(args):
    t0 = time.perf_counter()
    b = open(args.input, 'rb').read()
    assert b[:6] == MAGIC
    k = b[6]
    name = b[7:7 + k].decode()
    pos = 7 + k
    lt, lr, lw = struct.unpack('<III', b[pos:pos + 12])
    pos += 12
    table, raw_z, words = b[pos:pos + lt], b[pos + lt:pos + lt + lr], b[pos + lt + lr:pos + lt + lr + lw]
    raw = zstandard.ZstdDecompressor().decompress(raw_z) if raw_z else b''
    n, p = read_varint(table, 0)
    kinds = []
    for _ in range(n):
        v, p = read_varint(table, p)
        kinds.append(('raw', v >> 1) if v & 1 else ('llm', v >> 1))
    tok, model, bos = load(name)
    vocab = model.config.vocab_size
    lengths = [m for kind, m in kinds if kind == 'llm']
    decoded = [[] for _ in lengths]
    decs, p2 = [], 0
    for _ in lengths:
        k2, p2 = read_varint(words, p2)
        decs.append(ArithDecoder(words[p2:p2 + k2]))
        p2 += k2

    def step(rows, t, cum):
        tot = cum[:, -1].cpu().tolist()
        active = [t < lengths[i] for i in rows]
        tgt = torch.tensor([decs[i].target(tot[k]) if active[k] else 0 for k, i in enumerate(rows)], dtype=torch.long,
                           device='cuda')
        sym = torch.searchsorted(cum, tgt.view(-1, 1), right=True).squeeze(1)
        hi = cum.gather(1, sym.view(-1, 1)).squeeze(1)
        lo = torch.where(sym > 0, cum.gather(1, (sym - 1).clamp(min=0).view(-1, 1)).squeeze(1), torch.zeros_like(hi))
        trip = torch.stack([lo, hi, sym], 1).cpu().tolist()
        for k, i in enumerate(rows):
            if active[k]:
                decs[i].update(trip[k][0], trip[k][1], tot[k])
                decoded[i].append(trip[k][2])
        return sym

    run_batches(model, bos, lengths, args.batch, vocab, step)
    out, li, ri = bytearray(), 0, 0
    for kind, m in kinds:
        if kind == 'raw':
            out += raw[ri:ri + m]
            ri += m
        else:
            out += tok.decode(decoded[li], clean_up_tokenization_spaces=False).encode('utf-8')
            li += 1
    open(args.output, 'wb').write(bytes(out))
    print(json.dumps(dict(ms=(time.perf_counter() - t0) * 1000)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['c', 'd'])
    ap.add_argument('model')
    ap.add_argument('input')
    ap.add_argument('output')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--window', type=int, default=1023)
    ap.add_argument('--target_bytes', type=int, default=4096)
    a = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    (compress if a.cmd == 'c' else decompress)(a)


if __name__ == '__main__':
    main()
