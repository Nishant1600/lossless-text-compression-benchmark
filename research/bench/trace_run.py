"""TRACE (Mao et al., WWW 2022) as separate compress / decompress commands.

    python -I trace_run.py c|d INPUT OUTPUT

Uses the authors' model (compress_model.SLiMPerformer) and arithmetic coder from their repository,
with the hyper-parameters of their README (batch 512, hidden 256, ffn 4096, seq_len 8, vocab_dim 64,
lr 1e-3). Changes from the published script, none of which affect the coding itself:
* compression and decompression are separate runs, and the decoder reads only the container file;
* np.asscalar (removed from NumPy) is restored as a.item() for the authors' coder;
* bytes are read with np.frombuffer (np.fromstring is gone from NumPy 2) and written back as bytes
  (the original wrote decoded text through chr(max(32, b)), which cannot restore control bytes);
* when the input length is an exact multiple of the batch size the original skipped the last
  seq_len bytes; here the tail is always coded as a separate uniform stream.
"""
import io
import json
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party',
                    'A-Fast-Transformer-based-General-Purpose-LosslessCompressor')
sys.path.insert(0, os.path.abspath(REPO))
np.asscalar = lambda a: a.item()  # removed in NumPy 1.23; the authors' coder still calls it
import arithmeticcoding_fast as ac  # noqa: E402
import compress_model  # noqa: E402

BS, LR, HID, VOC_DIM, FFN, HEADS, LAYERS, SEQ0 = 512, 1e-3, 256, 64, 4096, 8, 1, 8
SEQ = SEQ0 * (HID // VOC_DIM)
V = 256
MAGIC = b'TRACE1'


def setup():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = compress_model.SLiMPerformer(V, VOC_DIM, HID, LAYERS, FFN, HEADS, 'sqr', 'iter').cuda()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0, betas=(.9, .999))
    return model, opt


class Buf(io.BytesIO):
    def close(self):  # BitOutputStream.close() would close the buffer before we read it
        pass


UNIFORM = np.zeros(V + 1, dtype=np.uint64)
UNIFORM[1:] = np.cumsum(np.ones(V) / V * 10000000 + 1)


def strided(a, L):
    n = a.size - L + 1
    return np.lib.stride_tricks.as_strided(a, shape=(n, L), strides=(a.strides[0], a.strides[0]))


def split_point(n):
    total = n - SEQ
    l = (total // BS) * BS
    if l == total:
        l -= BS
    return l


def write_streams(bufs):
    out = bytearray()
    for b in bufs:
        out += struct.pack('<I', len(b)) + b
    return bytes(out)


def compress(src, dst):
    data = open(src, 'rb').read()
    series = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    n = len(series)
    l = split_point(n)
    if n < 4 * BS * (SEQ + 1) or l <= 0:
        open(dst, 'wb').write(MAGIC + struct.pack('<QB', n, 1) + data)  # too small for the model: stored
        return {}
    train = strided(series[:l + SEQ], SEQ + 1)[:l]
    iters = l // BS
    ind = np.arange(BS) * iters
    steps = iters - SEQ
    bufs = [Buf() for _ in range(BS)]
    bitout = [ac.BitOutputStream(b) for b in bufs]
    enc = [ac.ArithmeticEncoder(32, o) for o in bitout]
    for i in range(BS):
        for j in range(SEQ):
            enc[i].write(UNIFORM, int(series[ind[i] + j]))
    model, opt = setup()
    cumul = np.zeros((BS, V + 1), dtype=np.uint64)
    for _ in range(steps):
        model.train()
        batch = train[ind, :]
        y = batch[:, -1]
        _, logits = model.full_loss(torch.from_numpy(batch).cuda().long(), with_grad=True)
        opt.step()
        opt.zero_grad(set_to_none=True)
        prob = F.softmax(logits.transpose(1, 2)[:, -1, :], dim=1).detach().cpu().numpy()
        cumul[:, 1:] = np.cumsum(prob * 10000000 + 1, axis=1)
        for i in range(BS):
            enc[i].write(cumul[i, :], int(y[i]))
        ind += 1
    for i in range(BS):
        enc[i].finish()
        bitout[i].close()
    # tail: everything from l on, uniform
    tb = Buf()
    to = ac.BitOutputStream(tb)
    te = ac.ArithmeticEncoder(32, to)
    for v in series[l:]:
        te.write(UNIFORM, int(v))
    te.finish()
    to.close()
    streams = [b.getvalue() for b in bufs] + [tb.getvalue()]
    open(dst, 'wb').write(MAGIC + struct.pack('<QB', n, 0) + write_streams(streams))
    return dict(steps=steps)


def decompress(src, dst):
    b = open(src, 'rb').read()
    assert b[:6] == MAGIC
    n, stored = struct.unpack('<QB', b[6:15])
    if stored:
        open(dst, 'wb').write(b[15:])
        return {}
    pos, streams = 15, []
    while pos < len(b):
        (k,) = struct.unpack('<I', b[pos:pos + 4])
        streams.append(b[pos + 4:pos + 4 + k])
        pos += 4 + k
    l = split_point(n)
    iters = l // BS
    s2 = np.zeros((BS, iters), dtype=np.int64)
    dec = [ac.ArithmeticDecoder(32, ac.BitInputStream(io.BytesIO(streams[i]))) for i in range(BS)]
    for i in range(BS):
        for j in range(SEQ):
            s2[i, j] = dec[i].read(UNIFORM, V)
    model, opt = setup()
    cumul = np.zeros((BS, V + 1), dtype=np.uint64)
    for t in range(iters - SEQ):
        model.train()
        batch = torch.from_numpy(s2[:, t:t + SEQ]).cuda().long()
        logits = model.forward(batch)
        prob = F.softmax(logits[:, -1, :], dim=1).detach().cpu().numpy()
        cumul[:, 1:] = np.cumsum(prob * 10000000 + 1, axis=1)
        for i in range(BS):
            s2[i, t + SEQ] = dec[i].read(cumul[i, :], V)
        logits = logits.transpose(1, 2)
        label = torch.from_numpy(s2[:, t + 1:t + SEQ + 1]).cuda()
        loss = F.cross_entropy(logits[:, :, -1], label[:, -1], reduction='mean')
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    td = ac.ArithmeticDecoder(32, ac.BitInputStream(io.BytesIO(streams[BS])))
    tail = [td.read(UNIFORM, V) for _ in range(n - l)]
    out = s2.reshape(-1).astype(np.uint8).tobytes() + bytes(tail)
    open(dst, 'wb').write(out)
    return {}


if __name__ == '__main__':
    cmd, src, dst = sys.argv[1:4]
    t = time.perf_counter()
    info = (compress if cmd == 'c' else decompress)(src, dst)
    print(json.dumps(dict(ms=(time.perf_counter() - t) * 1000, **info)))
