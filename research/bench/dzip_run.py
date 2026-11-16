"""DZip (Goyal et al., DCC 2021) as one compress / decompress command each.

    python -I dzip_run.py c|d com|bs INPUT OUTPUT

Runs the authors' scripts from third_party/Dzip-torch/coding-gpu in a private working folder:
run.py (symbol map), train_bootstrap.py (8 epochs, 64 timesteps), then compress_adaptive.py ("com",
the combined model, DZip's default) or compress_bootstrap.py ("bs", bootstrap model only, which is
DeepZip-style: a model trained on the file and stored with it). As in the paper, the bootstrap weights
are part of the output, compressed with BSC (-b128e2). The container holds the symbol map, the BSC'd
weights and the coded stream, so the decoder needs nothing else.

Changes to the authors' scripts, applied to private copies: DataLoader workers set to 0 (Windows
cannot fork module-level scripts), the removed `verbose` argument of ReduceLROnPlateau dropped, nll_loss targets cast to int64,
and np.asscalar restored as a.item() for NumPy 2.
"""
import glob
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'third_party', 'Dzip-torch', 'coding-gpu')
BSC = os.path.join(HERE, '..', 'third_party', 'libbsc', 'build', 'bsc.exe')
PY = sys.executable
MAGIC = b'DZIPC1'


def workdir():
    d = tempfile.mkdtemp(prefix='dzip-', dir=os.path.join(HERE, '..', 'tmp'))
    for p in glob.glob(os.path.join(SRC, '*.py')):
        s = open(p, encoding='utf-8').read()
        s = s.replace("'num_workers': 4", "'num_workers': 0").replace(', verbose=True', '').replace('F.nll_loss(pred, target)', 'F.nll_loss(pred, target.long())')
        if os.path.basename(p) == 'arithmeticcoding_fast.py':
            s = 'import numpy as np\nnp.asscalar = lambda a: a.item()\n' + s
        open(os.path.join(d, os.path.basename(p)), 'w', encoding='utf-8').write(s)
    return d


def sh(args, cwd):
    r = subprocess.run([PY, '-E', '-s', *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(' '.join(args[:1]) + ': ' + r.stderr[-800:])


def pack(parts):
    return MAGIC + b''.join(struct.pack('<I', len(p)) + p for p in parts)


def unpack(b):
    assert b[:6] == MAGIC
    pos, parts = 6, []
    while pos < len(b):
        (n,) = struct.unpack('<I', b[pos:pos + 4])
        parts.append(b[pos + 4:pos + 4 + n])
        pos += 4 + n
    return parts


def compress(mode, src, dst):
    d = workdir()
    try:
        shutil.copyfile(src, os.path.join(d, 'data.bin'))
        sh(['run.py', '--file_name', 'data.bin'], d)
        sh(['train_bootstrap.py', '--file_name', 'data', '--epochs', '8', '--timesteps', '64',
            '--model_weights_path', 'bstrap'], d)
        if mode == 'com':
            sh(['compress_adaptive.py', '--file_name', 'data', '--bs', '64', '--timesteps', '64',
                '--output', 'comp', '--model_weights_path', 'bstrap'], d)
        else:
            sh(['compress_bootstrap.py', '--file_name', 'data', '--output', 'comp',
                '--model_weights_path', 'bstrap'], d)
        subprocess.run([BSC, 'e', 'bstrap', 'bstrap.bsc', '-b128e2'], cwd=d, check=True, capture_output=True)
        params = open(os.path.join(d, 'comp.params'), 'rb').read()
        model = open(os.path.join(d, 'bstrap.bsc'), 'rb').read()
        stream = open(os.path.join(d, 'comp.combined'), 'rb').read()
        open(dst, 'wb').write(pack([mode.encode(), params, model, stream]))
        return dict(payload=len(stream), side=len(params) + len(model) + 6 + 16, model_bytes=len(model))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def decompress(mode, src, dst):
    d = workdir()
    try:
        m, params, model, stream = unpack(open(src, 'rb').read())
        open(os.path.join(d, 'comp.params'), 'wb').write(params)
        open(os.path.join(d, 'bstrap.bsc'), 'wb').write(model)
        open(os.path.join(d, 'comp.combined'), 'wb').write(stream)
        subprocess.run([BSC, 'd', 'bstrap.bsc', 'bstrap'], cwd=d, check=True, capture_output=True)
        script = 'decompress_adaptive.py' if m == b'com' else 'decompress_bootstrap.py'
        sh([script, '--file_name', 'comp', '--output', 'out.bin', '--model_weights_path', 'bstrap'], d)
        shutil.copyfile(os.path.join(d, 'out.bin'), dst)
        return {}
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    cmd, mode, src, dst = sys.argv[1:5]
    t = time.perf_counter()
    info = (compress if cmd == 'c' else decompress)(mode, os.path.abspath(src), os.path.abspath(dst))
    print(json.dumps(dict(ms=(time.perf_counter() - t) * 1000, **info)))
