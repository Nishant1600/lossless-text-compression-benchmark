"""Benchmark harness: compress, decompress, verify, and log one JSON line per (file, codec).

    python -I run_bench.py --results text.jsonl --codecs gzip,xz --set full FILE [FILE ...]

Every codec runs in a child process. The parent polls the child's resident memory (including its own
children) and measures wall time; codecs that report their own in-process time ("ms") also have that
recorded, which excludes interpreter start-up. A result is only written after the restored bytes
match the original (SHA-256). Existing (set, file, codec) rows are skipped, so runs can be resumed.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
TP = os.path.join(ROOT, 'third_party')
TMP = os.path.join(ROOT, 'tmp')
PY = sys.executable
NODE = shutil.which('node')
TORCH_LIB = os.path.join(os.path.dirname(os.__file__), 'site-packages', 'torch', 'lib')

BSC = os.path.join(TP, 'libbsc', 'build', 'bsc.exe')
ZPAQ = os.path.join(TP, 'zpaq', 'zpaq.exe')
PAQ = os.path.join(TP, 'paq8px', 'build_mingw', 'paq8px.exe')
NNCP = os.path.join(TP, 'nncp', 'nncp.exe')

# family: E = entropy coder only, D = dictionary / sorting, M = context model / neural, L = pretrained language model
CODECS = {
    'huffman':     dict(family='E', kind='js', label='Huffman (order-0)'),
    'ext-huffman': dict(family='E', kind='js', label='Extended Huffman'),
    'eehc':        dict(family='E', kind='js', label='EEHC'),
    'dyn-huffman': dict(family='E', kind='py', label='Blockwise dynamic Huffman'),
    'rans0':       dict(family='E', kind='py', label='rANS (order-0)'),
    'gzip':        dict(family='D', kind='py', label='gzip -9'),
    'bzip2':       dict(family='D', kind='py', label='bzip2 -9'),
    'xz':          dict(family='D', kind='py', label='xz -9e'),
    'brotli':      dict(family='D', kind='py', label='Brotli 11'),
    'zstd':        dict(family='D', kind='py', label='zstd 19 --long'),
    'bsc':         dict(family='D', kind='cli', label='bsc (BWT)'),
    'ppmd':        dict(family='M', kind='py', label='PPMd var. H (order 6)'),
    'zpaq':        dict(family='M', kind='zpaq', label='zpaq -m5'),
    'cm':          dict(family='M', kind='js', label='CM (this project)'),
    'paq8px':      dict(family='M', kind='cli', label='paq8px -9'),
    'nncp':        dict(family='M', kind='cli', label='NNCP (CPU)'),
    'dzip-bs':     dict(family='M', kind='dzip', label='DZip bootstrap only (DeepZip-style)'),
    'trace':       dict(family='M', kind='trace', label='TRACE'),
    'dzip':        dict(family='M', kind='dzip', label='DZip (combined)'),
    'llm-gpt2':    dict(family='L', kind='llm', model='openai-community/gpt2', label='GPT-2 124M + AC'),
    'llm-qwen0.5b': dict(family='L', kind='llm', model='Qwen/Qwen2.5-0.5B', label='Qwen2.5-0.5B + AC'),
    'llm-qwen1.5b': dict(family='L', kind='llm', model='Qwen/Qwen2.5-1.5B', label='Qwen2.5-1.5B + AC', batch=32),
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, cwd=None, env=None):
    """Run a command, returning (stdout, wall seconds, peak RSS in MB over the process tree)."""
    # Output goes to temporary files, not pipes: a pipe that is not read while polling fills up and blocks
    # chatty programs such as paq8px.
    with tempfile.TemporaryFile() as fo, tempfile.TemporaryFile() as fe:
        t = time.perf_counter()
        p = psutil.Popen(cmd, cwd=cwd, env=env, stdout=fo, stderr=fe)
        peak = 0
        while p.poll() is None:
            try:
                rss = p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
                peak = max(peak, rss)
            except psutil.Error:
                pass
            time.sleep(0.02)
        wall = time.perf_counter() - t
        fo.seek(0)
        fe.seek(0)
        out, err = fo.read(), fe.read()
    if p.returncode != 0:
        raise RuntimeError(f'{cmd[0]} exited {p.returncode}: {err.decode(errors="replace")[-600:]}')
    return out.decode(errors='replace'), wall, peak / 2**20


def last_json(text):
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith('{'):
            return json.loads(line)
    return {}


def commands(codec, src, packed, restored, work):
    kind = CODECS[codec]['kind']
    if kind == 'js':
        js = os.path.join(HERE, 'js_codec.mjs')
        return [NODE, js, 'c', codec, src, packed], [NODE, js, 'd', codec, packed, restored], None
    if kind == 'py':
        pc = os.path.join(HERE, 'pycodecs.py')
        return [PY, '-I', pc, 'c', codec, src, packed], [PY, '-I', pc, 'd', codec, packed, restored], None
    if codec == 'bsc':
        return [BSC, 'e', src, packed, '-b64e2', '-T'], [BSC, 'd', packed, restored, '-T'], None
    if codec == 'paq8px':
        return [PAQ, '-9', src, packed], [PAQ, '-d', packed, restored], None
    if kind == 'llm':
        llm = os.path.join(HERE, 'llm_ac.py')
        extra = ['--batch', str(CODECS[codec].get('batch', 64))]
        m = CODECS[codec]['model']
        return [PY, '-I', llm, 'c', m, src, packed, *extra], [PY, '-I', llm, 'd', m, packed, restored, *extra], None
    if kind == 'dzip':
        dz = os.path.join(HERE, 'dzip_run.py')
        mode = 'bs' if codec == 'dzip-bs' else 'com'
        return [PY, '-I', dz, 'c', mode, src, packed], [PY, '-I', dz, 'd', mode, packed, restored], None
    if kind == 'trace':
        tr = os.path.join(HERE, 'trace_run.py')
        return [PY, '-I', tr, 'c', src, packed], [PY, '-I', tr, 'd', packed, restored], None
    if codec == 'nncp':
        return [NNCP, '-q', 'c', src, packed], [NNCP, '-q', 'd', packed, restored], None
    raise KeyError(codec)


HEAVY = {'paq8px', 'nncp', 'trace', 'dzip', 'dzip-bs', 'llm-gpt2', 'llm-qwen0.5b', 'llm-qwen1.5b'}


class HeavyLock:
    """Only one memory-heavy codec (paq8px, NNCP, the neural and language-model coders) runs at a time
    across all harness processes; the laptop's 16 GB cannot hold two of them."""
    def __init__(self, codec):
        self.on = codec in HEAVY

    def __enter__(self):
        if self.on:
            import msvcrt
            os.makedirs(TMP, exist_ok=True)
            self.f = open(os.path.join(TMP, 'heavy.lock'), 'a+')
            while True:
                try:
                    self.f.seek(0)
                    msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(5)
        return self

    def __exit__(self, *exc):
        if self.on:
            import msvcrt
            self.f.seek(0)
            msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
            self.f.close()


def bench_one(codec, src, keep=False):
    with HeavyLock(codec):
        return _bench_one(codec, src, keep)


def _bench_one(codec, src, keep=False):
    work = os.path.join(TMP, f'{codec}-{os.getpid()}-{time.time_ns()}')
    os.makedirs(work)
    packed = os.path.join(work, 'packed.bin')
    restored = os.path.join(work, 'restored.bin')
    try:
        if CODECS[codec]['kind'] == 'zpaq':
            # zpaq is an archiver: add the file by its base name, extract into an empty folder.
            name = os.path.basename(src)
            stage = os.path.join(work, 'in')
            os.makedirs(stage)
            shutil.copyfile(src, os.path.join(stage, name))
            arc = os.path.join(work, 'a.zpaq')
            out_c, wc, mc = run([ZPAQ, 'a', arc, name, '-m5'], cwd=stage)
            os.makedirs(os.path.join(work, 'x'))
            out_d, wd, md = run([ZPAQ, 'x', arc, '-to', os.path.join(work, 'x')], cwd=stage)
            packed, restored = arc, os.path.join(work, 'x', name)
            info_c = info_d = {}
        else:
            cc, dc, env = commands(codec, src, packed, restored, work)
            out_c, wc, mc = run(cc, env=env)
            out_d, wd, md = run(dc, env=env)
            info_c, info_d = last_json(out_c), last_json(out_d)
        ok = sha(src) == sha(restored)
        row = dict(
            codec=codec, label=CODECS[codec]['label'], family=CODECS[codec]['family'],
            bytes=os.path.getsize(packed), verified=ok,
            comp_s=info_c.get('ms', wc * 1000) / 1000, decomp_s=info_d.get('ms', wd * 1000) / 1000,
            comp_wall_s=wc, decomp_wall_s=wd, peak_mb=max(mc, md),
        )
        for k in ('model_bytes', 'payload', 'side', 'codebook', 'blockSize', 'symbols', 'raw_bytes', 'raw_pieces', 'pieces', 'tokens', 'model_ms'):
            if k in info_c:
                row[k] = info_c[k]
        if not ok:
            raise RuntimeError(f'{codec}: restored file differs from the original')
        return row
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--codecs', required=True)
    ap.add_argument('--set', default='full')
    ap.add_argument('files', nargs='+')
    a = ap.parse_args()
    path = a.results if os.path.isabs(a.results) else os.path.join(ROOT, 'results', a.results)
    done = set()
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                if 'bytes' in r:  # failed rows are retried
                    done.add((r['set'], r['file'], r['codec']))
    for src in a.files:
        src = os.path.abspath(src)
        name = os.path.basename(src)
        for codec in a.codecs.split(','):
            if (a.set, name, codec) in done:
                continue
            try:
                row = bench_one(codec, src)
            except Exception as e:  # keep going; the failure is logged for the paper's notes
                row = dict(codec=codec, error=str(e)[-400:])
            row = dict(set=a.set, file=name, size=os.path.getsize(src), **row)
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(row) + '\n')
            msg = f"{a.set:6} {name:28} {codec:12} " + (
                f"{row['bytes']:>12,} B  {row['bytes'] * 8 / row['size']:.3f} bpb  "
                f"{row['comp_s']:.1f}s/{row['decomp_s']:.1f}s  {row['peak_mb']:.0f} MB"
                if 'bytes' in row else 'ERROR ' + row['error'][:200])
            print(msg, flush=True)


if __name__ == '__main__':
    main()
