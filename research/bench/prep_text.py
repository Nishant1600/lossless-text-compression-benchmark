"""Build the text benchmark (TextBench) from the downloaded corpora.

Writes research/data/text/bench/<name> (full files) and research/data/text/slices/<name>
(first ~1 MB of each, cut at a line boundary so it stays valid UTF-8).
"""
import glob
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(ROOT, 'data', 'text')
SCRATCH_CORPUS = sys.argv[1]  # folder holding the canterbury/ and silesia/ corpora downloaded earlier
OUT = os.path.join(DATA, 'bench')
SLICES = os.path.join(DATA, 'slices')
os.makedirs(OUT, exist_ok=True)
os.makedirs(SLICES, exist_ok=True)


def leipzig(folder):
    path = glob.glob(os.path.join(DATA, folder, '*', '*-sentences.txt'))[0]
    lines = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            lines.append(line.split('\t', 1)[1] if '\t' in line else line)
    return ''.join(lines).encode('utf-8')


def code_corpus():
    tp = os.path.join(ROOT, 'third_party')
    files = sorted(glob.glob(os.path.join(tp, 'paq8px', 'src', '**', '*.[ch]pp'), recursive=True))
    files += sorted(glob.glob(os.path.join(tp, 'libbsc', 'libbsc', '**', '*.[ch]*'), recursive=True))
    files += [os.path.join(tp, 'zpaq', 'libzpaq.cpp'), os.path.join(tp, 'zpaq', 'zpaq.cpp')]
    parts = []
    for p in files:
        with open(p, 'rb') as f:
            parts.append(f.read())
    return b''.join(parts)


def read(p, n=None):
    with open(p, 'rb') as f:
        return f.read(n) if n else f.read()


SOURCES = {
    'enwik8_10M': lambda: read(os.path.join(DATA, 'enwik8', 'enwik8'), 10_000_000),
    'en_sentences': lambda: leipzig('leipzig_eng'),
    'hi_sentences': lambda: leipzig('leipzig_hin'),
    'zh_sentences': lambda: leipzig('leipzig_zho'),
    'dickens': lambda: read(os.path.join(SCRATCH_CORPUS, 'silesia', 'dickens')),
    'xml': lambda: read(os.path.join(SCRATCH_CORPUS, 'silesia', 'xml')),
    'source_cpp': code_corpus,
}

for name, make in SOURCES.items():
    data = make()
    with open(os.path.join(OUT, name), 'wb') as f:
        f.write(data)
    cut = data[:1_000_000]
    if len(data) > 1_000_000:
        cut = cut[:cut.rfind(b'\n') + 1]
    try:
        cut.decode('utf-8')
        utf8 = 'valid UTF-8'
    except UnicodeDecodeError:
        utf8 = 'not UTF-8 (language models code such lines as escaped bytes)'
    with open(os.path.join(SLICES, name), 'wb') as f:
        f.write(cut)
    print(f'{name:14} {len(data):>11,} B   slice {len(cut):>9,} B   {utf8}')
