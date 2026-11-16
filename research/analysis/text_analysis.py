"""Tables, figures and quoted numbers for paper 1 (text)."""
import math
import os
import os

import numpy as np
import pandas as pd

from common import (COL, FAMILY_COLOR, FAMILY_NAME, FULL, ROOT, hf_model_bytes, load, plt, savefig,
                    tex_escape, write_numbers, write_tex)

P = 'p1_text'
df = load(['text.jsonl', 'text_fix.jsonl', 'text_gpu.jsonl', 'text_nncp.jsonl', 'text_dzip.jsonl'])

ORDER = ['huffman', 'dyn-huffman', 'rans0', 'ext-huffman', 'eehc', 'gzip', 'bzip2', 'zstd', 'xz', 'brotli', 'bsc',
         'ppmd', 'zpaq', 'cm', 'paq8px', 'dzip', 'dzip-bs', 'trace', 'nncp', 'llm-gpt2', 'llm-qwen0.5b',
         'llm-qwen1.5b']
SHORT = {'huffman': 'Huffman', 'dyn-huffman': 'Dyn.\\ Huffman', 'rans0': 'rANS-0', 'ext-huffman': 'Ext.\\ Huffman',
         'eehc': 'EEHC', 'gzip': 'gzip', 'bzip2': 'bzip2', 'zstd': 'zstd', 'xz': 'xz', 'brotli': 'Brotli',
         'bsc': 'bsc', 'ppmd': 'PPMd', 'zpaq': 'zpaq', 'cm': 'CM', 'paq8px': 'paq8px', 'dzip': 'DZip',
         'dzip-bs': 'DZip-bs', 'trace': 'TRACE', 'nncp': 'NNCP', 'llm-gpt2': 'GPT-2', 'llm-qwen0.5b': 'Qwen-0.5B',
         'llm-qwen1.5b': 'Qwen-1.5B'}
FAMILY = df.drop_duplicates('codec').set_index('codec')['family'].to_dict()
FILES = ['en_sentences', 'hi_sentences', 'zh_sentences', 'enwik8_10M', 'dickens', 'xml', 'source_cpp']
FLABEL = {'en_sentences': 'English', 'hi_sentences': 'Hindi', 'zh_sentences': 'Chinese', 'enwik8_10M': 'enwik8',
          'dickens': 'dickens', 'xml': 'xml', 'source_cpp': 'C/C++', 'canterbury': 'Canterbury'}
MODELS = {'llm-gpt2': 'openai-community/gpt2', 'llm-qwen0.5b': 'Qwen/Qwen2.5-0.5B', 'llm-qwen1.5b': 'Qwen/Qwen2.5-1.5B'}
BROTLI_DICT = 122_784  # bytes in Brotli's built-in static dictionary (RFC 7932, Appendix A)

full = df[df.set == 'full']
CANT = ['alice29.txt', 'asyoulik.txt', 'lcet10.txt', 'plrabn12.txt', 'cp.html', 'fields.c', 'grammar.lsp', 'xargs.1']
_c = full[full.file.isin(CANT)].groupby('codec').agg(bytes=('bytes', 'sum'), size=('size', 'sum'),
                                                     n=('file', 'nunique')).reset_index()
_c = _c[_c.n == len(CANT)].assign(file='canterbury', set='full')
_c['bpb'] = _c.bytes * 8 / _c['size']
full = pd.concat([full[~full.file.isin(CANT)], _c[['set', 'file', 'codec', 'bytes', 'size', 'bpb']]], ignore_index=True)
sl = df[df.set == 'slice']
numbers = {}


def order0(path):
    b = np.frombuffer(open(path, 'rb').read(), dtype=np.uint8)
    c = np.bincount(b, minlength=256).astype(float)
    p = c[c > 0] / len(b)
    return float(-(p * np.log2(p)).sum())


# ---------------------------------------------------------------- corpus table
rows = []
for f in FILES:
    path = os.path.join(ROOT, 'data', 'text', 'bench', f)
    data = open(path, 'rb').read()
    try:
        chars = len(data.decode('utf-8'))
    except UnicodeDecodeError:
        chars = len(data)
    rows.append((FLABEL[f], len(data), len(data) / chars, order0(path)))
desc = {'English': 'Wikipedia sentences$^a$', 'Hindi': 'Wikipedia sentences$^a$',
        'Chinese': 'Wikipedia sentences$^a$', 'enwik8': 'enwik8, first 10\\,MB',
        'dickens': 'Silesia: novels', 'xml': 'Silesia: XML', 'C/C++': 'paq8px, bsc, zpaq code'}
t = ['\\begin{table}[t]\\centering\\caption{TextBench. B/ch: UTF-8 bytes per character; '
     '$H_0$: order-0 byte entropy in bits per byte.}\\label{tab:corpus}\\footnotesize\\setlength{\\tabcolsep}{4pt}',
     '\\begin{tabular}{@{}llrrr@{}}\\toprule', 'File & Content & MB & B/ch & $H_0$\\\\\\midrule']
for name, n, bpc, h in rows:
    t.append(f'{name} & {desc[name]} & {n / 1e6:.2f} & {bpc:.2f} & {h:.2f}\\\\')
t.append('Canterbury & 8 text files & 2.18 & 1.00 & --\\\\')
t += ['\\bottomrule\\multicolumn{5}{@{}l}{$^a$\\,Leipzig Corpora Collection.}\\end{tabular}\\end{table}']
write_tex(P, 'corpus', '\n'.join(t))

# ---------------------------------------------------------------- methods table
METH = [
    ('E', 'Huffman', 'order-0, binary tree, table stored', 'project JS'),
    ('E', 'Dyn.\\ Huffman', 'tree rebuilt per 4\\,KiB, no table~\\cite{klein2024dynhuff}', 'ours (Python)'),
    ('E', 'rANS-0', 'order-0 static ANS~\\cite{duda2013ans}', '\\texttt{constriction}'),
    ('E', 'Ext.\\ Huffman', 'binary tree over 4/8/16-B blocks', 'project JS'),
    ('E', 'EEHC', 'base-8 tree over 4/8/16-B blocks~\\cite{yeful2026eehc}', 'project JS'),
    ('D', 'gzip / bzip2 / xz', 'level 9 / 9 / 9e', 'Python stdlib'),
    ('D', 'zstd', 'level 19, long mode, 128\\,MiB window', '\\texttt{zstandard}'),
    ('D', 'Brotli', 'quality 11, 16\\,MiB window', '\\texttt{brotli}'),
    ('D', 'bsc', 'BWT + QLFC, 64\\,MB blocks, single thread', 'libbsc 3.3'),
    ('M', 'PPMd', 'variant H, order 8, 256\\,MiB', '\\texttt{pyppmd}'),
    ('M', 'zpaq', 'method 5', 'zpaq 7.15'),
    ('M', 'CM', 'context mixing, 2 mixers + SSE', 'project JS'),
    ('M', 'paq8px', 'level 9 (4.3\\,GB)', 'v217'),
    ('M', 'DZip / DZip-bs', 'biGRU, 8 epochs; combined / bootstrap only', 'authors\' code'),
    ('M', 'TRACE', '1-layer transformer, online', 'authors\' code'),
    ('M', 'NNCP', 'transformer, online, default profile, CPU', '2024-06-05'),
    ('L', 'GPT-2 / Qwen-0.5B / Qwen-1.5B', 'pretrained LM + arithmetic coding', 'ours, HF weights'),
]
t = ['\\begin{table}[t]\\centering\\caption{Methods. Families: E = entropy coder only, D = dictionary or block '
     'sorting, M = context model or adaptive neural network, L = pretrained language model.}\\label{tab:methods}'
     '\\footnotesize\\setlength{\\tabcolsep}{3pt}', '\\begin{tabular}{@{}c>{\\raggedright\\arraybackslash}p{1.55cm}'
     '>{\\raggedright\\arraybackslash}p{3.65cm}>{\\raggedright\\arraybackslash}p{2.05cm}@{}}\\toprule',
     'Fam. & Method & Setting & Implementation\\\\\\midrule']
for fam, m, s, impl in METH:
    t.append(f'{fam} & {m} & {s} & {impl}\\\\')
t += ['\\bottomrule\\end{tabular}\\end{table}']
write_tex(P, 'methods', '\n'.join(t))
numbers['NumMethods'] = len([c for c in ORDER if c in set(df.codec)])


# ---------------------------------------------------------------- result matrices
def matrix(sub, files, caption, label, codecs):
    piv = sub.pivot_table(index='codec', columns='file', values='bpb')
    piv = piv.reindex([c for c in codecs if c in piv.index])
    cols = [f for f in files if f in piv.columns]
    best = piv[cols].min()
    t = [f'\\begin{{table*}}[t]\\centering\\caption{{{caption}}}\\label{{{label}}}\\small',
         '\\begin{tabular}{@{}cl' + 'r' * len(cols) + '@{}}\\toprule',
         'Fam. & Method & ' + ' & '.join(FLABEL[c] for c in cols) + '\\\\\\midrule']
    prev = None
    for c in piv.index:
        fam = FAMILY.get(c, '')
        if prev is not None and fam != prev:
            t.append('\\midrule')
        prev = fam
        cells = []
        for f in cols:
            v = piv.loc[c, f]
            if pd.isna(v):
                cells.append('\\na')
            else:
                s = f'{v:.3f}'
                cells.append(f'\\best{{{s}}}' if abs(v - best[f]) < 1e-9 else s)
        t.append(f'{fam} & {SHORT[c]} & ' + ' & '.join(cells) + '\\\\')
    t += ['\\bottomrule\\end{tabular}\\end{table*}']
    return '\n'.join(t)


write_tex(P, 'full_results', matrix(full, FILES + ['canterbury'], 'Compressed size in bits per byte on the full TextBench files '
          '(self-contained output, lower is better; best per column in bold; Canterbury: total of its eight '
          'text files). paq8px ran on xml and Canterbury only; neural and language-model methods are reported on '
          '1\\,MB slices in \\cref{tab:slice}.', 'tab:full', ORDER))
write_tex(P, 'slice_results', matrix(sl, FILES, 'Bits per byte on the 1\\,MB slices, all methods (self-contained '
          'output; language-model rows exclude the model weights, see \\cref{tab:cost}). \\na: not run.',
          'tab:slice', ORDER))

# ---------------------------------------------------------------- headline numbers
def bpb(codec, file, sub=sl):
    r = sub[(sub.codec == codec) & (sub.file == file)]
    return float(r.bpb.iloc[0]) if len(r) else math.nan


classical = [c for c in ORDER if FAMILY.get(c) in ('E', 'D', 'M') and c not in ('dzip', 'dzip-bs', 'trace', 'nncp')]
llms = [c for c in MODELS if c in set(sl.codec)]
best_gain = None
for f in FILES:
    cl = sl[(sl.file == f) & (sl.codec.isin(classical))]
    lm = sl[(sl.file == f) & (sl.codec.isin(llms))]
    if len(cl) and len(lm):
        bc, bl = cl.loc[cl.bpb.idxmin()], lm.loc[lm.bpb.idxmin()]
        gain = (1 - bl.bpb / bc.bpb) * 100
        if best_gain is None or bl.bpb < best_gain[1]:
            best_gain = (f, bl.bpb, bl.codec, bc.codec, bc.bpb, gain)
if best_gain:
    f, v, c, bc, bv, g = best_gain
    numbers.update(NumBestLLMfile=FLABEL[f], NumBestLLMbpb=f'{v:.3f}', NumBestLLMcodec=SHORT[c],
                   NumBestLLMgain=f'{g:.0f}\\%', NumBestClassicalCodec=SHORT[bc], NumBestClassicalbpb=f'{bv:.3f}')

import text_cost  # noqa: E402

print(text_cost.run(globals()))

# every value as a macro: \NumS<codec><file> (slices) and \NumF<codec><file> (full files)
CK = {'huffman': 'Huf', 'dyn-huffman': 'Dyn', 'rans0': 'Rans', 'ext-huffman': 'Ext', 'eehc': 'Eehc', 'gzip': 'Gzip',
      'bzip2': 'Bzip', 'zstd': 'Zstd', 'xz': 'Xz', 'brotli': 'Brotli', 'bsc': 'Bsc', 'ppmd': 'Ppmd', 'zpaq': 'Zpaq',
      'cm': 'Cm', 'paq8px': 'Paq', 'trace': 'Trace', 'nncp': 'Nncp', 'dzip': 'Dzip', 'dzip-bs': 'Dzipbs',
      'llm-gpt2': 'Gpt', 'llm-qwen0.5b': 'Qs', 'llm-qwen1.5b': 'Ql'}
FK = {'en_sentences': 'En', 'hi_sentences': 'Hi', 'zh_sentences': 'Zh', 'enwik8_10M': 'Wk', 'dickens': 'Dk',
      'xml': 'Xml', 'source_cpp': 'Src', 'canterbury': 'Cant'}
extra = {}
for tag, sub in (('S', sl), ('F', full)):
    for _, r in sub.iterrows():
        if r.codec in CK and r.file in FK:
            extra[f'Num{tag}{CK[r.codec]}{FK[r.file]}'] = f'{r.bpb:.3f}'
numbers.update(extra)
from common import write_numbers  # noqa: E402
write_numbers(P, numbers)

# derived numbers quoted in the results
g = {(r.codec, r.file): r for _, r in sl.iterrows()}
def tokshare(f):
    a, b = g.get(('llm-gpt2', f)), g.get(('llm-qwen0.5b', f))
    if a is None or b is None:
        return None
    tpb_a, tpb_b = a.tokens / a['size'], b.tokens / b['size']
    ra, rb = a.payload / a['size'], b.payload / b['size']
    return math.log(tpb_a / tpb_b) / math.log(ra / rb) * 100
for f, k in (('hi_sentences', 'Hi'), ('zh_sentences', 'Zh'), ('en_sentences', 'En'), ('enwik8_10M', 'Wk')):
    v = tokshare(f)
    if v is not None:
        numbers[f'NumTokShare{k}'] = f'{v:.0f}\%'
for f, k in (('en_sentences', 'En'), ('hi_sentences', 'Hi')):
    a, b = g.get(('llm-qwen0.5b', f)), g.get(('llm-qwen1.5b', f))
    if a is not None and b is not None:
        numbers[f'NumScale{k}'] = f'{(1 - b.payload / a.payload) * 100:.0f}\%'
for f, k in (('en_sentences', 'En'), ('hi_sentences', 'Hi'), ('zh_sentences', 'Zh')):
    r = g.get(('llm-qwen0.5b', f))
    if r is not None:
        data = open(os.path.join(ROOT, 'data', 'text', 'slices', f), 'rb').read()
        numbers[f'NumQs{k}bpcV'] = f'{r.payload * 8 / len(data.decode("utf-8")):.2f}'
ee = full[(full.codec == 'eehc') & (full.file == 'en_sentences')]
if len(ee):
    numbers['NumEEHCpayloadEn'] = f'{ee.payload.iloc[0] * 8 / ee["size"].iloc[0]:.2f}'
# Huffman vs order-0 entropy, dynamic vs static, base-8 vs binary (full files except Canterbury aggregate)
h0 = {name: h for name, _, _, h in rows}
inv = {v: k for k, v in FLABEL.items()}
gap = []
for name, h in h0.items():
    f = inv[name]
    hb = full[(full.codec == 'huffman') & (full.file == f)].bpb
    if len(hb):
        gap.append((hb.iloc[0] - h) / h * 100)
numbers['NumHuffOverHzeroMax'] = f'{max(gap):.1f}'
def ratio(a, b):
    x = full[full.codec == a].set_index('file').bpb
    y = full[full.codec == b].set_index('file').bpb
    j = x.index.intersection(y.index)
    return ((x[j] / y[j] - 1) * 100)
numbers['NumDynOverStaticMax'] = f'{ratio("dyn-huffman", "huffman").max():.2f}'
numbers['NumBaseEightMax'] = f'{ratio("eehc", "ext-huffman").max():.2f}'
numbers['NumRansUnderHuff'] = f'{-ratio("rans0", "huffman").mean():.2f}'
write_numbers(P, numbers)
