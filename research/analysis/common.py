"""Shared helpers for turning benchmark JSONL files into LaTeX tables, figures and numbers."""
import glob
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS = os.path.join(ROOT, 'results')
PAPERS = os.path.join(ROOT, 'papers')

# One colour per method family (the reference categorical palette; validated for colour-vision deficiency).
FAMILY_COLOR = {'E': '#2a78d6', 'D': '#eb6834', 'M': '#1baf7a', 'S': '#eda100', 'L': '#6250d6'}
FAMILY_NAME = {'E': 'Entropy coder only', 'D': 'Dictionary / block-sorting', 'M': 'Context model / neural',
               'S': 'Format-specific', 'L': 'Pretrained language model'}

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e1e0d9',
    'grid.linewidth': 0.6, 'axes.axisbelow': True, 'pdf.fonttype': 42,
})
COL = 3.45   # IEEE column width in inches
FULL = 7.16  # IEEE text width


def load(patterns):
    """Latest successful row per (set, file, codec) from the given JSONL files."""
    rows = []
    for pat in patterns:
        for path in sorted(glob.glob(os.path.join(RESULTS, pat))):
            with open(path, encoding='utf-8') as f:
                for i, line in enumerate(f):
                    r = json.loads(line)
                    if 'bytes' in r:
                        r['_order'] = (path, i)
                        rows.append(r)
    df = pd.DataFrame(rows)
    df = df.sort_values('_order').groupby(['set', 'file', 'codec'], as_index=False).last()
    df['bpb'] = df['bytes'] * 8 / df['size']
    df['pct_smaller'] = (1 - df['bytes'] / df['size']) * 100
    return df.drop(columns=['_order'])


def savefig(fig, paper, name):
    d = os.path.join(PAPERS, paper, 'figures')
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, name + '.pdf'), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def guard_width(text):
    """Shrink a tabular that would be wider than its column (table) or the page (table*); narrower ones are untouched."""
    if '\\resizebox' in text or 'adjustbox' in text:
        return text
    width = '\\textwidth' if 'table*' in text else '\\columnwidth'
    return text.replace('\\begin{tabular}', '\\begin{adjustbox}{max width=' + width + '}\\begin{tabular}', 1).replace(
        '\\end{tabular}', '\\end{tabular}\\end{adjustbox}', 1)


def write_tex(paper, name, text):
    text = guard_width(text)
    d = os.path.join(PAPERS, paper, 'tables')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + '.tex'), 'w', encoding='utf-8') as f:
        f.write(text)


def write_numbers(paper, numbers):
    """Numbers quoted in the prose, as LaTeX macros (\\newcommand{\\NumName}{value})."""
    d = os.path.join(PAPERS, paper)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'numbers.tex'), 'w', encoding='utf-8') as f:
        for k, v in numbers.items():
            f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    with open(os.path.join(d, 'numbers.json'), 'w', encoding='utf-8') as f:
        json.dump(numbers, f, indent=1)


def fmt(x, nd=3):
    return f'{x:.{nd}f}'


def tex_escape(s):
    return (str(s).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%')
            .replace('&', r'\&').replace('#', r'\#').replace('+', '+'))


def hf_model_bytes(repo):
    """Size of a cached Hugging Face model's weight files (what a decoder must hold)."""
    base = os.path.expanduser('~/.cache/huggingface/hub/models--' + repo.replace('/', '--'))
    total = 0
    for p in glob.glob(os.path.join(base, 'snapshots', '*', '*.safetensors')):
        total += os.path.getsize(os.path.realpath(p))
    # sizes from the Hugging Face API, for when the cache has been cleared after the runs
    return total or HF_WEIGHT_BYTES.get(repo, 0)


HF_WEIGHT_BYTES = {'openai-community/gpt2': 548_105_171, 'Qwen/Qwen2.5-0.5B': 988_097_824,
                   'Qwen/Qwen2.5-1.5B': 3_087_467_144}
