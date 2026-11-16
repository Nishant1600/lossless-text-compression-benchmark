"""Second half of the paper-1 analysis: cost levels, tokeniser decomposition, figures. Imported by
text_analysis.py after the result tables are built (shares its namespace through `ns`)."""
import math
import os

import numpy as np
import pandas as pd

from common import COL, FAMILY_COLOR, FAMILY_NAME, FULL, ROOT, hf_model_bytes, plt, savefig, write_numbers, write_tex


def human(nbytes):
    for unit, sc in (('TB', 1e12), ('GB', 1e9), ('MB', 1e6), ('KB', 1e3)):
        if nbytes >= sc:
            return f'{nbytes / sc:.1f}\\,{unit}'
    return f'{nbytes:.0f}\\,B'


def run(ns):
    P, sl, full, ORDER, FILES, SHORT, FLABEL, FAMILY, MODELS, numbers = (
        ns['P'], ns['sl'], ns['full'], ns['ORDER'], ns['FILES'], ns['SHORT'], ns['FLABEL'], ns['FAMILY'],
        ns['MODELS'], ns['numbers'])
    BROTLI_DICT, bpb = ns['BROTLI_DICT'], ns['bpb']
    llms = [c for c in MODELS if c in set(sl.codec)]

    # ------------------------------------------------------------ cost levels (on the four slices all LMs ran on)
    LANG = ['en_sentences', 'hi_sentences', 'zh_sentences', 'enwik8_10M']
    rows = []
    for c in ORDER:
        for f in LANG:
            r = sl[(sl.codec == c) & (sl.file == f)]
            if not len(r):
                continue
            r = r.iloc[0]
            pl = r.get('payload', np.nan)
            payload = r.bytes if pd.isna(pl) else pl
            K = hf_model_bytes(MODELS[c]) if c in MODELS else (BROTLI_DICT if c == 'brotli' else 0)
            rows.append(dict(codec=c, file=f, N=r['size'], P=payload, F=r.bytes, K=K))
    cost = pd.DataFrame(rows)
    cost['rP'] = cost.P * 8 / cost.N
    cost['rF'] = cost.F * 8 / cost.N
    refs = {k: cost[cost.codec == k].set_index('file').rF for k in ('xz', 'paq8px')}
    lines = ['\\begin{table}[t]\\centering\\caption{Three cost levels on the English, Hindi, Chinese and enwik8 slices '
             '(bits per byte, mean over the slices; Qwen-1.5B: English and Hindi only): payload $r_P$, self-contained '
             'file $r_F$, and $r_K$ with the pretrained knowledge charged once on 1\\,GB or 1\\,TB of input. $N^\\ast$: '
             'input size at which that knowledge is paid back against xz and against paq8px on the same slices.}'
             '\\label{tab:cost}\\footnotesize\\setlength{\\tabcolsep}{3.5pt}',
             '\\begin{tabular}{@{}lrrrrrr@{}}\\toprule',
             'Method & $r_P$ & $r_F$ & $r_K$@1GB & $r_K$@1TB & $N^\\ast_{\\mathrm{xz}}$ & $N^\\ast_{\\mathrm{paq8px}}$\\\\\\midrule']
    breakeven = {}
    for c in ORDER:
        sub = cost[cost.codec == c]
        if not len(sub):
            continue
        K = sub.K.iloc[0]
        rP, rF = sub.rP.mean(), sub.rF.mean()
        rk = [rF + 8 * K / n for n in (1e9, 1e12)]
        ns = []
        for ref_name, ref in refs.items():
            d = np.nanmean(ref.reindex(sub.file).values - sub.rF.values)
            if K == 0:
                ns.append('--')
            elif d > 0:
                ns.append(human(8 * K / d))
                breakeven[c + ('' if ref_name == 'xz' else '|paq')] = 8 * K / d
            else:
                ns.append('never')
        lines.append(f'{SHORT[c]} & {rP:.3f} & {rF:.3f} & {rk[0]:.3f} & {rk[1]:.3f} & {ns[0]} & {ns[1]}\\\\')
    lines += ['\\bottomrule\\end{tabular}\\end{table}']
    write_tex(P, 'cost', '\n'.join(lines))
    keys = {'llm-gpt2': 'Gpt', 'llm-qwen0.5b': 'Small', 'llm-qwen1.5b': 'Large', 'brotli': 'Brotli'}
    for c, n in breakeven.items():
        base, _, tag = c.partition('|')
        numbers['NumBreakEven' + keys.get(base, 'Other') + ('Paq' if tag else '')] = human(n)
    for c, key in (('llm-gpt2', 'Gpt'), ('llm-qwen0.5b', 'QwenSmall'), ('llm-qwen1.5b', 'QwenLarge')):
        numbers[f'Num{key}GB'] = f'{hf_model_bytes(MODELS[c]) / 1e9:.2f}'

    # ------------------------------------------------------------ EEHC payload vs file
    ee = full[full.codec == 'eehc'].copy()
    if len(ee):
        ee['rP'] = ee.payload * 8 / ee['size']
        hf = full[full.codec == 'huffman'].set_index('file').bpb
        numbers['NumEEHCpayloadMean'] = f'{ee.rP.mean():.2f}'
        numbers['NumEEHCfileMean'] = f'{ee.bpb.mean():.2f}'
        numbers['NumEEHCworseThanHuffman'] = str(int((ee.set_index('file').bpb > hf.reindex(ee.file).values).sum()))
        numbers['NumEEHCfiles'] = str(len(ee))
        numbers['NumHuffmanMean'] = f'{hf.reindex(ee.file).mean():.2f}'

    # ------------------------------------------------------------ tokeniser decomposition
    trows = []
    for c in llms:
        for f in ('en_sentences', 'hi_sentences', 'zh_sentences', 'enwik8_10M', 'dickens', 'xml', 'source_cpp'):
            r = sl[(sl.codec == c) & (sl.file == f)]
            if not len(r) or pd.isna(r.iloc[0].get('tokens', np.nan)):
                continue
            r = r.iloc[0]
            data = open(os.path.join(ROOT, 'data', 'text', 'slices', f), 'rb').read()
            chars = len(data.decode('utf-8', errors='replace'))
            trows.append(dict(codec=c, file=f, tpb=r.tokens / r['size'], bpt=r.payload * 8 / r.tokens,
                              rP=r.payload * 8 / r['size'], bpc=r.payload * 8 / chars,
                              bpl=r.payload * 8 / data.count(b'\n'), raw=(r.get('raw_bytes', 0) or 0) / r['size']))
    tok = pd.DataFrame(trows)
    if len(tok):
        lines = ['\\begin{table}[t]\\centering\\caption{Language-model payload rate split into tokens per byte '
                 '(tokeniser) and bits per token (prediction): $r_P=(T/N)(8P/T)$; also bits per character and per '
                 'line (one sentence per line in the Leipzig files). No piece needed the raw fallback.}\\label{tab:tok}\\footnotesize\\setlength{\\tabcolsep}{3.5pt}',
                 '\\begin{tabular}{@{}llrrrrr@{}}\\toprule',
                 'Model & Text & Tok./byte & Bits/tok. & $r_P$ & Bits/char & Bits/line\\\\\\midrule']
        for c in llms:
            for _, r in tok[tok.codec == c].iterrows():
                lines.append(f'{SHORT[c]} & {FLABEL[r.file]} & {r.tpb:.3f} & {r.bpt:.2f} & {r.rP:.3f} & '
                             f'{r.bpc:.2f} & {r.bpl:.0f}\\\\')
            lines.append('\\midrule')
        lines[-1] = '\\bottomrule\\end{tabular}\\end{table}'
        write_tex(P, 'tok', '\n'.join(lines))
        g = tok.set_index(['codec', 'file'])
        for c, key in (('llm-gpt2', 'Gpt'), ('llm-qwen0.5b', 'Qs'), ('llm-qwen1.5b', 'Ql')):
            for f, fk in (('en_sentences', 'En'), ('hi_sentences', 'Hi'), ('zh_sentences', 'Zh'),
                          ('enwik8_10M', 'Wk')):
                if (c, f) in g.index:
                    numbers[f'Num{key}{fk}tpb'] = f'{g.loc[(c, f)].tpb:.3f}'
                    numbers[f'Num{key}{fk}bpt'] = f'{g.loc[(c, f)].bpt:.2f}'
                    numbers[f'Num{key}{fk}rP'] = f'{g.loc[(c, f)].rP:.3f}'

    # ------------------------------------------------------------ figure: languages
    sel = [c for c in ['huffman', 'eehc', 'gzip', 'xz', 'brotli', 'bsc', 'ppmd', 'cm', 'paq8px', 'trace', 'nncp',
                       'llm-gpt2', 'llm-qwen0.5b', 'llm-qwen1.5b'] if c in set(sl.codec)]
    langs = ['en_sentences', 'hi_sentences', 'zh_sentences', 'enwik8_10M']
    fig, ax = plt.subplots(figsize=(FULL, 2.4))
    w = 0.84 / len(sel)
    top = 0
    for k, c in enumerate(sel):
        ys = np.array([bpb(c, f) for f in langs])
        xs = np.arange(len(langs)) + (k - len(sel) / 2 + 0.5) * w
        ax.bar(xs, ys, w * 0.9, color=FAMILY_COLOR.get(FAMILY.get(c), '#888'), edgecolor='white', linewidth=0.3)
        for x, y in zip(xs, ys):
            if not math.isnan(y):
                ax.text(x, y + 0.04, SHORT[c].replace('\\', ''), rotation=90, ha='center', va='bottom', fontsize=4.4)
                top = max(top, y)
    ax.set_xticks(np.arange(len(langs)))
    ax.set_xticklabels([FLABEL[f] + ' (1 MB slice)' for f in langs])
    ax.set_ylabel('bits per byte (lower is better)')
    ax.set_ylim(0, top * 1.3)
    fams = [k for k in ('E', 'D', 'M', 'L') if any(FAMILY.get(c) == k for c in sel)]
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[k]) for k in fams], [FAMILY_NAME[k] for k in fams],
              ncol=len(fams), loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=6, handlelength=1, columnspacing=1.5)
    savefig(fig, P, 'languages')

    # ------------------------------------------------------------ figure: speed vs size
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    agg = sl.groupby('codec').agg(bpb=('bpb', 'mean'), t=('comp_s', 'sum'), n=('size', 'sum')).reset_index()
    agg['mbps'] = agg.n / 1e6 / agg.t
    agg = agg[agg.codec.isin(list(SHORT))]
    for _, r in agg.iterrows():
        ax.scatter(r.mbps, r.bpb, s=16, color=FAMILY_COLOR.get(FAMILY.get(r.codec), '#888'), edgecolor='white',
                   linewidth=0.6, zorder=3)
    ax.set_xscale('log')
    ax.set_xlabel('compression speed on the slices, MB/s (log)')
    ax.set_ylabel('mean bits per byte')
    # Greedy label placement: each label takes the first offset whose box clears the labels already placed.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    from matplotlib.transforms import Bbox
    placed = [Bbox.from_bounds(*(ax.transData.transform((x, y)) - 4), 8, 8) for x, y in zip(agg.mbps, agg.bpb)]
    offsets = [(4, 2, 'left', 'bottom'), (4, -2, 'left', 'top'), (-4, 2, 'right', 'bottom'), (-4, -2, 'right', 'top'),
               (0, 5, 'center', 'bottom'), (0, -5, 'center', 'top'), (4, 8, 'left', 'bottom'),
               (4, -8, 'left', 'top'), (-4, 8, 'right', 'bottom'), (-4, -8, 'right', 'top'),
               (10, 12, 'left', 'bottom'), (10, -12, 'left', 'top'), (-10, 12, 'right', 'bottom'),
               (-10, -12, 'right', 'top'), (14, 0, 'left', 'center'), (-14, 0, 'right', 'center')]
    leader = dict(arrowstyle='-', color='#8a8984', linewidth=0.35, shrinkA=0, shrinkB=2.5)
    # points that nearly coincide get fixed sides: the upper point labelled above, the lower one below
    pref = {'bzip2': (5, 3, 'left', 'bottom'), 'bsc': (5, -3, 'left', 'top'), 'zstd': (5, 3, 'left', 'bottom'),
            'xz': (-5, -3, 'right', 'top'), 'eehc': (5, 3, 'left', 'bottom'), 'ext-huffman': (5, -3, 'left', 'top'),
            'huffman': (0, -5, 'center', 'top'), 'rans0': (0, 5, 'center', 'bottom')}
    for _, r in agg.sort_values('bpb').iterrows():
        for dx, dy, ha, va in ([pref[r.codec]] if r.codec in pref else []) + offsets:
            a = ax.annotate(SHORT[r.codec].replace('\\', ''), (r.mbps, r.bpb), xytext=(dx, dy),
                            textcoords='offset points', fontsize=5.2, ha=ha, va=va)
            box = a.get_window_extent(renderer).expanded(1.05, 1.1)
            a.remove()
            if not any(box.overlaps(b) for b in placed):
                placed.append(box)
                ax.annotate(SHORT[r.codec].replace('\\', ''), (r.mbps, r.bpb), xytext=(dx, dy),
                            textcoords='offset points', fontsize=5.2, ha=ha, va=va,
                            arrowprops=leader if abs(dx) + abs(dy) > 10 else None)
                break
        else:
            placed.append(ax.annotate(SHORT[r.codec].replace('\\', ''), (r.mbps, r.bpb), xytext=(4, 2),
                                      textcoords='offset points', fontsize=5.2).get_window_extent(renderer))
    savefig(fig, P, 'speed')

    # ------------------------------------------------------------ figure: break-even
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    Ns = np.logspace(5, 13, 200)
    shades = {'llm-gpt2': '#9085e9', 'llm-qwen0.5b': '#6250d6', 'llm-qwen1.5b': '#3a2c96', 'brotli': '#eb6834'}
    for c in llms + ['brotli']:
        sub = cost[(cost.codec == c) & (cost.file.isin(['en_sentences', 'hi_sentences', 'zh_sentences']))]
        if not len(sub):
            continue
        ax.plot(Ns, sub.rF.mean() + 8 * sub.K.iloc[0] / Ns, color=shades[c], label=SHORT[c].replace('\\', '') + r' ($r_K$)')
    for c, ls in (('xz', '--'), ('paq8px', ':')):
        sub = cost[(cost.codec == c) & (cost.file.isin(['en_sentences', 'hi_sentences', 'zh_sentences']))]
        if len(sub):
            ax.axhline(sub.rF.mean(), color='#52514e', linestyle=ls, linewidth=1, label=SHORT[c] + r' ($r_F$)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('input size $N$ in bytes (log); mean of the three language slices')
    ax.set_ylabel('bits per byte incl. knowledge')
    ax.legend(frameon=False, fontsize=5.5)
    savefig(fig, P, 'breakeven')

    write_numbers(P, numbers)
    return numbers
