#!/usr/bin/env bash
# Every run behind the paper, in the order they were made. Each line can be stopped and restarted:
# rows already in the results file are skipped, failed rows are retried.
set -u
cd "$(dirname "$0")/.."
B="python -I bench/run_bench.py"
CLASSIC=huffman,ext-huffman,eehc,dyn-huffman,rans0,gzip,bzip2,xz,brotli,zstd,zpaq,cm,paq8px
CANT="data/text/canterbury/alice29.txt data/text/canterbury/asyoulik.txt data/text/canterbury/cp.html
      data/text/canterbury/fields.c data/text/canterbury/grammar.lsp data/text/canterbury/lcet10.txt
      data/text/canterbury/plrabn12.txt data/text/canterbury/xargs.1"

# 1. Classical and context-mixing methods on the full files and on the 1 MB slices (CPU)
$B --results text.jsonl --set full  --codecs $CLASSIC data/text/bench/* $CANT
$B --results text.jsonl --set slice --codecs $CLASSIC data/text/slices/*

# 2. bsc and PPMd, re-run with their final settings (bsc single-threaded, PPMd order 6)
$B --results text_fix.jsonl --set full  --codecs bsc,ppmd data/text/bench/* $CANT
$B --results text_fix.jsonl --set slice --codecs bsc,ppmd data/text/slices/*

# 3. Pretrained language models and TRACE on the slices (GPU)
$B --results text_gpu.jsonl --set slice --codecs llm-gpt2,llm-qwen0.5b,llm-qwen1.5b,trace data/text/slices/*

# 4. NNCP (CPU build) and DZip on selected slices: both take hours per megabyte
$B --results text_nncp.jsonl --set slice --codecs nncp \
   data/text/slices/en_sentences data/text/slices/hi_sentences data/text/slices/zh_sentences data/text/slices/enwik8_10M
$B --results text_dzip.jsonl --set slice --codecs dzip,dzip-bs data/text/slices/en_sentences data/text/slices/hi_sentences
