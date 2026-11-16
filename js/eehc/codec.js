// EEHC compression and decompression pipeline (paper Fig 2).
//
//   compress:   bytes → blocks → frequencies → base-r Huffman tree
//               → canonical codes → base-r digit stream → packed bytes
//   decompress: header → canonical tables → digits → blocks → bytes → SHA-256 check
//
// Base-R transformation (paper §iv): the paper reads the concatenated base-8
// digits Y as one integer Z and stores Bytes(Z). Because r = 8 = 2^3, writing
// every digit as 3 bits (most significant first) produces exactly the same
// bits, with the digit count stored so leading zeros survive. That keeps the
// conversion linear instead of quadratic big-integer arithmetic.

import { formBlocks, selectBlockSize, PAPER_BLOCK_SIZES } from './blocks.js';
import { bitsPerDigit, huffmanCodeLengths, canonicalCode, decoderTables } from './huffman.js';
import { containerLayout, encodeText, readContainer, writeHeader } from './container.js';
import { blockEntropy, sha256, sameBytes, toHex } from './stats.js';

export { isEEHC, FILE_EXTENSION } from './container.js';
export { selectBlockSize, PAPER_BLOCK_SIZES } from './blocks.js';

export const SUPPORTED_RADICES = [2, 4, 8, 16];
export const DEFAULT_RADIX = 8;

const PROGRESS_EVERY = 1 << 18;
const POW2 = Array.from({ length: 64 }, (_, i) => 2 ** i);
const CORRUPT = 'Corrupt EEHC data: the bitstream does not match the codebook';
const noop = () => {};
const now = () => performance.now();

/**
 * Steps 2–4 (blocks, frequencies, tree, canonical codes). Yields the exact
 * output sizes without encoding anything.
 */
export function analyze(bytes, { radix = DEFAULT_RADIX, blockSize = 'paper', name = '', mime = '', onProgress = noop } = {}) {
  const k = bitsPerDigit(radix);
  const B = blockSize === 'paper' ? selectBlockSize(bytes.length) : Number(blockSize);

  const t0 = now();
  const blocks = formBlocks(bytes, B, (done, total) => onProgress(`Forming ${B}-byte blocks`, done / total));
  const t1 = now();
  onProgress(`Building base-${radix} Huffman tree (${blocks.symbolCount.toLocaleString()} symbols)`, 0);
  const lengths = huffmanCodeLengths(blocks.freq, radix);
  const code = canonicalCode(lengths, radix);
  let digitCount = 0;
  for (let s = 0; s < blocks.symbolCount; s++) digitCount += blocks.freq[s] * lengths[s];
  const t2 = now();

  const payloadBytes = Math.ceil((digitCount * k) / 8);
  const nameBytes = encodeText(name, 0xffff);
  const mimeBytes = encodeText(mime, 0xff);
  const layout = containerLayout({
    nameBytes,
    mimeBytes,
    maxLen: code.maxLen,
    symbolCount: blocks.symbolCount,
    blockSize: B,
    payloadBytes,
  });
  return {
    radix, k, blockSize: B, blocks, lengths, code, digitCount, payloadBytes,
    nameBytes, mimeBytes, layout,
    timings: { blocksMs: t1 - t0, treeMs: t2 - t1 },
  };
}

/** Exact sizes for one configuration, without building the output file. */
export function measure(bytes, options) {
  const t0 = now();
  const p = analyze(bytes, options);
  return {
    radix: p.radix,
    blockSize: p.blockSize,
    blockCount: p.blocks.blockCount,
    symbolCount: p.blocks.symbolCount,
    payloadBytes: p.payloadBytes,
    metadataBytes: p.layout.metadataBytes,
    totalBytes: p.layout.totalBytes,
    ms: now() - t0,
  };
}

/**
 * Compress `bytes` into a .eehc container.
 * @param {object} options
 *   radix      2 | 4 | 8 | 16 (paper: 8)
 *   blockSize  'paper' (f(|X|)), 'optimize' (try 4/8/16, keep smallest file) or 1|2|4|8|16
 *   name, mime stored so decompression can restore the original file name/type
 *   onProgress (label, fraction) callback
 */
export async function compress(bytes, { radix = DEFAULT_RADIX, blockSize = 'paper', name = '', mime = '', onProgress = noop } = {}) {
  const tStart = now();
  let plan = null;
  let candidates = null;

  if (blockSize === 'optimize') {
    candidates = [];
    for (const B of PAPER_BLOCK_SIZES) {
      const p = analyze(bytes, {
        radix, blockSize: B, name, mime,
        onProgress: (label, f) => onProgress(`Trying B = ${B}: ${label}`, f),
      });
      candidates.push({ blockSize: B, payloadBytes: p.payloadBytes, metadataBytes: p.layout.metadataBytes, totalBytes: p.layout.totalBytes });
      if (!plan || p.layout.totalBytes < plan.layout.totalBytes) plan = p;
    }
  } else {
    plan = analyze(bytes, { radix, blockSize, name, mime, onProgress });
  }

  const { blocks, code, layout } = plan;
  onProgress('Computing SHA-256 of the original', 0);
  const t3 = now();
  const digest = await sha256(bytes);
  const t4 = now();

  const out = new Uint8Array(layout.totalBytes);
  writeHeader(out, layout, {
    radix,
    blockSize: plan.blockSize,
    maxLen: code.maxLen,
    originalSize: bytes.length,
    sha256: digest,
    nameBytes: plan.nameBytes,
    mimeBytes: plan.mimeBytes,
    symbolCount: blocks.symbolCount,
    counts: code.counts,
    digitCount: plan.digitCount,
  });
  writeSymbolTable(out, layout.tableOffset, blocks.keys, code.order, plan.blockSize);
  encodePayload(out, layout.payloadOffset, plan, onProgress);
  const t5 = now();

  const n = bytes.length;
  const H = blockEntropy(blocks.freq, blocks.blockCount);
  const stats = {
    name,
    mime,
    originalSize: n,
    blockSize: plan.blockSize,
    blockRule: blockSize === 'paper' || blockSize === 'optimize' ? blockSize : 'fixed',
    radix,
    bitsPerDigit: plan.k,
    blockCount: blocks.blockCount,
    symbolCount: blocks.symbolCount,
    maxCodeLength: code.maxLen,
    digitCount: plan.digitCount,
    payloadBytes: plan.payloadBytes,
    metadataBytes: layout.metadataBytes,
    codebookBytes: blocks.symbolCount * plan.blockSize,
    totalBytes: layout.totalBytes,
    avgDigitsPerBlock: blocks.blockCount ? plan.digitCount / blocks.blockCount : 0,
    avgBitsPerBlock: blocks.blockCount ? (plan.digitCount * plan.k) / blocks.blockCount : 0,
    blockEntropyBits: H,
    entropyBoundBytes: Math.ceil((H * blocks.blockCount) / 8),
    paperGain: n ? 1 - plan.payloadBytes / n : 0,
    trueGain: n ? 1 - layout.totalBytes / n : 0,
    sha256: toHex(digest),
    candidates,
    timings: {
      ...plan.timings,
      hashMs: t4 - t3,
      encodeMs: t5 - t4,
      totalMs: t5 - tStart,
    },
  };
  return { container: out, stats };
}

/** Restore the original bytes from a .eehc container. */
export async function decompress(bytes, { onProgress = noop } = {}) {
  const t0 = now();
  const h = readContainer(bytes);
  const restored = decodePayload(h, onProgress);
  const t1 = now();
  onProgress('Checking SHA-256', 1);
  const digest = await sha256(restored);
  const t2 = now();

  return {
    bytes: restored,
    name: h.name,
    mime: h.mime,
    hashOk: sameBytes(digest, h.sha256),
    stats: {
      originalSize: h.originalSize,
      blockSize: h.blockSize,
      radix: h.radix,
      symbolCount: h.symbolCount,
      maxCodeLength: h.maxLen,
      blockCount: Math.ceil(h.originalSize / h.blockSize),
      digitCount: h.digitCount,
      payloadBytes: h.payloadBytes,
      metadataBytes: h.metadataBytes,
      totalBytes: bytes.length,
      sha256: toHex(h.sha256),
      timings: { decodeMs: t1 - t0, hashMs: t2 - t1, totalMs: t2 - t0 },
    },
  };
}

// Symbol table in canonical order: entry i is the block bytes of symbol order[i].
function writeSymbolTable(out, offset, keys, order, B) {
  const N = order.length;
  if (B % 4 === 0 && keys.byteOffset % 4 === 0) {
    const W = B >>> 2;
    const src = new Uint32Array(keys.buffer, keys.byteOffset, N * W);
    const dst = new Uint32Array(out.buffer, out.byteOffset + offset, N * W);
    for (let i = 0; i < N; i++) {
      const s = order[i] * W;
      const d = i * W;
      for (let j = 0; j < W; j++) dst[d + j] = src[s + j];
    }
  } else {
    for (let i = 0; i < N; i++) out.set(keys.subarray(order[i] * B, order[i] * B + B), offset + i * B);
  }
}

// Step 5: replace each block with its code, packing log2(r) bits per digit.
function encodePayload(out, offset, plan, onProgress) {
  const { k, lengths } = plan;
  const { symbolIds, blockCount: m } = plan.blocks;
  const { codes } = plan.code;
  let pos = offset;
  let acc = 0;
  let accBits = 0;

  for (let i = 0; i < m; i++) {
    const s = symbolIds[i];
    let v = codes[s];
    let n = lengths[s] * k;
    while (n > 24) {
      n -= 24;
      const hi = Math.floor(v / POW2[n]);
      v -= hi * POW2[n];
      acc = ((acc << 24) | hi) >>> 0;
      accBits += 24;
      while (accBits >= 8) {
        accBits -= 8;
        out[pos++] = acc >>> accBits;
      }
      acc &= (1 << accBits) - 1;
    }
    acc = ((acc << n) | v) >>> 0;
    accBits += n;
    while (accBits >= 8) {
      accBits -= 8;
      out[pos++] = acc >>> accBits;
    }
    acc &= (1 << accBits) - 1;
    if ((i & (PROGRESS_EVERY - 1)) === 0) onProgress('Encoding blocks', i / m);
  }
  if (accBits > 0) out[pos++] = acc << (8 - accBits);
}

// Decompression: read digits, match canonical codes, emit block bytes.
function decodePayload(h, onProgress) {
  const { radix: r, blockSize: B, counts, maxLen, originalSize: n, digitCount, payload } = h;
  const k = bitsPerDigit(r);
  const digitMask = r - 1;
  const m = Math.ceil(n / B);
  const out = new Uint8Array(m * B);
  const { firstCode, offset } = decoderTables(counts, r);

  const W = B % 4 === 0 ? B >>> 2 : 0;
  let table = h.table;
  let table32 = null;
  let out32 = null;
  if (W) {
    if (table.byteOffset % 4 !== 0) table = table.slice();
    table32 = new Uint32Array(table.buffer, table.byteOffset, table.length >>> 2);
    out32 = new Uint32Array(out.buffer);
  }

  let pos = 0;
  let acc = 0;
  let accBits = 0;
  let code = 0;
  let len = 0;
  let blk = 0;
  for (let d = 0; d < digitCount; d++) {
    if (accBits < k) {
      acc = ((acc << 8) | payload[pos++]) & 0xffff;
      accBits += 8;
    }
    accBits -= k;
    code = code * r + ((acc >>> accBits) & digitMask);
    len++;
    const idx = code - firstCode[len];
    if (idx >= 0 && idx < counts[len]) {
      if (blk === m) throw new Error(CORRUPT);
      const s = offset[len] + idx;
      if (W) {
        const o = blk * W;
        const t = s * W;
        for (let j = 0; j < W; j++) out32[o + j] = table32[t + j];
      } else {
        const o = blk * B;
        const t = s * B;
        for (let j = 0; j < B; j++) out[o + j] = table[t + j];
      }
      blk++;
      code = 0;
      len = 0;
      if ((blk & (PROGRESS_EVERY - 1)) === 0) onProgress('Decoding blocks', blk / m);
    } else if (len >= maxLen) {
      throw new Error(CORRUPT);
    }
  }
  if (blk !== m || len !== 0) throw new Error(CORRUPT);
  return out.subarray(0, n);
}
