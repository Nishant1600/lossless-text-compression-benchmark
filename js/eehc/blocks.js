// Adaptive block formation and frequency analysis (paper: Methodology §i, Steps 2–3).
//
// The input byte sequence X is cut into fixed-size blocks of B bytes and every
// distinct block becomes one symbol. Blocks are interned through an
// open-addressing hash table built on typed arrays, which keeps memory flat
// even when a video produces millions of mostly-unique 16-byte blocks.

export const PAPER_BLOCK_SIZES = [4, 8, 16];
export const SUPPORTED_BLOCK_SIZES = [1, 2, 4, 8, 16];

const PROGRESS_EVERY = 1 << 18;

/** The paper's block-size rule f(|X|). */
export function selectBlockSize(n) {
  if (n < 50_000) return 4;
  if (n < 1_000_000) return 8;
  return 16;
}

/**
 * Split `bytes` into B-byte blocks and map each distinct block to a symbol id.
 * The final partial block (if any) is zero-padded; the caller keeps the
 * original length so the padding can be dropped on decode.
 *
 * @returns {{blockSize:number, blockCount:number, symbolCount:number,
 *            symbolIds:Uint32Array, freq:Uint32Array, keys:Uint8Array}}
 *   symbolIds[i] is the symbol of block i, freq[s] its count and
 *   keys[s*B .. s*B+B) the block's bytes.
 */
export function formBlocks(bytes, B, onProgress) {
  if (!SUPPORTED_BLOCK_SIZES.includes(B)) throw new RangeError(`Unsupported block size: ${B}`);
  return B < 4 ? internDirect(bytes, B, onProgress) : internHashed(bytes, B, onProgress);
}

// B = 1 or 2: the block value itself indexes a lookup table.
function internDirect(bytes, B, onProgress) {
  const n = bytes.length;
  const m = Math.ceil(n / B);
  const space = 1 << (8 * B);
  const table = new Int32Array(space).fill(-1);
  const keyOf = new Uint32Array(space);
  const freq = new Uint32Array(space);
  const symbolIds = new Uint32Array(m);
  let N = 0;

  for (let i = 0; i < m; i++) {
    const p = i * B;
    const key = B === 1 ? bytes[p] : bytes[p] | ((p + 1 < n ? bytes[p + 1] : 0) << 8);
    let id = table[key];
    if (id < 0) {
      id = table[key] = N;
      keyOf[N++] = key;
    }
    freq[id]++;
    symbolIds[i] = id;
    if (onProgress && (i & (PROGRESS_EVERY - 1)) === 0) onProgress(i, m);
  }

  const keys = new Uint8Array(N * B);
  for (let s = 0; s < N; s++) {
    keys[s * B] = keyOf[s] & 255;
    if (B === 2) keys[s * B + 1] = keyOf[s] >>> 8;
  }
  return { blockSize: B, blockCount: m, symbolCount: N, symbolIds, freq: freq.slice(0, N), keys };
}

// B = 4, 8, 16: blocks are compared as B/4 32-bit words.
function internHashed(bytes, B, onProgress) {
  const n = bytes.length;
  const W = B >>> 2;
  const m = Math.ceil(n / B);
  const full = Math.floor(n / B);
  const src = wordView(bytes, full * B);
  let tail = null;
  if (full < m) {
    const t = new Uint8Array(B);
    t.set(bytes.subarray(full * B));
    tail = new Uint32Array(t.buffer);
  }

  let cap = Math.max(64, Math.min(m, 1 << 16));
  let symWords = new Uint32Array(cap * W);
  let symHash = new Uint32Array(cap);
  let freq = new Uint32Array(cap);
  let tableSize = nextPow2(cap * 2);
  let table = new Int32Array(tableSize).fill(-1);
  let mask = tableSize - 1;
  let N = 0;
  const symbolIds = new Uint32Array(m);

  for (let i = 0; i < m; i++) {
    let words = src;
    let base = i * W;
    if (i === full) {
      words = tail;
      base = 0;
    }

    const h = hashWords(words, base, W);
    let slot = h & mask;
    let id;
    for (;;) {
      id = table[slot];
      if (id < 0) break;
      if (symHash[id] === h && sameWords(symWords, id * W, words, base, W)) break;
      slot = (slot + 1) & mask;
    }

    if (id < 0) {
      if (N === cap) {
        cap *= 2;
        symWords = grow(symWords, cap * W);
        symHash = grow(symHash, cap);
        freq = grow(freq, cap);
      }
      id = N++;
      symHash[id] = h;
      const o = id * W;
      for (let j = 0; j < W; j++) symWords[o + j] = words[base + j];
      table[slot] = id;

      if (N * 2 > tableSize) {
        tableSize *= 2;
        mask = tableSize - 1;
        table = new Int32Array(tableSize).fill(-1);
        for (let s = 0; s < N; s++) {
          let k = symHash[s] & mask;
          while (table[k] >= 0) k = (k + 1) & mask;
          table[k] = s;
        }
      }
    }

    freq[id]++;
    symbolIds[i] = id;
    if (onProgress && (i & (PROGRESS_EVERY - 1)) === 0) onProgress(i, m);
  }

  // The words were read through a native-endian view of the input, so viewing
  // them as bytes again yields the original block bytes.
  const keys = new Uint8Array(symWords.buffer, 0, N * B);
  return { blockSize: B, blockCount: m, symbolCount: N, symbolIds, freq: freq.subarray(0, N), keys };
}

function wordView(bytes, len) {
  if (bytes.byteOffset % 4 === 0) return new Uint32Array(bytes.buffer, bytes.byteOffset, len >>> 2);
  const copy = new Uint8Array(len);
  copy.set(bytes.subarray(0, len));
  return new Uint32Array(copy.buffer);
}

// MurmurHash3 (x86, 32-bit) over W words.
function hashWords(words, base, W) {
  let h = 0x9747b28c;
  for (let j = 0; j < W; j++) {
    let k = words[base + j];
    k = Math.imul(k, 0xcc9e2d51);
    k = (k << 15) | (k >>> 17);
    k = Math.imul(k, 0x1b873593);
    h ^= k;
    h = (h << 13) | (h >>> 19);
    h = (Math.imul(h, 5) + 0xe6546b64) | 0;
  }
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

function sameWords(a, ao, b, bo, W) {
  for (let j = 0; j < W; j++) if (a[ao + j] !== b[bo + j]) return false;
  return true;
}

function grow(arr, size) {
  const next = new arr.constructor(size);
  next.set(arr);
  return next;
}

function nextPow2(x) {
  let p = 1;
  while (p < x) p *= 2;
  return p;
}
