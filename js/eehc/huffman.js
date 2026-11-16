// High-radix (r-ary) Huffman coding (paper: Methodology §iii, Step 4).
//
// Code lengths are measured in base-r digits. The tree is built with the
// two-queue method: leaves sorted by frequency in one queue, newly merged
// nodes (whose weights only ever increase) in the other, so each merge simply
// takes the r smallest fronts. Codes are then assigned canonically, so the
// decoder only needs "how many codes of each length" plus the symbol table.

/** Bits needed to store one base-r digit (r must be a power of two). */
export function bitsPerDigit(r) {
  const k = Math.log2(r);
  if (!Number.isInteger(k) || k < 1 || k > 4) throw new RangeError(`Unsupported radix: ${r}`);
  return k;
}

/** Largest code (in bits) that still fits a JS number exactly. */
export const MAX_CODE_BITS = 52;

/**
 * r-ary Huffman code lengths for symbols 0..N-1.
 * @param {Uint32Array} freq  frequency of each symbol
 * @param {number} r          radix
 * @returns {Uint8Array} length in base-r digits for each symbol
 */
export function huffmanCodeLengths(freq, r) {
  const N = freq.length;
  const lengths = new Uint8Array(N);
  if (N === 0) return lengths;
  if (N === 1) {
    lengths[0] = 1;
    return lengths;
  }

  // Pad with zero-weight dummy leaves so every merge takes exactly r nodes.
  const d = (r - 1 - ((N - 1) % (r - 1))) % (r - 1);
  const L = N + d;
  const internal = (L - 1) / (r - 1);
  const T = L + internal;

  const order = sortByFrequency(freq);
  const weight = new Float64Array(T);
  const parent = new Int32Array(T);
  for (let i = 0; i < N; i++) weight[d + i] = freq[order[i]];

  let leaf = 0;
  let queued = L;
  let next = L;
  for (let n = 0; n < internal; n++) {
    let sum = 0;
    for (let c = 0; c < r; c++) {
      const pick = leaf < L && (queued >= next || weight[leaf] <= weight[queued]) ? leaf++ : queued++;
      parent[pick] = next;
      sum += weight[pick];
    }
    weight[next++] = sum;
  }

  // Every node's parent was created after it, so one backwards pass gives depths.
  const depth = new Uint16Array(T);
  for (let i = T - 2; i >= 0; i--) depth[i] = depth[parent[i]] + 1;
  for (let i = 0; i < N; i++) {
    const len = depth[d + i];
    if (len > 255) throw new Error('Huffman code too long');
    lengths[order[i]] = len;
  }
  return lengths;
}

// Symbol ids sorted by ascending frequency (ties by id). Counting sort for the
// common small frequencies, comparison sort only for the rare large ones.
function sortByFrequency(freq) {
  const N = freq.length;
  const LIMIT = 1 << 16;
  const start = new Uint32Array(LIMIT + 2);
  for (let i = 0; i < N; i++) start[Math.min(freq[i], LIMIT) + 1]++;
  for (let f = 1; f <= LIMIT + 1; f++) start[f] += start[f - 1];
  const bigStart = start[LIMIT];
  const order = new Uint32Array(N);
  for (let i = 0; i < N; i++) order[start[Math.min(freq[i], LIMIT)]++] = i;
  if (bigStart < N) order.subarray(bigStart).sort((a, b) => freq[a] - freq[b] || a - b);
  return order;
}

/**
 * Canonical code assignment.
 * @returns {{maxLen:number, counts:Uint32Array, order:Uint32Array, codes:Float64Array}}
 *   counts[len] = number of codes of that length (index 0 unused),
 *   order       = symbol ids in canonical order (the stored symbol table order),
 *   codes[s]    = numeric value of symbol s's code (read as a base-r number).
 */
export function canonicalCode(lengths, r) {
  const N = lengths.length;
  let maxLen = 0;
  for (let i = 0; i < N; i++) if (lengths[i] > maxLen) maxLen = lengths[i];
  if (maxLen * bitsPerDigit(r) > MAX_CODE_BITS) {
    throw new Error(`Longest code (${maxLen} digits) exceeds ${MAX_CODE_BITS} bits`);
  }

  const counts = new Uint32Array(maxLen + 1);
  for (let i = 0; i < N; i++) counts[lengths[i]]++;

  const offset = lengthOffsets(counts);
  const firstCode = firstCodes(counts, r);
  const order = new Uint32Array(N);
  const codes = new Float64Array(N);
  const rank = offset.slice();
  for (let s = 0; s < N; s++) {
    const len = lengths[s];
    const pos = rank[len]++;
    order[pos] = s;
    codes[s] = firstCode[len] + (pos - offset[len]);
  }
  return { maxLen, counts, order, codes };
}

/** Lookup tables for canonical decoding from the per-length counts. */
export function decoderTables(counts, r) {
  return { firstCode: firstCodes(counts, r), offset: lengthOffsets(counts) };
}

// firstCode[len] = value of the first canonical code of that length.
function firstCodes(counts, r) {
  const maxLen = counts.length - 1;
  const first = new Float64Array(maxLen + 1);
  let code = 0;
  for (let len = 1; len <= maxLen; len++) {
    first[len] = code;
    code = (code + counts[len]) * r;
  }
  return first;
}

// offset[len] = index in the canonical symbol table of the first code of that length.
function lengthOffsets(counts) {
  const offset = new Uint32Array(counts.length);
  for (let len = 1; len < counts.length; len++) offset[len] = offset[len - 1] + counts[len - 1];
  return offset;
}
