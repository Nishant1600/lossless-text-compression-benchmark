// Measurements used in the results panel and the paper-style figures.

/** Shannon entropy of the block distribution, in bits per block. */
export function blockEntropy(freq, blockCount) {
  if (blockCount === 0) return 0;
  let h = 0;
  for (let i = 0; i < freq.length; i++) {
    const p = freq[i] / blockCount;
    if (p > 0) h -= p * Math.log2(p);
  }
  return h;
}

/**
 * Byte-by-byte comparison of the original and the restored data, plus the
 * 256×256 confusion matrix (paper Figs 8–9): matrix[orig * 256 + restored].
 */
export function verifyBytes(original, restored) {
  const n = Math.min(original.length, restored.length);
  const matrix = new Uint32Array(256 * 256);
  let mismatches = Math.abs(original.length - restored.length);
  let firstMismatch = original.length === restored.length ? -1 : n;
  for (let i = 0; i < n; i++) {
    const a = original[i];
    const b = restored[i];
    matrix[(a << 8) | b]++;
    if (a !== b) {
      if (firstMismatch < 0 || i < firstMismatch) firstMismatch = i;
      mismatches++;
    }
  }
  return { equal: mismatches === 0, mismatches, firstMismatch, matrix };
}

export async function sha256(bytes) {
  return new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
}

export function toHex(bytes) {
  let s = '';
  for (let i = 0; i < bytes.length; i++) s += bytes[i].toString(16).padStart(2, '0');
  return s;
}

export function sameBytes(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
