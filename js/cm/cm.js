// Context-mixing (CM) compressor, written for this project.
//
// Every bit of the input is predicted from several models, the predictions are
// combined by small neural mixers, refined by two SSE stages, and coded with a
// binary arithmetic coder. Decoding runs the same predictor, so it is exactly as
// slow as encoding. All arithmetic is integer (or exact float) so the encoder and
// decoder stay in lockstep on any JavaScript engine.
//
// Models (one prediction each, per bit):
//   order 0, 1, 2, 3, 4, 6   hashed byte contexts
//   word                     current run of letters (case-folded) + previous byte
//   sparse                   bytes 2 and 3 back, skipping the last one (binary/tables)
//   match                    the byte that followed the last occurrence of the
//                            previous 6+ bytes, weighted by how long that match is
//
// Stream layout: u32 length (low), u32 length (high), arithmetic-coded bits.

const HASH_BITS = 22;
const HASH_SIZE = 1 << HASH_BITS;
const MATCH_BITS = 20;
const MIN_MATCH = 6;
const MAX_MATCH = 65535;
const COUNTER_LIMIT = 50;
const N_COUNTERS = 8;
const N_INPUTS = N_COUNTERS + 2; // + match + bias
const PROGRESS_EVERY = 1 << 16;

// ---------------------------------------------------------------- tables

const SQUASH = new Int16Array(4096); // stretch domain (-2047..2047) -> probability (1..4095)
const STRETCH = new Int16Array(4096); // probability -> stretch domain
for (let i = 0; i < 4096; i++) {
  const v = Math.round(4096 / (1 + Math.exp(-(i - 2048) / 256)));
  SQUASH[i] = Math.min(4095, Math.max(1, v));
}
{
  let pi = 0;
  for (let x = -2047; x <= 2047; x++) {
    const v = SQUASH[x + 2048];
    for (let j = pi; j <= v; j++) STRETCH[j] = x;
    pi = v + 1;
  }
  for (let j = pi; j < 4096; j++) STRETCH[j] = 2047;
}
const squash = (d) => (d > 2047 ? 4095 : d < -2047 ? 1 : SQUASH[d + 2048]);

// Adaptive learning rate for counters: 1 / (n + 1.5), in 1/65536 units.
const RATE = new Float64Array(256);
for (let n = 0; n < 256; n++) RATE[n] = 1 / (n + 1.5);

// ---------------------------------------------------------------- APM / SSE

class APM {
  constructor(contexts, rate) {
    this.t = new Uint16Array(contexts * 33);
    this.rate = rate;
    for (let c = 0; c < contexts; c++) {
      for (let j = 0; j < 33; j++) this.t[c * 33 + j] = squash((j - 16) * 128) * 16;
    }
    this.index = 0;
  }

  /** Refine probability `pr` (1..4095) in context `cx`. */
  pp(pr, cx) {
    const s = STRETCH[pr] + 2048;
    const lo = s >> 7;
    const w = s & 127;
    const i = cx * 33 + lo;
    this.index = w < 64 ? i : i + 1; // update the nearer bin
    return (this.t[i] * (128 - w) + this.t[i + 1] * w) >> 11;
  }

  update(bit) {
    const g = (bit << 16) + (bit << this.rate) - bit - bit;
    this.t[this.index] += (g - this.t[this.index]) >> this.rate;
  }
}

// ---------------------------------------------------------------- predictor

class Predictor {
  constructor(buf) {
    this.buf = buf; // input (encoder) or output (decoder); filled byte by byte
    this.pos = 0;

    this.prob = [];
    this.count = [];
    const sizes = [256, 65536, HASH_SIZE, HASH_SIZE, HASH_SIZE, HASH_SIZE, HASH_SIZE, HASH_SIZE];
    for (const size of sizes) {
      this.prob.push(new Uint16Array(size).fill(32768));
      this.count.push(new Uint8Array(size));
    }
    this.base = new Uint32Array(N_COUNTERS); // per-byte context hashes
    this.idx = new Int32Array(N_COUNTERS); // per-bit table slots
    this.st = new Int32Array(N_INPUTS);

    // Two mixers with different weight-set selectors, averaged.
    this.wA = new Int32Array(256 * N_INPUTS).fill(1 << 14); // selected by partial byte c0
    this.wB = new Int32Array(64 * N_INPUTS).fill(1 << 14); // selected by match length and bit position
    this.selA = 0;
    this.selB = 0;
    this.dotA = 0;
    this.dotB = 0;
    this.prMix = 2048;

    this.apm1 = new APM(65536, 7); // order 1
    this.apm2 = new APM(65536, 7); // hashed order 2
    this.pr = 2048;

    // Match model.
    this.matchTable = new Int32Array(1 << MATCH_BITS);
    this.matchPtr = 0;
    this.matchLen = 0;
    this.matchByte = -1;
    this.matchProb = new Uint16Array(64 * 2).fill(32768);
    this.matchCount = new Uint8Array(64 * 2);
    this.matchSlot = -1;

    this.c0 = 1; // partial byte with a leading 1
    this.bitPos = 0;
    this.c4 = 0; // last 4 bytes
    this.c8 = 0; // bytes 5..8 back
    this.word = 0;
    this.computeContexts();
    this.predictBit();
  }

  computeContexts() {
    const c4 = this.c4;
    const c1 = c4 & 0xff;
    const b = this.base;
    b[0] = 0;
    b[1] = c1 << 8;
    b[2] = Math.imul((c4 & 0xffff) + 0x10000, 0x9e3779b1);
    b[3] = Math.imul((c4 & 0xffffff) + 0x3000000, 0x85ebca6b);
    b[4] = Math.imul(c4 ^ 0x5bd1e995, 0xc2b2ae35) ^ 0x04000000;
    b[5] = Math.imul(c4 ^ Math.imul(this.c8 & 0xffff, 0x27d4eb2f), 0x165667b1) ^ 0x06000000;
    b[6] = Math.imul(this.word ^ Math.imul(c1, 0x01000193) ^ 0x7000000, 0x2f0f3b1d);
    b[7] = Math.imul(((c4 >>> 8) & 0xffff) | 0x8000000, 0x6c8e9cf5);
  }

  predictBit() {
    const c0 = this.c0;
    const st = this.st;
    // Counters.
    this.idx[0] = c0;
    this.idx[1] = this.base[1] | c0;
    const mix = Math.imul(c0, 0x2545f491);
    for (let i = 2; i < N_COUNTERS; i++) this.idx[i] = ((this.base[i] ^ mix) >>> 0) >>> (32 - HASH_BITS);
    for (let i = 0; i < N_COUNTERS; i++) st[i] = STRETCH[this.prob[i][this.idx[i]] >>> 4];

    // Match model: only predicts while the bits so far agree with the expected byte.
    let expectBit = 0;
    if (this.matchLen > 0 && ((this.matchByte + 256) >> (8 - this.bitPos)) === c0) {
      expectBit = (this.matchByte >> (7 - this.bitPos)) & 1;
      const lenBucket = this.matchLen < 16 ? this.matchLen : this.matchLen < 32 ? 16 + ((this.matchLen - 16) >> 2) : Math.min(63, 20 + (this.matchLen >> 5));
      this.matchSlot = lenBucket * 2 + expectBit;
      st[N_COUNTERS] = STRETCH[this.matchProb[this.matchSlot] >>> 4];
    } else {
      this.matchSlot = -1;
      st[N_COUNTERS] = 0;
    }
    st[N_COUNTERS + 1] = 256; // bias

    // Mixers.
    const lenSel = this.matchSlot < 0 ? 0 : 1 + Math.min(6, this.matchLen >> 2);
    this.selA = c0 * N_INPUTS;
    this.selB = (lenSel * 8 + this.bitPos) * N_INPUTS;
    let dA = 0;
    let dB = 0;
    for (let i = 0; i < N_INPUTS; i++) {
      dA += st[i] * this.wA[this.selA + i];
      dB += st[i] * this.wB[this.selB + i];
    }
    dA = clampStretch(dA >> 16);
    dB = clampStretch(dB >> 16);
    this.dotA = dA;
    this.dotB = dB;
    this.prA = squash(dA);
    this.prB = squash(dB);
    const prMix = squash((dA + dB) >> 1);
    this.prMix = prMix;

    // SSE.
    const c1 = this.c4 & 0xff;
    const p1 = this.apm1.pp(prMix, c0 | (c1 << 8));
    const h2 = (((this.c4 & 0xffff) * 0x9e37) >>> 0) & 0xff;
    const p2 = this.apm2.pp(prMix, c0 | (h2 << 8));
    let pr = (prMix * 2 + p1 + p2 + 2) >> 2;
    if (pr < 1) pr = 1;
    else if (pr > 4095) pr = 4095;
    this.pr = pr;
    return pr;
  }

  update(bit) {
    // Counters.
    for (let i = 0; i < N_COUNTERS; i++) {
      const t = this.prob[i];
      const n = this.count[i];
      const j = this.idx[i];
      const k = n[j];
      const p = t[j];
      t[j] = p + (((bit ? 65535 : 0) - p) * RATE[k]) | 0;
      if (k < COUNTER_LIMIT) n[j] = k + 1;
    }
    if (this.matchSlot >= 0) {
      const j = this.matchSlot;
      const k = this.matchCount[j];
      const p = this.matchProb[j];
      this.matchProb[j] = p + (((bit ? 65535 : 0) - p) * RATE[k]) | 0;
      if (k < 255) this.matchCount[j] = k + 1;
    }

    // Mixers: w += stretch * error, scaled.
    const st = this.st;
    const eA = ((bit << 12) - this.prA) * 6;
    const eB = ((bit << 12) - this.prB) * 6;
    const wA = this.wA;
    const wB = this.wB;
    for (let i = 0; i < N_INPUTS; i++) {
      let a = wA[this.selA + i] + ((st[i] * eA) >> 16);
      let b = wB[this.selB + i] + ((st[i] * eB) >> 16);
      wA[this.selA + i] = a > 524287 ? 524287 : a < -524287 ? -524287 : a;
      wB[this.selB + i] = b > 524287 ? 524287 : b < -524287 ? -524287 : b;
    }
    this.apm1.update(bit);
    this.apm2.update(bit);

    // Advance the bit/byte state.
    this.c0 = (this.c0 << 1) | bit;
    this.bitPos++;
    if (this.bitPos === 8) this.endByte(this.c0 & 0xff);
    this.predictBit();
  }

  endByte(c) {
    this.buf[this.pos++] = c;
    this.c8 = ((this.c8 << 8) | (this.c4 >>> 24)) >>> 0;
    this.c4 = ((this.c4 << 8) | c) >>> 0;
    this.c0 = 1;
    this.bitPos = 0;
    const lower = c | 32;
    this.word = lower >= 97 && lower <= 122 ? Math.imul(this.word ^ lower, 0x01000193) >>> 0 : 0;

    // Match model: extend the current match or look for a new one.
    if (this.matchLen > 0 && this.buf[this.matchPtr] === c) {
      this.matchPtr++;
      if (this.matchLen < MAX_MATCH) this.matchLen++;
    } else {
      this.matchLen = 0;
    }
    if (this.pos >= MIN_MATCH) {
      const h = (Math.imul(this.c4, 0x9e3779b1) ^ Math.imul(this.c8 & 0xffff, 0x85ebca6b)) >>> (32 - MATCH_BITS);
      if (this.matchLen === 0) {
        const cand = this.matchTable[h];
        if (cand > 0) {
          let k = 0;
          while (k < 32 && cand - 1 - k >= 0 && this.buf[cand - 1 - k] === this.buf[this.pos - 1 - k]) k++;
          if (k >= MIN_MATCH) {
            this.matchLen = k;
            this.matchPtr = cand;
          }
        }
      }
      this.matchTable[h] = this.pos;
    }
    this.matchByte = this.matchLen > 0 && this.matchPtr < this.pos ? this.buf[this.matchPtr] : -1;
    if (this.matchByte < 0) this.matchLen = 0;
    this.computeContexts();
  }
}

function clampStretch(d) {
  return d > 2047 ? 2047 : d < -2047 ? -2047 : d;
}

// ---------------------------------------------------------------- coder

const EMPTY = new Uint8Array(0);

/**
 * Feed `prime` through the model via update() with no arithmetic coding, so the match
 * model and context tables learn it exactly as if it were the real start of the stream.
 * Used to "prime" the compressor with a shared dictionary or an earlier version of the
 * same file that both sides already have, so the coded stream only has to pay for what
 * is actually new.
 */
function primeModel(model, prime) {
  for (let i = 0; i < prime.length; i++) {
    const c = prime[i];
    for (let j = 7; j >= 0; j--) model.update((c >> j) & 1);
  }
}

export function compressCM(input, onProgress = () => {}, prime = EMPTY) {
  const n = input.length;
  const out = new GrowableBytes(Math.max(1024, n >>> 2));
  out.u32(n % 4294967296);
  out.u32(Math.floor(n / 4294967296));
  const model = new Predictor(new Uint8Array(prime.length + n));
  primeModel(model, prime);
  let x1 = 0;
  let x2 = 0xffffffff;
  for (let i = 0; i < n; i++) {
    const c = input[i];
    for (let j = 7; j >= 0; j--) {
      const bit = (c >> j) & 1;
      const p = model.pr;
      const range = x2 - x1;
      const xmid = x1 + Math.floor(range / 4096) * p + Math.floor(((range % 4096) * p) / 4096);
      if (bit) x2 = xmid;
      else x1 = xmid + 1;
      model.update(bit);
      while (((x1 ^ x2) & 0xff000000) === 0) {
        out.push(x2 >>> 24);
        x1 = (x1 << 8) >>> 0;
        x2 = ((x2 << 8) | 255) >>> 0;
      }
    }
    if ((i & (PROGRESS_EVERY - 1)) === 0) onProgress(i / n);
  }
  out.push(x1 >>> 24);
  out.push((x1 >>> 16) & 255);
  out.push((x1 >>> 8) & 255);
  out.push(x1 & 255);
  return out.bytes();
}

export function decompressCM(stream, onProgress = () => {}, prime = EMPTY) {
  if (stream.length < 8) throw new Error('Corrupt CM stream');
  const dv = new DataView(stream.buffer, stream.byteOffset, stream.byteLength);
  const n = dv.getUint32(0, true) + dv.getUint32(4, true) * 4294967296;
  const out = new Uint8Array(prime.length + n);
  const model = new Predictor(out);
  primeModel(model, prime);
  let pos = 8;
  let x1 = 0;
  let x2 = 0xffffffff;
  let x = 0;
  for (let k = 0; k < 4; k++) x = ((x << 8) | (pos < stream.length ? stream[pos++] : 0)) >>> 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < 8; j++) {
      const p = model.pr;
      const range = x2 - x1;
      const xmid = x1 + Math.floor(range / 4096) * p + Math.floor(((range % 4096) * p) / 4096);
      const bit = x <= xmid ? 1 : 0;
      if (bit) x2 = xmid;
      else x1 = xmid + 1;
      model.update(bit);
      while (((x1 ^ x2) & 0xff000000) === 0) {
        x1 = (x1 << 8) >>> 0;
        x2 = ((x2 << 8) | 255) >>> 0;
        x = ((x << 8) | (pos < stream.length ? stream[pos++] : 0)) >>> 0;
      }
    }
    if ((i & (PROGRESS_EVERY - 1)) === 0) onProgress(i / n);
  }
  return prime.length ? out.subarray(prime.length) : out;
}

class GrowableBytes {
  constructor(capacity) {
    this.buf = new Uint8Array(capacity);
    this.length = 0;
  }

  push(b) {
    if (this.length === this.buf.length) {
      const next = new Uint8Array(this.buf.length * 2);
      next.set(this.buf);
      this.buf = next;
    }
    this.buf[this.length++] = b;
  }

  u32(v) {
    this.push(v & 255);
    this.push((v >>> 8) & 255);
    this.push((v >>> 16) & 255);
    this.push((v >>> 24) & 255);
  }

  bytes() {
    return this.buf.slice(0, this.length);
  }
}
