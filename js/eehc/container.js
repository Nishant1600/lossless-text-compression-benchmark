// The .eehc container: one file holding the metadata the paper pickles
// separately (block size, radix, digit count, codebook) followed by the
// encoded bitstream (the paper's .bin).
//
// Layout (little-endian):
//   0   "EEHC" magic                     4
//   4   format version                   u8
//   5   radix r                          u8
//   6   block size B                     u8
//   7   max code length L (digits)       u8
//   8   original size                    u64
//   16  SHA-256 of original              32
//   48  name length + UTF-8 name         u16 + n
//       MIME length + UTF-8 MIME         u8 + n
//       unique symbol count N            u32
//       codes per length [1..L]          u32 × L
//       zero padding to a 4-byte boundary
//       symbol table, canonical order    N × B
//       digit count                      u64
//       payload: digits packed log2(r) bits each, MSB first

import { bitsPerDigit, MAX_CODE_BITS } from './huffman.js';
import { SUPPORTED_BLOCK_SIZES } from './blocks.js';

export const FORMAT_VERSION = 1;
export const FILE_EXTENSION = '.eehc';
const MAGIC = [0x45, 0x45, 0x48, 0x43]; // "EEHC"
const FIXED_HEADER = 48;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

export function isEEHC(bytes) {
  return bytes.length >= 4 && MAGIC.every((b, i) => bytes[i] === b);
}

export function encodeText(text, maxBytes) {
  const b = encoder.encode(text || '');
  return b.length > maxBytes ? b.subarray(0, maxBytes) : b;
}

/** Byte offsets of each section, given the sizes that vary per file. */
export function containerLayout({ nameBytes, mimeBytes, maxLen, symbolCount, blockSize, payloadBytes }) {
  const countsOffset = FIXED_HEADER + 2 + nameBytes.length + 1 + mimeBytes.length + 4;
  const tableOffset = Math.ceil((countsOffset + 4 * maxLen) / 4) * 4;
  const digitCountOffset = tableOffset + symbolCount * blockSize;
  const payloadOffset = digitCountOffset + 8;
  return {
    countsOffset,
    tableOffset,
    digitCountOffset,
    payloadOffset,
    metadataBytes: payloadOffset,
    totalBytes: payloadOffset + payloadBytes,
  };
}

// ---------------------------------------------------------------- shared prefix
// Every .eehc format starts the same way; byte 4 says which decoder to use:
//   1 = EEHC block Huffman (this file), 2 = Repack (js/repack/repack.js),
//   3 = sync pack (js/sync/pack.js), 4 = device manifest (js/sync/manifest.js).
// Bytes 5–7 are format-specific.

export const FORMAT_EEHC = 1;
export const FORMAT_REPACK = 2;
export const FORMAT_SYNC_PACK = 3;
export const FORMAT_MANIFEST = 4;

export const FORMAT_NAMES = {
  [FORMAT_EEHC]: 'EEHC',
  [FORMAT_REPACK]: 'Repack',
  [FORMAT_SYNC_PACK]: 'sync pack',
  [FORMAT_MANIFEST]: 'manifest',
};

/** Format byte of a container, or 0 if it is not an EEHC file. */
export function containerFormat(bytes) {
  return isEEHC(bytes) && bytes.length > 4 ? bytes[4] : 0;
}

export function commonHeaderSize(nameBytes, mimeBytes) {
  return FIXED_HEADER + 2 + nameBytes.length + 1 + mimeBytes.length;
}

/** Magic, format, 3 format-specific bytes, size, SHA-256, name, MIME. Returns the next offset. */
export function writeCommonHeader(out, { format, extra = [0, 0, 0], originalSize, sha256, nameBytes, mimeBytes }) {
  const dv = new DataView(out.buffer, out.byteOffset, out.byteLength);
  out.set(MAGIC, 0);
  dv.setUint8(4, format);
  out.set(extra, 5);
  dv.setBigUint64(8, BigInt(originalSize), true);
  out.set(sha256, 16);
  let p = FIXED_HEADER;
  dv.setUint16(p, nameBytes.length, true);
  out.set(nameBytes, p + 2);
  p += 2 + nameBytes.length;
  dv.setUint8(p, mimeBytes.length);
  out.set(mimeBytes, p + 1);
  return p + 1 + mimeBytes.length;
}

export function readCommonHeader(bytes) {
  if (!isEEHC(bytes)) throw new Error('Not an EEHC file (missing "EEHC" signature)');
  if (bytes.length < FIXED_HEADER + 3) throw new Error('EEHC file is truncated');
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let p = FIXED_HEADER;
  const nameLen = dv.getUint16(p, true);
  const name = decoder.decode(bytes.subarray(p + 2, p + 2 + nameLen));
  p += 2 + nameLen;
  if (p >= bytes.length) throw new Error('EEHC file is truncated');
  const mimeLen = dv.getUint8(p);
  const mime = decoder.decode(bytes.subarray(p + 1, p + 1 + mimeLen));
  p += 1 + mimeLen;
  if (p > bytes.length) throw new Error('EEHC file is truncated');
  return {
    format: bytes[4],
    extra: [bytes[5], bytes[6], bytes[7]],
    originalSize: toSafeNumber(dv.getBigUint64(8, true)),
    sha256: bytes.slice(16, 48),
    name,
    mime,
    end: p,
  };
}

// ---------------------------------------------------------------- format 1

/** Write the header fields into `out`. The codec fills in the symbol table and payload. */
export function writeHeader(out, layout, h) {
  const dv = new DataView(out.buffer, out.byteOffset, out.byteLength);
  const p = writeCommonHeader(out, {
    format: FORMAT_EEHC,
    extra: [h.radix, h.blockSize, h.maxLen],
    originalSize: h.originalSize,
    sha256: h.sha256,
    nameBytes: h.nameBytes,
    mimeBytes: h.mimeBytes,
  });
  dv.setUint32(p, h.symbolCount, true);

  for (let len = 1; len <= h.maxLen; len++) {
    dv.setUint32(layout.countsOffset + 4 * (len - 1), h.counts[len], true);
  }
  dv.setBigUint64(layout.digitCountOffset, BigInt(h.digitCount), true);
}

/** Parse and validate a container. Sections are returned as views, not copies. */
export function readContainer(bytes) {
  if (!isEEHC(bytes)) throw new Error('Not an EEHC file (missing "EEHC" signature)');
  if (bytes.length < FIXED_HEADER + 3) throw new Error('EEHC file is truncated');
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  const version = dv.getUint8(4);
  if (version === FORMAT_REPACK) throw new Error('This .eehc file was made with Repack, not EEHC');
  if (version !== FORMAT_VERSION) throw new Error(`Unsupported EEHC format version ${version}`);
  const radix = dv.getUint8(5);
  const k = bitsPerDigit(radix);
  const blockSize = dv.getUint8(6);
  if (!SUPPORTED_BLOCK_SIZES.includes(blockSize)) throw new Error(`Invalid block size ${blockSize}`);
  const maxLen = dv.getUint8(7);
  if (maxLen * k > MAX_CODE_BITS) throw new Error('Invalid code length table');
  const originalSize = toSafeNumber(dv.getBigUint64(8, true));
  const sha256 = bytes.slice(16, 48);

  let p = FIXED_HEADER;
  const nameLen = dv.getUint16(p, true);
  const nameBytes = bytes.subarray(p + 2, p + 2 + nameLen);
  p += 2 + nameLen;
  const mimeLen = dv.getUint8(p);
  const mimeBytes = bytes.subarray(p + 1, p + 1 + mimeLen);
  p += 1 + mimeLen;
  if (p + 4 > bytes.length) throw new Error('EEHC file is truncated');
  const symbolCount = dv.getUint32(p, true);

  const layout = containerLayout({ nameBytes, mimeBytes, maxLen, symbolCount, blockSize, payloadBytes: 0 });
  if (layout.payloadOffset > bytes.length) throw new Error('EEHC file is truncated');

  const counts = new Uint32Array(maxLen + 1);
  let total = 0;
  for (let len = 1; len <= maxLen; len++) {
    counts[len] = dv.getUint32(layout.countsOffset + 4 * (len - 1), true);
    total += counts[len];
  }
  if (total !== symbolCount) throw new Error('Corrupt EEHC header: code counts do not match symbol count');

  const digitCount = toSafeNumber(dv.getBigUint64(layout.digitCountOffset, true));
  const payloadBytes = bytes.length - layout.payloadOffset;
  if (Math.ceil((digitCount * k) / 8) > payloadBytes) throw new Error('EEHC file is truncated');

  return {
    version,
    radix,
    blockSize,
    maxLen,
    originalSize,
    sha256,
    name: decoder.decode(nameBytes),
    mime: decoder.decode(mimeBytes),
    symbolCount,
    counts,
    table: bytes.subarray(layout.tableOffset, layout.tableOffset + symbolCount * blockSize),
    digitCount,
    payload: bytes.subarray(layout.payloadOffset),
    metadataBytes: layout.payloadOffset,
    payloadBytes,
  };
}

function toSafeNumber(big) {
  if (big > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('Corrupt EEHC header: size out of range');
  return Number(big);
}
