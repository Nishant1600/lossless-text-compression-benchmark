// Command-line wrapper around this project's JavaScript coders, for the Python benchmark harness.
//
//   node js_codec.mjs c <method> <input> <output>   prints a JSON line with sizes and timings
//   node js_codec.mjs d <method> <input> <output>
//
// Methods: huffman (binary tree, 1-byte symbols), ext-huffman (binary tree, paper block size),
// eehc (base-8 tree, paper block size), eehc-opt (base-8, best of 4/8/16), cm (context mixing).

import fs from 'node:fs';
import { compress, decompress, selectBlockSize } from '../../js/eehc/codec.js';
import { compressCM, decompressCM } from '../../js/cm/cm.js';

const [cmd, method, input, output] = process.argv.slice(2);
const bytes = new Uint8Array(fs.readFileSync(input));
const t0 = performance.now();

const EEHC = {
  huffman: () => ({ radix: 2, blockSize: 1 }),
  'ext-huffman': (n) => ({ radix: 2, blockSize: selectBlockSize(n) }),
  eehc: () => ({ radix: 8, blockSize: 'paper' }),
  'eehc-opt': () => ({ radix: 8, blockSize: 'optimize' }),
};

let info = {};
if (method === 'cm') {
  const out = cmd === 'c' ? compressCM(bytes) : decompressCM(bytes);
  fs.writeFileSync(output, out);
} else if (EEHC[method]) {
  if (cmd === 'c') {
    const { container, stats } = await compress(bytes, EEHC[method](bytes.length));
    fs.writeFileSync(output, container);
    // payload = the encoded block codes alone (what the EEHC paper reports); side = codebook + header.
    info = { payload: stats.payloadBytes, side: stats.metadataBytes, codebook: stats.codebookBytes, blockSize: stats.blockSize, symbols: stats.symbolCount };
  } else {
    const r = await decompress(bytes);
    if (!r.hashOk) throw new Error('SHA-256 mismatch after decoding');
    fs.writeFileSync(output, r.bytes);
  }
} else {
  throw new Error(`unknown method: ${method}`);
}
console.log(JSON.stringify({ ms: performance.now() - t0, ...info }));
