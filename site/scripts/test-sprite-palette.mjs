import assert from 'node:assert/strict';
import '../sprite-palette.js';
const source = new Uint8ClampedArray([255,0,0,255, 0,255,0,128, 0,0,255,255, 240,240,230,200]);
const original = Array.from(source);
assert.deepEqual(Array.from(CREW_PALETTE.recolor(source, 'red')), [198,17,17,255,148,201,219,128,122,8,56,255,240,240,230,200]);
assert.deepEqual(Array.from(CREW_PALETTE.recolor(source, 'black')), [63,71,78,255,148,201,219,128,30,31,38,255,240,240,230,200]);
assert.deepEqual(Array.from(source), original);
assert.throws(() => CREW_PALETTE.recolor(source, 'unknown'));
console.log('PASS: red/black palettes, alpha, neutral highlights and original mask preservation.');
