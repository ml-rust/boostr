// Deterministic test signals for the narration glitch oracle. No transcendental function runs here: a tone comes from
// the recurrence y[n] = c*y[n-1] - y[n-2] with c = 2cos(w) and y[1] = sin(w) written as literals, noise from an
// integer xorshift32 generator, envelopes from linear ramps. Every sample is stored through Math.fround.
// tests/common/signals.rs builds the same buffers bit for bit; each case's input hash checks that.

// 2cos(w) and sin(w) for a tone of the named frequency at 48 kHz, or at 44.1 kHz for the `_44` names.
const TONE = {
  60: [1.9999383152895793, 0.007853900888711332],
  120: [1.9997532649633212, 0.015707317311820672],
  125: [1.9997322758191236, 0.01636173162648678],
  130: [1.9997104300531479, 0.01701613893230685],
  135: [1.9996877276747522, 0.017670538948953582],
  150: [1.9996144809641296, 0.0196336924606283],
  180: [1.9994448604360011, 0.023559764833610154],
  200: [1.9993146499511145, 0.026176948307873153],
  220: [1.9991707363874793, 0.02879395236826347],
  260: [1.9988418040633458, 0.034027350502167444],
  300: [1.9984580724814458, 0.03925981575906861],
  '70_44': [1.999900533911886, 0.009973144676492593],
  '200_44': [1.9991880801438362, 0.028491315390844964],
};

// A tone of `len` samples from `start`: amplitude `amp`, rising linearly over `attack` samples and falling over
// `release` (both at least 1).
export const tone = (hz, start, len, amp, attack, release) => ({ kind: 'tone', hz: String(hz), start, len, amp, attack, release });
// Uniform noise in [-amp, amp) over `len` samples from `start`, from xorshift32 seeded with `seed` (not 0).
export const noise = (start, len, amp, seed) => ({ kind: 'noise', start, len, amp, seed });
// One sample of `value` added at `at`.
export const spike = (at, value) => ({ kind: 'spike', at, value });

// Adds each op into `n` zero samples, in order: x[i] = fround(x[i] + v).
export function render(n, ops) {
  const x = new Float32Array(n);
  for (const op of ops) {
    if (op.kind === 'tone') {
      const [c, s1] = TONE[op.hz];
      let a = 0;
      let b = 0;
      for (let i = 0; i < op.len; i++) {
        const y = i === 0 ? 0 : i === 1 ? s1 : c * b - a;
        a = b;
        b = y;
        let g = i / op.attack;
        if (g > 1) g = 1;
        const r = (op.len - 1 - i) / op.release;
        if (r < g) g = r;
        x[op.start + i] = Math.fround(x[op.start + i] + op.amp * g * y);
      }
    } else if (op.kind === 'noise') {
      let s = op.seed >>> 0;
      for (let i = 0; i < op.len; i++) {
        s = (s ^ (s << 13)) >>> 0;
        s = (s ^ (s >>> 17)) >>> 0;
        s = (s ^ (s << 5)) >>> 0;
        x[op.start + i] = Math.fround(x[op.start + i] + op.amp * (s / 2147483648 - 1));
      }
    } else if (op.kind === 'spike') {
      x[op.at] = Math.fround(x[op.at] + op.value);
    } else {
      throw new Error(`signals: unknown op ${op.kind}`);
    }
  }
  return x;
}

// FNV-1a (32-bit) over each sample's f32 bit pattern, low byte first, as 8 hex digits.
export function fnv1a(x) {
  const words = new Uint32Array(x.buffer, x.byteOffset, x.length);
  let h = 0x811c9dc5;
  for (const w of words) {
    for (let b = 0; b < 32; b += 8) {
      h ^= (w >>> b) & 0xff;
      h = Math.imul(h, 0x01000193) >>> 0;
    }
  }
  return h.toString(16).padStart(8, '0');
}

// Glitch cases: 1 s at 48 kHz, a noise floor and one sustained tone as the speech.
const FLOOR = noise(0, 48000, 0.00001, 1);
const THUMP_FRONT = tone(60, 960, 1440, 0.3, 240, 240);
const THUMP_BACK = tone(60, 40800, 1920, 0.3, 240, 240);

// Click cases: 2 s at 48 kHz, speech, a pause from 0.8 to 1.2 s, then speech.
const PAUSED = [noise(0, 96000, 0.00002, 7), tone(200, 0, 38400, 0.3, 480, 2400), tone(200, 57600, 38400, 0.3, 2400, 480)];

// `run` names the detectors read on the case: `edges` (tidy, clicks, repairClicks) or `pitch` (pitches, medianPitch,
// heat for each of `voiceMedians`, where null is no reference and 'own' the case's own median pitch).
export const CASES = [
  { name: 'speech ending in silence', rate: 48000, n: 48000, run: 'edges', ops: [FLOOR, tone(200, 4800, 38400, 0.4, 2400, 4800)] },
  { name: 'thump before the speech', rate: 48000, n: 48000, run: 'edges', ops: [FLOOR, THUMP_FRONT, tone(200, 9600, 33600, 0.4, 2400, 4800)] },
  { name: 'thump after the speech', rate: 48000, n: 48000, run: 'edges', ops: [FLOOR, tone(200, 2400, 33600, 0.4, 2400, 4800), THUMP_BACK] },
  { name: 'thumps at both edges', rate: 48000, n: 48000, run: 'edges', ops: [FLOOR, THUMP_FRONT, tone(200, 9600, 28800, 0.4, 2400, 4800), THUMP_BACK] },
  { name: 'speech cut off at the end', rate: 48000, n: 48000, run: 'edges', ops: [FLOOR, tone(200, 4800, 43200, 0.4, 2400, 1)] },
  { name: 'speech decaying at the end', rate: 48000, n: 48000, run: 'edges', ops: [FLOOR, tone(200, 4800, 43200, 0.4, 2400, 960)] },
  { name: 'dark burst too close to the speech', rate: 48000, n: 48000, run: 'edges', ops: [FLOOR, THUMP_FRONT, tone(200, 3360, 38400, 0.4, 2400, 4800)] },
  { name: 'bright burst before the speech', rate: 48000, n: 48000, run: 'edges', ops: [FLOOR, noise(960, 1440, 0.3, 99), tone(200, 9600, 33600, 0.4, 2400, 4800)] },
  { name: 'dark sound too long for a thump', rate: 48000, n: 48000, run: 'edges', ops: [FLOOR, tone(60, 960, 3840, 0.3, 240, 240), tone(200, 9600, 33600, 0.4, 2400, 4800)] },
  { name: 'buffer shorter than a frame', rate: 48000, n: 100, run: 'edges', ops: [noise(0, 100, 0.1, 5)] },
  { name: 'empty buffer', rate: 48000, n: 0, run: 'edges', ops: [] },
  {
    name: '44.1 kHz thump before the speech, click after it',
    rate: 44100,
    n: 44100,
    run: 'edges',
    ops: [noise(0, 44100, 0.00001, 3), tone('70_44', 882, 1323, 0.3, 220, 220), tone('200_44', 8820, 30870, 0.4, 2205, 4410), spike(42000, 0.2)],
  },
  { name: 'one click in a pause', rate: 48000, n: 96000, run: 'edges', ops: [...PAUSED, spike(48010, 0.2)] },
  { name: 'clicks in a pause', rate: 48000, n: 96000, run: 'edges', ops: [...PAUSED, spike(42010, 0.15), spike(46810, -0.2), spike(51650, 0.25)] },
  { name: 'clicks under 0.05 s apart', rate: 48000, n: 96000, run: 'edges', ops: [...PAUSED, spike(46000, 0.2), spike(47300, 0.2)] },
  { name: 'click inside speech', rate: 48000, n: 96000, run: 'edges', ops: [...PAUSED, spike(20010, 0.3)] },
  { name: 'wide burst in a pause', rate: 48000, n: 96000, run: 'edges', ops: [...PAUSED, noise(48000, 480, 0.2, 11)] },
  // Each click's smoothed span centers on its time rounded to 10 ms, so where the click sits in that rounding sets the
  // half width that removes it. A click under 0.05 s after one found is only found once that one is gone.
  { name: 'click removed at 2 ms', rate: 48000, n: 96000, run: 'edges', ops: [...PAUSED, spike(47949, 0.2)] },
  { name: 'click removed at 4 ms', rate: 48000, n: 96000, run: 'edges', ops: [...PAUSED, spike(47900, 0.2)] },
  { name: 'click removed at 8 ms', rate: 48000, n: 96000, run: 'edges', ops: [...PAUSED, spike(48250, 0.2)] },
  { name: 'click removed at 12 ms', rate: 48000, n: 96000, run: 'edges', ops: [...PAUSED, spike(48250, 0.2), spike(49250, 0.2)] },
  { name: 'click left after every repair', rate: 48000, n: 96000, run: 'edges', ops: [...PAUSED, spike(48250, 0.2), spike(49250, 0.2), spike(50150, 0.2)] },
  { name: 'steady voice', rate: 48000, n: 96000, run: 'pitch', voiceMedians: [null, 0, 150, 'own'], ops: [tone(150, 0, 96000, 0.3, 480, 480)] },
  {
    name: 'heated climb',
    rate: 48000,
    n: 96000,
    run: 'pitch',
    voiceMedians: [120, 'own'],
    ops: [tone(180, 0, 24000, 0.3, 480, 480), tone(220, 24000, 24000, 0.3, 480, 480), tone(260, 48000, 24000, 0.3, 480, 480), tone(300, 72000, 24000, 0.3, 480, 480)],
  },
  {
    name: 'calm read',
    rate: 48000,
    n: 96000,
    run: 'pitch',
    voiceMedians: [125, 'own'],
    ops: [tone(120, 0, 24000, 0.3, 480, 480), tone(125, 24000, 24000, 0.3, 480, 480), tone(130, 48000, 24000, 0.3, 480, 480), tone(135, 72000, 24000, 0.3, 480, 480)],
  },
  { name: 'part too short to judge', rate: 48000, n: 24000, run: 'pitch', voiceMedians: [150], ops: [tone(150, 0, 24000, 0.3, 480, 480)] },
  { name: 'voiceless part', rate: 48000, n: 96000, run: 'pitch', voiceMedians: [150, 'own'], ops: [noise(0, 96000, 0.005, 13)] },
];
