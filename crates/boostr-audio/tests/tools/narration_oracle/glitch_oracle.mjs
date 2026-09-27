// Reference outputs for the sample-level checks in `boostr_audio::narration`: runs the JavaScript tidy, click and
// pitch checks over the signals in signals.mjs and prints one JSON document to stdout. Each case carries the FNV-1a
// hash of its input, so a signal built differently is reported as such. Sample outputs are written as hashes and
// numbers as JavaScript's String(x), which reads back as the exact double.
//
// Usage: NARRATION_JS_DIR=<directory holding speak.mjs> node glitch_oracle.mjs > ../../fixtures/narration_glitch_oracle.json
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { CASES, fnv1a, render } from './signals.mjs';

const jsDir = process.env.NARRATION_JS_DIR;
if (!jsDir) {
  console.error('glitch_oracle: NARRATION_JS_DIR is not set. Set it to the directory that holds speak.mjs.');
  process.exit(2);
}
const speak = await import(pathToFileURL(join(jsDir, 'speak.mjs')).href);

const exact = (list) => list.map(String);
const cases = CASES.map(({ name, rate, n, run, ops, voiceMedians }) => {
  const x = render(n, ops);
  const out = { name, rate, n, input: fnv1a(x) };
  if (run === 'edges') {
    const t = speak.tidy(x, rate);
    out.tidy = { samples: fnv1a(t.samples), faults: t.faults, repairs: t.repairs };
    out.clicks = exact(speak.clicks(x, rate));
    const r = speak.repairClicks(x, rate);
    out.repairClicks = { samples: fnv1a(r.samples), fixed: exact(r.fixed), left: exact(r.left) };
  } else if (run === 'pitch') {
    const median = speak.medianPitch(x);
    out.pitches = exact(speak.pitches(x));
    out.medianPitch = String(median);
    out.heat = voiceMedians.map((v) => {
      const voiceMedian = v === 'own' ? median : v;
      return { voiceMedian: voiceMedian == null ? null : String(voiceMedian), heat: String(speak.heat(x, voiceMedian ?? undefined)) };
    });
  } else {
    throw new Error(`glitch_oracle: case ${name} names an unknown run ${run}`);
  }
  return out;
});

process.stdout.write(`${JSON.stringify({ heated: String(speak.HEATED), cases }, null, 1)}\n`);
