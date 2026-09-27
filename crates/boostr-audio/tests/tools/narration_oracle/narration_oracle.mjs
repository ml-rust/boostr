// Reference outputs for `boostr_audio::narration`: runs the JavaScript narration-script modules over the corpus
// next to this file and prints one JSON document to stdout.
//
// Usage: NARRATION_JS_DIR=<directory holding speak.mjs, subtitles.mjs, numbers.mjs, voice.mjs> \
//          node narration_oracle.mjs > ../../fixtures/narration_oracle.json
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const jsDir = process.env.NARRATION_JS_DIR;
if (!jsDir) {
  console.error('narration_oracle: NARRATION_JS_DIR is not set. Set it to the directory that holds speak.mjs, subtitles.mjs, numbers.mjs and voice.mjs.');
  process.exit(2);
}
const load = (name) => import(pathToFileURL(join(jsDir, name)).href);
const speak = await load('speak.mjs');
const subtitles = await load('subtitles.mjs');
const numbers = await load('numbers.mjs');
const voice = await load('voice.mjs');

const corpus = join(dirname(fileURLToPath(import.meta.url)), 'corpus');
const read = (rel) => readFileSync(join(corpus, rel), 'utf8');
const list = (dir) => readdirSync(join(corpus, dir)).sort().map((f) => `${dir}/${f}`);
const lines = (rel) => read(rel).split('\n').filter((l) => l.length);

const scripts = list('scripts').map(read);
const sentenceLines = lines('sentences.txt');
const timed = list('timed');
const subs = list('subtitles');

const cases = [];
const add = (fn, input, run) => {
  let output;
  try {
    output = run();
  } catch (e) {
    output = { error: e.message };
  }
  cases.push({ fn, input, output });
};

// `sentences` is private: a single line with no pause marks or notes gives its sentences through `phrases`.
for (const s of [...scripts, ...sentenceLines]) add('phrases', s, () => speak.phrases(s));

const chunkOptions = [
  { lines: false, max: 40, whole: false },
  { lines: true, max: 40, whole: false },
  { lines: false, max: 8, whole: false },
  { lines: false, max: 40, whole: true },
  { lines: false, max: 12, whole: true },
  { lines: true, max: 40, whole: true },
  { lines: false, max: 1, whole: false },
];
for (const script of scripts) for (const options of chunkOptions) add('chunks', { script, options }, () => speak.chunks(script, options));

for (const s of [...scripts, ...sentenceLines]) {
  for (const fn of ['withoutTags', 'say', 'shown', 'spoken', 'sayable']) add(fn, s, () => speak[fn](s));
}

for (const { ear, written } of JSON.parse(read('written.json'))) {
  add('writtenScript', { ear, written }, () => {
    const w = speak.writtenScript(ear, written);
    return { text: w.text, spans: w.spans(speak.phrases(ear)) };
  });
}

for (const f of [...timed, ...subs, ...list('scripts')]) {
  const raw = read(f);
  add('parseSubtitles', raw, () => subtitles.parseSubtitles(raw));
  add('parseTimedScript', raw, () => subtitles.parseTimedScript(raw));
}

// readScript takes a path: run from the corpus directory so the path in its message is the corpus-relative one.
process.chdir(corpus);
for (const file of [...timed, ...subs, ...list('scripts')]) {
  add('readScript', { file, raw: read(file) }, () => {
    const r = subtitles.readScript(file);
    // The subtitle branch leaves `targets` out; it reads as null.
    return { text: r.text, phrases: r.phrases ?? null, targets: r.targets ?? null };
  });
}

const numberLines = [...lines('numbers.txt'), ...sentenceLines, ...scripts];
for (const s of numberLines) {
  add('spokenNumbers', s, () => numbers.spokenNumbers(s));
  add('writtenNumbers', s, () => numbers.writtenNumbers(s));
  add('numbersIn', s, () => numbers.numbersIn(s));
}
for (const w of lines('number_words.txt')) add('isNumberWord', w, () => numbers.isNumberWord(w));

// The speech check runs every case of every file under speech/: heard words and the script they should say. Numbers
// in these outputs are written as JavaScript's String(x), which reads back as the exact double.
const exact = (v) => {
  if (typeof v === 'number') return String(v);
  if (Array.isArray(v)) return v.map(exact);
  if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, exact(x)]));
  return v;
};
for (const f of list('speech')) {
  for (const { name, heard, text } of JSON.parse(read(f))) {
    const input = { name: `${f}: ${name}`, heard, text };
    add('checkSpeech', input, () => exact(voice.checkSpeech(heard, text)));
    add('alignScript', input, () => exact(voice.alignScript(heard, text)));
    add('strayPauses', input, () => exact(speak.strayPauses(heard, text, voice.alignScript)));
  }
}

process.stdout.write(`${JSON.stringify({ pauses: speak.PAUSES, strayPause: speak.STRAY_PAUSE, cases }, null, 1)}\n`);
