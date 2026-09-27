#!/usr/bin/env python3
"""Oracle fixture for Kokoro's text front end: `KPipeline`'s chunking plus
`KModel`'s phoneme -> input-id mapping, no G2P fallback attached.

For each case (~25 sentences, one embedded-newline paragraph, one long
multi-chunk paragraph) and each voice (`af_heart` = American English,
`bf_emma` = British English), this dumps the exact sequence of (graphemes,
phonemes, input_ids) chunks `KPipeline.__call__` would hand to
`KModel.forward`, computed without loading any model or voice pack:

* G2P: `misaki.en.G2P(trf=False, british=<voice's dialect>, fallback=None,
  unk='')` — reference Kokoro's `KPipeline` always attaches an
  `EspeakFallback` for `a*`/`b*` voices; this port's Kokoro engine does not
  (a separate unit wires the neural fallback in), so the oracle omits it
  too, for parity.
* Chunking: `KPipeline.en_tokenize`, called unbound (it never reads
  `self`), so no `KPipeline` instance — and so no HF download or model
  load — is needed.
* Ids: `config.json`'s `vocab` dict directly, `KModel.forward`'s own
  mapping (`[0, *ids, 0]`, unknown characters dropped).

Output JSON:
    {"vocab": {symbol: id, ...}, "cases": [case, ...]}
    case = {
      "name": str,
      "text": str,
      "voices": {
        "af_heart": [{"graphemes": str, "phonemes": str, "input_ids": [int, ...]}, ...],
        "bf_emma": [...],
      },
    }

Requires: misaki 0.9.4, kokoro 0.9.4 (spaCy + en_core_web_sm come with
misaki[en]).

Environment:
    KOKORO_DIR              Kokoro-82M directory (has `config.json`);
                             default `$BOOSTR_MODELS_DIR/Kokoro-82M`
    KOKORO_PIPELINE_CORPUS  corpus path (default: corpus.txt beside this
                             script)
    KOKORO_PIPELINE_FIXTURE output path (default: argv[1])

Usage:
    python3 crates/boostr-audio/tests/tools/kokoro_pipeline_oracle/kokoro_pipeline_oracle.py \
        crates/boostr-audio/tests/fixtures/kokoro_pipeline_oracle.json
"""
import json
import os
import re
import sys

from kokoro.pipeline import KPipeline
from misaki import en

VOICES = {"af_heart": False, "bf_emma": True}
NUM_SENTENCES = 25


def kokoro_dir():
    env = os.environ.get("KOKORO_DIR")
    if env:
        return env
    root = os.environ.get("BOOSTR_MODELS_DIR")
    if root:
        return os.path.join(root, "Kokoro-82M")
    sys.exit("set KOKORO_DIR or BOOSTR_MODELS_DIR")


def load_vocab(kokoro_dir_path):
    with open(os.path.join(kokoro_dir_path, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    return config["vocab"]


def to_input_ids(vocab, phonemes):
    ids = [vocab[p] for p in phonemes if p in vocab]
    return [0, *ids, 0]


def chunks_for(g2p, vocab, text):
    # Mirrors `KPipeline.__call__`'s own segment split (default
    # `split_pattern = r'\n+'` over `text.strip()`) before G2P + chunking
    # run on each segment independently.
    out = []
    for segment in re.split(r"\n+", text.strip()):
        if not segment.strip():
            continue
        _, tokens = g2p(segment)
        for gs, ps, _tks in KPipeline.en_tokenize(None, tokens):
            if not ps:
                continue
            out.append({"graphemes": gs, "phonemes": ps, "input_ids": to_input_ids(vocab, ps)})
    return out


def build_cases(corpus_path):
    with open(corpus_path, encoding="utf-8") as f:
        sentences = [line.rstrip("\n") for line in f if line.strip()]
    sentences = sentences[:NUM_SENTENCES]

    cases = [{"name": f"sentence_{i:02d}", "text": s} for i, s in enumerate(sentences)]

    # A paragraph with an embedded blank line: KPipeline's default
    # `split_pattern = r'\n+'` phonemizes and chunks each side separately.
    cases.append({
        "name": "embedded_blank_line",
        "text": sentences[0] + "\n\n" + sentences[1],
    })

    # A long paragraph that crosses the 510-phoneme-character budget for
    # both dialects, so `en_tokenize` must cut it into more than one chunk.
    cases.append({
        "name": "long_multi_chunk_paragraph",
        "text": " ".join(sentences[:12] + sentences[:12]),
    })

    return cases


def main():
    out = os.environ.get("KOKORO_PIPELINE_FIXTURE") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if out is None:
        sys.exit("usage: kokoro_pipeline_oracle.py <fixture.json> (or set KOKORO_PIPELINE_FIXTURE)")

    here = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.environ.get("KOKORO_PIPELINE_CORPUS", os.path.join(here, "..", "kokoro_g2p_oracle", "corpus.txt"))
    vocab = load_vocab(kokoro_dir())
    cases = build_cases(corpus_path)

    g2ps = {british: en.G2P(trf=False, british=british, fallback=None, unk="") for british in VOICES.values()}

    fixture_cases = []
    for case in cases:
        voices = {}
        for voice, british in VOICES.items():
            voices[voice] = chunks_for(g2ps[british], vocab, case["text"])
        fixture_cases.append({"name": case["name"], "text": case["text"], "voices": voices})

    with open(out, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "cases": fixture_cases}, f, ensure_ascii=False, indent=None, separators=(",", ":"))
        f.write("\n")
    print(f"wrote {out}: {len(fixture_cases)} cases x {len(VOICES)} voices")


if __name__ == "__main__":
    main()
