#!/usr/bin/env python3
"""Oracle fixture for the spaCy tagger port
(`boostr_audio::g2p::spacy_tagger`).

Runs `en_core_web_sm` 3.8.0 as misaki loads it
(`spacy.load("en_core_web_sm", enable=["tok2vec", "tagger"])`) over every
line of `corpus.txt`, as misaki passes it (`G2P.preprocess` strips leading
whitespace), and over the stress texts of `spacy_tokenizer_oracle.py`. For
each token it records the text and `Token.tag_`. For the first
`VECTOR_TEXTS` corpus sentences it also records the `tok2vec` output
(`Doc.tensor`).

Output JSON, one case per line:
    {"source": str, "width": int, "cases": [case, ...]}
    case = {
      "set": "corpus" | "stress",
      "text": str,
      "tokens": [[text, tag], ...],
      "tok2vec": [float, ...]        # optional, n_tokens * width, row-major
    }

The Rust integration test `crates/boostr-audio/tests/spacy_tagger_oracle.rs`
asserts exact tags and `tok2vec` within 1e-5.

Requires: spaCy 3.8 with `en_core_web_sm` 3.8.0.

Environment:
    KOKORO_G2P_CORPUS     corpus path (default: corpus.txt beside this script)
    SPACY_TAGGER_FIXTURE  output path (default: argv[1])

Usage:
    python3 crates/boostr-audio/tests/tools/kokoro_g2p_oracle/spacy_tagger_oracle.py \\
        crates/boostr-audio/tests/fixtures/spacy_tagger_oracle.json
"""
import json
import os
import sys

import numpy as np
import spacy

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spacy_tokenizer_oracle import STRESS  # noqa: E402

VECTOR_TEXTS = 5


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    corpus = os.environ.get("KOKORO_G2P_CORPUS", os.path.join(here, "corpus.txt"))
    out = os.environ.get("SPACY_TAGGER_FIXTURE") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if out is None:
        sys.exit("usage: spacy_tagger_oracle.py <fixture.json> (or set SPACY_TAGGER_FIXTURE)")
    with open(corpus, encoding="utf-8") as f:
        sentences = [line.rstrip("\n").lstrip() for line in f if line.strip()]

    nlp = spacy.load("en_core_web_sm", enable=["tok2vec", "tagger"])
    if nlp.pipe_names != ["tok2vec", "tagger"]:
        sys.exit(f"unexpected pipeline {nlp.pipe_names}")
    source = f"spaCy {spacy.__version__}, {nlp.meta['lang']}_{nlp.meta['name']} {nlp.meta['version']}"
    width = nlp.get_pipe("tok2vec").model.get_dim("nO")
    cases = [("corpus", s) for s in sentences] + [("stress", s) for s in STRESS]

    rows = []
    for i, (kind, text) in enumerate(cases):
        doc = nlp(text)
        row = {"set": kind, "text": text, "tokens": [[t.text, t.tag_] for t in doc]}
        if kind == "corpus" and i < VECTOR_TEXTS:
            tensor = np.asarray(doc.tensor, dtype=np.float32)
            if tensor.shape != (len(doc), width):
                sys.exit(f"Doc.tensor has shape {tensor.shape} for {text!r}")
            row["tok2vec"] = [float(x) for x in tensor.reshape(-1)]
        rows.append(row)

    # One case per line keeps the fixture compact and its diffs readable.
    with open(out, "w", encoding="utf-8") as f:
        f.write('{"source": ' + json.dumps(source) + ', "width": ' + str(width) + ', "cases": [\n')
        f.write(",\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in rows))
        f.write("\n]}\n")
    print(f"wrote {out}: {len(sentences)} corpus + {len(STRESS)} stress texts, "
          f"{VECTOR_TEXTS} with tok2vec")


if __name__ == "__main__":
    main()
