#!/usr/bin/env python3
"""Dumps the corpus's out-of-vocabulary words with the phonemes misaki's own
espeak fallback produced for them, for `examples/g2p_per_eval.rs`'s
model-vs-espeak comparison.

Reproduces exactly what `KPipeline(lang_code='a'|'b')` builds
(`kokoro/pipeline.py`): `misaki.en.G2P(trf=False, british=..., unk='',
fallback=misaki.espeak.EspeakFallback(british=...))`. Every line of the
corpus goes through both dialects' G2P; a word the lexicon could not resolve
gets `MToken._.rating == 2` (misaki's rating for the espeak fallback path),
and this script records that word's text and the fallback's phoneme output.

Output JSON:
    {"rows": [{"word": str, "dialect": "us"|"gb", "espeak_phonemes": str}, ...]}

Requires: misaki 0.9.4 with the espeak extra (`phonemizer`, `espeakng-loader`,
a system espeak-ng), spaCy with `en_core_web_sm`.

Environment:
    G2P_OOV_CORPUS   corpus path (default: kokoro_g2p_oracle/corpus.txt,
                     the sibling oracle's corpus, beside this script's
                     grandparent directory)

Usage:
    python3 crates/boostr-audio/tests/tools/g2p_oov_oracle/g2p_oov_oracle.py \\
        <out.json>
"""
import json
import os
import sys

from misaki import en, espeak


def main():
    if len(sys.argv) != 2:
        print("usage: g2p_oov_oracle.py <out.json>", file=sys.stderr)
        return 1
    out_path = sys.argv[1]

    here = os.path.dirname(os.path.abspath(__file__))
    default_corpus = os.path.join(here, "..", "kokoro_g2p_oracle", "corpus.txt")
    corpus_path = os.environ.get("G2P_OOV_CORPUS", default_corpus)
    with open(corpus_path, encoding="utf-8") as f:
        sentences = [line.rstrip("\n") for line in f if line.strip()]

    rows = []
    seen = set()
    for dialect, british in (("us", False), ("gb", True)):
        fallback = espeak.EspeakFallback(british=british)
        g2p = en.G2P(trf=False, british=british, fallback=fallback, unk="")
        for sentence in sentences:
            _, tokens = g2p(sentence)
            for tok in tokens:
                if tok._.rating != 2:
                    continue
                key = (dialect, tok.text)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "word": tok.text,
                    "dialect": dialect,
                    "espeak_phonemes": tok.phonemes or "",
                })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path}: {len(rows)} out-of-vocabulary words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
