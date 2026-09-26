#!/usr/bin/env python3
"""Oracle fixture for the misaki English G2P port (`boostr_audio::g2p::misaki`).

Runs upstream misaki `en.G2P(trf=False, british=<bool>, fallback=None)` over
every line of `corpus.txt`, for US and GB. For each sentence it records the
tokens misaki builds from spaCy `en_core_web_sm` (text, PTB fine tag,
trailing whitespace) and the G2P output for two `unk` strings: the misaki
default `❓` and the empty string Kokoro passes.

Output JSON, one case per line:
    {"cases": [case, ...]}
    case = {
      "sentence": str,
      "tokens": [[text, tag, whitespace], ...],
      "us": result,
      "gb": result,
    }
    result = {
      "phonemes": str,              # unk = "❓"
      "phonemes_kokoro": str,       # unk = ""
      "token_phonemes": [[text, whitespace, phonemes, rating], ...],
    }
`tokens` is identical for both dialects. `token_phonemes` holds misaki's
final tokens for unk = "❓"; `rating` is `MToken._.rating`.

The Rust integration test `crates/boostr-audio/tests/misaki_g2p_oracle.rs`
feeds `tokens` to the Rust G2P and asserts exact equality.

Requires: misaki 0.9.4, spaCy with `en_core_web_sm`.

Environment:
    KOKORO_G2P_CORPUS   corpus path (default: corpus.txt beside this script)
    KOKORO_G2P_FIXTURE  output path (default: argv[1])

Usage:
    python3 crates/boostr-audio/tests/tools/kokoro_g2p_oracle/kokoro_g2p_oracle.py \
        crates/boostr-audio/tests/fixtures/misaki_g2p_oracle.json
"""
import json
import os
import sys

from misaki import en


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    corpus = os.environ.get("KOKORO_G2P_CORPUS", os.path.join(here, "corpus.txt"))
    out = os.environ.get("KOKORO_G2P_FIXTURE") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if out is None:
        sys.exit("usage: kokoro_g2p_oracle.py <fixture.json> (or set KOKORO_G2P_FIXTURE)")
    with open(corpus, encoding="utf-8") as f:
        sentences = [line.rstrip("\n") for line in f if line.strip()]

    dialects = {}
    for dialect, british in (("us", False), ("gb", True)):
        g2p = en.G2P(trf=False, british=british, fallback=None)
        rows = []
        for sentence in sentences:
            text, words, features = en.G2P.preprocess(sentence)
            if features:
                sys.exit(f"markdown link features are out of scope: {sentence!r}")
            tokens = [[t.text, t.tag, t.whitespace] for t in g2p.tokenize(text, words, features)]
            g2p.unk = "❓"
            phonemes, final = g2p(sentence)
            g2p.unk = ""
            phonemes_kokoro, _ = g2p(sentence)
            rows.append((tokens, {
                "phonemes": phonemes,
                "phonemes_kokoro": phonemes_kokoro,
                "token_phonemes": [[t.text, t.whitespace, t.phonemes, t._.rating] for t in final],
            }))
        dialects[dialect] = rows

    # One case per line keeps the fixture compact and its diffs readable.
    with open(out, "w", encoding="utf-8") as f:
        f.write('{"cases": [\n')
        for i, sentence in enumerate(sentences):
            us_tokens, us = dialects["us"][i]
            gb_tokens, gb = dialects["gb"][i]
            if us_tokens != gb_tokens:
                sys.exit(f"spaCy tokens differ between runs: {sentence!r}")
            case = {"sentence": sentence, "tokens": us_tokens, "us": us, "gb": gb}
            f.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")))
            f.write(",\n" if i + 1 < len(sentences) else "\n")
        f.write("]}\n")
    print(f"wrote {out}: {len(sentences)} sentences x 2 dialects")


if __name__ == "__main__":
    main()
