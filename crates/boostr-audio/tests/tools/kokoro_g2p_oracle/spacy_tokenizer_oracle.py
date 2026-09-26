#!/usr/bin/env python3
"""Oracle fixture for the spaCy tokenizer port
(`boostr_audio::g2p::spacy_tokenizer`).

Runs the tokenizer of spaCy's `en_core_web_sm` 3.8.0 (`nlp.tokenizer`, the
step before `tok2vec`) over every line of `corpus.txt`, as misaki passes it
(`G2P.preprocess` strips leading whitespace), and over the stress texts
below. For each token it records the text and the six lexical attributes
the pipeline's `tok2vec` embeds.

Output JSON, one case per line:
    {"source": str, "cases": [case, ...]}
    case = {
      "set": "corpus" | "stress",
      "text": str,
      "tokens": [[text, spacy, norm, prefix, suffix, shape, is_space], ...],
    }
`spacy` is `token.whitespace_ == " "`; the rest are `norm_`, `prefix_`,
`suffix_`, `shape_` and `is_space`.

The Rust integration test `crates/boostr-audio/tests/spacy_tokenizer_oracle.rs`
asserts exact equality.

Requires: spaCy 3.8 with `en_core_web_sm` 3.8.0.

Environment:
    KOKORO_G2P_CORPUS     corpus path (default: corpus.txt beside this script)
    SPACY_TOKENIZER_FIXTURE  output path (default: argv[1])

Usage:
    python3 crates/boostr-audio/tests/tools/kokoro_g2p_oracle/spacy_tokenizer_oracle.py \\
        crates/boostr-audio/tests/fixtures/spacy_tokenizer_oracle.json
"""
import json
import os
import sys

import spacy

STRESS = [
    # URLs and emails
    "Visit https://www.example.com/path?q=1&r=2#frag today.",
    "See http://localhost:8080/api/v1/users, or ftp://files.example.org/a.txt.",
    "(www.example.co.uk/page) and <https://docs.rs/regex>.",
    "Mail me at john.doe+tag@example.com or support@sub.domain.org.",
    "IPs like 192.168.0.1 and 10.0.0.255:8080/x are private.",
    # emoticons and special cases with affixes
    "Great :) but sad :( and :-) or ;) or <3 xD :P (:",
    "o.O O.o -_- ^_^ :') =) :] :-/ ¯\\_(ツ)_/¯",
    # hyphenation
    "A well-known, state-of-the-art, mother-in-law's co-op.",
    "Pages 10-20, 1990-1995, and -5 to +7 degrees.",
    "e-mail, re-enter, x-ray, T-shirt, pre- and post-war.",
    # contractions
    "I don't know, can't say, won't go, shouldn't've, y'all're gonna wanna.",
    "She's here; they're there; we'll see; I'd go; it's fine; you've won.",
    "Don't DON'T don’t CAN'T Can't ain't lovin' 'cause 'em.",
    "gonna gotta wanna lemme gimme dunno Gonna",
    # abbreviations
    "The U.S. and U.K. met, e.g. at 3 p.m., i.e. late, etc. Dr. Smith, Jr.",
    "Mr. and Mrs. Jones vs. Ms. Lee; Jan. Feb. Sept. Inc. Ltd. Co. St.",
    "(U.S.) [e.g.] \"etc.\" a.m. A.M. p.m.",
    # numbers, currency, units, percent
    "Pi is 3.14, e is 2.718, and 1,000,000 is big.",
    "It cost $1.5M, then €20, £3.50, ¥100, and US$5.",
    "Growth was 13.5% and 7%; temp 20°C, 68°F, 5km, 10kg, 3.5GB.",
    "Scores: 3-2, 1/2, 3:45pm, 10:30 a.m., 2nd, 3rd, 21st, 100th.",
    "Call 555-1234 or +1 (800) 555-0199 ext. 42.",
    # ellipses, quotes, dashes
    "Wait... what?! Really?? No!!! Hmm…",
    "He said \"hello\" and 'goodbye' and “curly” and ‘single’.",
    "Word—word – word -- word --- word—",
    "‘’ ’’ “” ``quoted'' ,, „low“ «guillemets» ‹single›",
    # unicode punctuation
    "¿Qué? ¡Hola! 你好。 Ünïcödé naïve café résumé",
    "Math: 2+2=4, 3*4, 6/2, 2^3, x<y>z, a|b, ~tilde, #hash, @at, &amp",
    # whitespace
    "Multiple   spaces  here.",
    "Tabs\tand\t\ttabs.",
    "New\nline and\n\nparagraph.",
    "  leading and trailing  ",
    " \t mixed \n whitespace   nbsp em　ideo ",
    "a b   c\x1cd",
    "",
    " ",
    "\n",
    # misc
    "Let's go! Let’s go!",
    "C++ and C# and .NET and node.js",
    "#hashtag @mention $TICKER 50% off!!!",
    "(parenthetical) [bracketed] {braced} <angled>",
    "rock'n'roll 'n' ol' 90's '90s",
    "shouldn't've y'all'd've",
    "’’Hi’’ said he",
    "A" * 120 + " done",
]


def token_row(t):
    return [t.text, t.whitespace_ == " ", t.norm_, t.prefix_, t.suffix_, t.shape_, t.is_space]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    corpus = os.environ.get("KOKORO_G2P_CORPUS", os.path.join(here, "corpus.txt"))
    out = os.environ.get("SPACY_TOKENIZER_FIXTURE") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if out is None:
        sys.exit("usage: spacy_tokenizer_oracle.py <fixture.json> (or set SPACY_TOKENIZER_FIXTURE)")
    with open(corpus, encoding="utf-8") as f:
        sentences = [line.rstrip("\n").lstrip() for line in f if line.strip()]

    nlp = spacy.load("en_core_web_sm")
    source = f"spaCy {spacy.__version__}, {nlp.meta['lang']}_{nlp.meta['name']} {nlp.meta['version']}"
    cases = [("corpus", s) for s in sentences] + [("stress", s) for s in STRESS]
    rows = [{"set": kind, "text": text, "tokens": [token_row(t) for t in nlp.tokenizer(text)]}
            for kind, text in cases]

    # One case per line keeps the fixture compact and its diffs readable.
    with open(out, "w", encoding="utf-8") as f:
        f.write('{"source": ' + json.dumps(source) + ', "cases": [\n')
        f.write(",\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in rows))
        f.write("\n]}\n")
    print(f"wrote {out}: {len(sentences)} corpus + {len(STRESS)} stress texts")


if __name__ == "__main__":
    main()
