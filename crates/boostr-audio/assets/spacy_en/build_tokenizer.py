#!/usr/bin/env python3
"""Build the tokenizer data for `boostr_audio::g2p::spacy_tokenizer`.

Reads the tokenizer of the installed spaCy pipeline `en_core_web_sm` 3.8.0
(spaCy 3.8.x, MIT, https://github.com/explosion/spaCy) and writes one JSON
file holding everything `Tokenizer.__call__` and the lexeme attribute getters
read:

    {
      "source":           str,         # spaCy and pipeline versions
      "faster_heuristics": bool,
      "prefix":           str,         # prefix_search, for a leftmost search
      "suffix":           str,         # suffix_search, for a leftmost search
      "infix":            str,         # infix_finditer, for find_iter
      "token_match":      str | null,  # token_match, anchored at the start
      "url_match":        str | null,  # url_match, anchored at the start
      "rules":            [[chunk, [[orth, norm | null], ...]], ...],
      "base_norms":       [[string, norm], ...],
      "lexeme_norm":      [[hash, norm], ...],
    }

`rules` is `nlp.tokenizer.rules`, sorted by chunk. `base_norms` is
`spacy.lang.norm_exceptions.BASE_NORMS`. `lexeme_norm` is the pipeline's
`lexeme_norm` lookups table, keyed by spaCy `hash_string` (MurmurHash64A,
seed 1, over UTF-8), sorted by hash. The table ships without its key strings.

Pattern translation. The Python patterns go through `re._parser` and come
out in the syntax of the `regexr` crate:
  - every literal outside ASCII alphanumerics is written `\\x{H}`;
  - every character class, `\\d`, `\\w` and `\\S` is resolved to explicit
    code point ranges with this interpreter's `str.isdecimal`, `str.isalnum`
    and `str.isspace`, which are what CPython `re` uses in Unicode mode;
  - `.` without DOTALL becomes `[^\\n]`;
  - groups become non-capturing (the tokenizer reads match bounds only);
  - `^` and `$` stay as they are: in `regexr`, as in `re`, `$` also matches
    before a final newline.
`token_match` and `url_match` are wrapped in `^(?:...)` because the tokenizer
calls them through `re.match`. The script refuses any construct outside that
set, and any flag besides `re.UNICODE`.

The output is deterministic for a fixed spaCy, pipeline and Python build.

Usage:
    python3 crates/boostr-audio/assets/spacy_en/build_tokenizer.py \\
        crates/boostr-audio/assets/spacy_en/tokenizer.json
Environment:
    SPACY_TOKENIZER_OUT  output path (overrides argv[1])
"""
import json
import os
import re
import sys
from re import _constants as C
from re import _parser as P

import spacy
from spacy.attrs import NORM, ORTH
from spacy.lang.norm_exceptions import BASE_NORMS
from spacy.strings import hash_string
from spacy.symbols import IDS as SYMBOL_IDS

MAX_CP = 0x10FFFF


def char_set(pred):
    """Code point ranges (inclusive pairs) where `pred(chr(cp))` holds."""
    out = []
    start = None
    for cp in range(MAX_CP + 2):
        ok = cp <= MAX_CP and not (0xD800 <= cp <= 0xDFFF) and pred(chr(cp))
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append((start, cp - 1))
            start = None
    return out


def complement(ranges):
    out = []
    lo = 0
    for a, b in ranges:
        if a > lo:
            out.append((lo, a - 1))
        lo = b + 1
    if lo <= MAX_CP:
        out.append((lo, MAX_CP))
    return out


def normalize(ranges):
    out = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


DIGIT = char_set(str.isdecimal)
WORD = normalize(char_set(str.isalnum) + [(ord("_"), ord("_"))])
SPACE = char_set(str.isspace)
CATEGORIES = {
    C.CATEGORY_DIGIT: DIGIT,
    C.CATEGORY_NOT_DIGIT: complement(DIGIT),
    C.CATEGORY_WORD: WORD,
    C.CATEGORY_NOT_WORD: complement(WORD),
    C.CATEGORY_SPACE: SPACE,
    C.CATEGORY_NOT_SPACE: complement(SPACE),
}


def lit(cp):
    c = chr(cp)
    if c.isascii() and c.isalnum():
        return c
    return "\\x{%X}" % cp


def cls(ranges):
    """A class over `ranges`, negated when that is shorter."""
    ranges = normalize(ranges)
    neg = complement(ranges)
    body, prefix = (neg, "^") if len(neg) < len(ranges) else (ranges, "")
    if not body:
        # Python `re` never builds an empty or full class from these patterns.
        raise SystemExit("empty or full character class")
    parts = []
    for a, b in body:
        parts.append(lit(a) if a == b else f"{lit(a)}-{lit(b)}")
    return "[" + prefix + "".join(parts) + "]"


def class_ranges(items):
    negate = False
    ranges = []
    for op, av in items:
        if op == C.NEGATE:
            negate = True
        elif op == C.LITERAL:
            ranges.append((av, av))
        elif op == C.RANGE:
            ranges.append(av)
        elif op == C.CATEGORY:
            ranges.extend(CATEGORIES[av])
        else:
            raise SystemExit(f"unsupported class item {op}")
    ranges = normalize(ranges)
    return complement(ranges) if negate else ranges


def repeat(lo, hi, lazy):
    hi_s = "" if hi == C.MAXREPEAT else str(hi)
    if (lo, hi) == (0, C.MAXREPEAT):
        q = "*"
    elif (lo, hi) == (1, C.MAXREPEAT):
        q = "+"
    elif (lo, hi) == (0, 1):
        q = "?"
    elif lo == hi:
        q = "{%d}" % lo
    else:
        q = "{%d,%s}" % (lo, hi_s)
    return q + ("?" if lazy else "")


def emit(seq):
    out = []
    for op, av in seq:
        if op == C.LITERAL:
            out.append(lit(av))
        elif op == C.NOT_LITERAL:
            out.append(cls(complement([(av, av)])))
        elif op == C.ANY:
            out.append(cls(complement([(10, 10)])))
        elif op == C.IN:
            out.append(cls(class_ranges(av)))
        elif op == C.BRANCH:
            out.append("(?:" + "|".join(emit(a) for a in av[1]) + ")")
        elif op == C.SUBPATTERN:
            _group, add_flags, del_flags, body = av
            if add_flags or del_flags:
                raise SystemExit("inline flag groups are unsupported")
            out.append("(?:" + emit(body) + ")")
        elif op in (C.MAX_REPEAT, C.MIN_REPEAT):
            lo, hi, body = av
            out.append("(?:" + emit(body) + ")" + repeat(lo, hi, op == C.MIN_REPEAT))
        elif op in (C.ASSERT, C.ASSERT_NOT):
            direction, body = av
            head = {(C.ASSERT, 1): "?=", (C.ASSERT, -1): "?<=",
                    (C.ASSERT_NOT, 1): "?!", (C.ASSERT_NOT, -1): "?<!"}[(op, direction)]
            out.append("(" + head + emit(body) + ")")
        elif op == C.AT:
            if av == C.AT_BEGINNING:
                out.append("^")
            elif av == C.AT_END:
                out.append("$")
            else:
                raise SystemExit(f"unsupported anchor {av}")
        else:
            raise SystemExit(f"unsupported regex construct {op}")
    return "".join(out)


def translate(compiled, anchored):
    if compiled is None:
        return None
    if compiled.flags & ~re.UNICODE:
        raise SystemExit(f"unsupported flags {compiled.flags:#x}")
    body = emit(P.parse(compiled.pattern, compiled.flags))
    return f"^(?:{body})" if anchored else body


def pattern_of(bound):
    return None if bound is None else bound.__self__


def main():
    out = os.environ.get("SPACY_TOKENIZER_OUT") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if out is None:
        sys.exit("usage: build_tokenizer.py <tokenizer.json> (or set SPACY_TOKENIZER_OUT)")

    nlp = spacy.load("en_core_web_sm")
    tok = nlp.tokenizer

    rules = []
    for chunk, subs in sorted(tok.rules.items()):
        row = []
        for spec in subs:
            extra = set(spec) - {ORTH, NORM}
            if extra:
                sys.exit(f"rule {chunk!r} sets attributes besides ORTH and NORM: {extra}")
            row.append([spec[ORTH], spec.get(NORM)])
        if "".join(o for o, _ in row) != chunk:
            sys.exit(f"rule {chunk!r}: ORTH pieces do not concatenate to the chunk")
        rules.append([chunk, row])

    table = nlp.vocab.lookups.get_table("lexeme_norm")
    lexeme_norm = sorted([int(k), v] for k, v in table.items())
    # The Rust side hashes every string. That is exact only if no table key
    # is a symbol ID and no symbol string hashes onto a table key.
    symbol_hashes = {hash_string(s) for s in SYMBOL_IDS if s}
    for k, _ in lexeme_norm:
        if k < 2**32 or k in symbol_hashes:
            sys.exit(f"lexeme_norm key {k} collides with the symbol table")

    data = {
        "source": f"spaCy {spacy.__version__}, {nlp.meta['lang']}_{nlp.meta['name']} "
                  f"{nlp.meta['version']}",
        "faster_heuristics": bool(tok.faster_heuristics),
        "prefix": translate(pattern_of(tok.prefix_search), False),
        "suffix": translate(pattern_of(tok.suffix_search), False),
        "infix": translate(pattern_of(tok.infix_finditer), False),
        "token_match": translate(pattern_of(tok.token_match), True),
        "url_match": translate(pattern_of(tok.url_match), True),
        "rules": rules,
        "base_norms": sorted([k, v] for k, v in BASE_NORMS.items()),
        "lexeme_norm": lexeme_norm,
    }
    # One scalar field or array element per line keeps diffs readable.
    def dump(v):
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))

    lines = []
    for key, value in data.items():
        if isinstance(value, list):
            body = ",\n".join(dump(x) for x in value)
            lines.append(f"{dump(key)}:[\n{body}\n]")
        else:
            lines.append(f"{dump(key)}:{dump(value)}")
    with open(out, "w", encoding="utf-8") as f:
        f.write("{\n" + ",\n".join(lines) + "\n}\n")
    print(f"wrote {out}: {len(rules)} rules, {len(lexeme_norm)} lexeme norms")


if __name__ == "__main__":
    main()
