#!/usr/bin/env python3
"""Oracle fixture for the VITS (MMS-TTS) character tokenizer unit test.

Runs HF `transformers.models.vits.tokenization_vits.VitsTokenizer` (via
`VitsTokenizer.from_pretrained`) over a fixed list of strings covering mixed
case, punctuation, digits, apostrophes, hyphens, an en dash, characters
outside the vocabulary, an empty string, and repeated spaces. Writes a JSON
list of `{"text": ..., "input_ids": [...]}` objects, one per case, to the
path given on the command line. Rust's `vits_tokenizer_oracle` integration
test (`crates/boostr-audio/tests/vits_tokenizer_oracle.rs`) asserts exact id
equality against this fixture.

Requires: transformers. Reads the checkpoint directory from
`$MMS_TTS_ZLM_DIR`, or `$BOOSTR_MODELS_DIR/facebook/mms-tts-zlm` when that is
unset; the directory needs `vocab.json`, `tokenizer_config.json`, and
`special_tokens_map.json`.

Usage:
    python3 crates/boostr-audio/tests/tools/vits_tokenizer_oracle/vits_tokenizer_oracle.py \
        crates/boostr-audio/tests/fixtures/vits_tokenizer_zlm.json
"""
import json
import os
import sys


CASES = [
    "Hello world",
    "SELAMAT PAGI",
    "Selamat Pagi, Malaysia!",
    "  multiple   spaces  ",
    "",
    "   ",
    "abc-def",
    "a's",
    "don't stop",
    "well-known co-worker",
    "5 6 7",
    "tahun 2024 ini",
    "harga RM5.50 sahaja",
    "café naïve", # accented Latin letters outside the vocabulary
    "\U0001f600 test", # emoji outside the vocabulary
    "你好 world", # CJK outside the vocabulary
    "en–dash", # en dash, present in the zlm vocabulary
    "Kuala Lumpur",
    "Selamat pagi, apa khabar?",
    "UPPER lower MiXeD",
    "one\ttab\nnewline",
    "trailing punctuation!!!",
    "...leading punctuation",
    "quotes \"like this\" and 'this'",
    "numbers 0123456789",
    "under_score",
    "a",
    "A",
    "-",
    "'",
    "–",
    "y",
    "Y",
    "многоязычный", # Cyrillic, entirely outside the vocabulary
    "malaysia boleh!",
    "1-2-3 go",
    "double  space  test",
    "tab\tand space",
    "  ",
    "mixed CASE with-hyphen and'apostrophe",
    "zzz zzz",
]


def resolve_dir():
    d = os.environ.get("MMS_TTS_ZLM_DIR")
    if d:
        return d
    root = os.environ.get("BOOSTR_MODELS_DIR")
    if root:
        return os.path.join(root, "facebook", "mms-tts-zlm")
    raise SystemExit(
        "set MMS_TTS_ZLM_DIR to the facebook/mms-tts-zlm checkpoint directory, "
        "or BOOSTR_MODELS_DIR to the models root that contains facebook/mms-tts-zlm"
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <out.json>")
    out_path = sys.argv[1]

    from transformers import VitsTokenizer

    checkpoint_dir = resolve_dir()
    tokenizer = VitsTokenizer.from_pretrained(checkpoint_dir)

    fixture = []
    for text in CASES:
        input_ids = tokenizer(text)["input_ids"]
        fixture.append({"text": text, "input_ids": input_ids})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"wrote {len(fixture)} cases to {out_path}")


if __name__ == "__main__":
    main()
