#!/usr/bin/env python3
"""Build the grapheme-to-phoneme training set for boostr's g2p_lm model.

Reads misaki 0.9.4's four English lexicons (`{us,gb}_{gold,silver}.json`,
Apache-2.0, https://github.com/hexgrad/misaki) and Kokoro-82M's phoneme
vocab, then writes:

    vocab.json   the single source of truth for token ids. oxidizr's
                 training config and boostr's inference module both read
                 this file, so it never changes shape without updating
                 both.
    train.bin    oxidizr's flat little-endian u32 format (see
                 oxidizr/docs/data/format.md), cut into rows of exactly
                 `--row-tokens` tokens. Each row holds whole
                 `<bos> <dialect> graphemes... <sep> phonemes... <eos>`
                 sequences, shuffled with a fixed seed and packed greedily,
                 then padded with the pad id (0) to the row's end. No
                 sequence straddles a row boundary, and the file is a whole
                 number of rows.
    dev.jsonl    held-out gold pairs, one `{"dialect", "word", "phonemes"}`
                 object per line.
    test.jsonl   held-out gold pairs, same shape as dev.jsonl.

Casing: lexicon keys carry phonetic signal in their case (`ABS` the
acronym vs `abs` the word; `Anglo-American` the proper noun), and misaki's
own OOV fallback looks up the word exactly as written before trying any
lowercased variant. The grapheme vocab keeps case for the same reason: a
model reading raw, as-written text must see the same casing misaki sees.

Phoneme symbols: every symbol that occurs in any lexicon value (`DEFAULT`
or a POS-tagged variant) must already be a single-character key in
Kokoro's vocab; the build asserts this and reports any symbol it is
missing. Stress marks (`ˈ`, `ˌ`) are ordinary members of that same set.

Split: for each dialect, a word's fate is decided by
`sha256(word) mod 100` — under 5 is dev, under 10 is test, else train.
This is independent of dialect and of gold vs. silver, so reruns and
future additions land on the same side of the split. A silver entry
whose word falls in a dialect's held-out set is dropped entirely (not
placed in train, not placed in dev/test, which only ever come from gold).

Row contract: oxidizr's loader reads rows of `max_seq_len + 1` tokens at
fixed offsets (row r is tokens [r * (max_seq_len + 1), (r + 1) *
(max_seq_len + 1))), so `--row-tokens` must equal the training config's
`model.max_seq_len + 1`. Any other value misaligns every row after the
first. Pad (0) is never scored under the config's prefix_lm mask
(`target_base: 61`, `always_score: [3]`): 0 is below target_base and not
in always_score. Pads only follow a row's last `<eos>`, so under causal
attention no scored position attends to one.

Determinism: dict iteration order in this script always follows the
order the JSON parser returns (which follows the file's own key order,
stable across runs of the same input), the only shuffle applied before
packing train.bin uses a fixed seed, and packing is a deterministic
greedy pass over the shuffled order. Running this script twice
against the same inputs produces byte-identical output files.

Usage:
    python3 crates/boostr-audio/assets/g2p_lm/build_dataset.py <output_dir> [--row-tokens N]

    --row-tokens N   tokens per train.bin row (default 129 = max_seq_len
                     128 + 1). Must equal the training config's
                     `model.max_seq_len + 1`.

Environment:
    MISAKI_DATA_DIR  directory holding the four lexicon JSON files
                     (default: the installed `misaki.data` package, found
                     via `importlib.resources`)
    KOKORO_CONFIG    path to Kokoro-82M's `config.json`, whose `vocab`
                     field maps phoneme symbols to integer ids (required)
"""
import argparse
import hashlib
import importlib.resources
import json
import os
import random
import struct
import sys

FORMAT_VERSION = 1
SHUFFLE_SEED = 0
DEV_THRESHOLD = 5
TEST_THRESHOLD = 10
DEFAULT_ROW_TOKENS = 129
DIALECTS = ("us", "gb")
CATEGORIES = ("gold", "silver")


def load_lexicon(data_dir, name):
    """Read one lexicon JSON file, from MISAKI_DATA_DIR or the misaki package."""
    if data_dir:
        with open(os.path.join(data_dir, name), encoding="utf-8") as f:
            return json.load(f)
    from misaki import data
    return json.loads(importlib.resources.files(data).joinpath(name).read_text(encoding="utf-8"))


def resolve_default(value):
    """Return the DEFAULT phoneme string for a lexicon value, or None to skip it."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        default = value.get("DEFAULT")
        return default if isinstance(default, str) else None
    return None


def all_phoneme_strings(value):
    """Yield every phoneme string a lexicon value holds (DEFAULT and tagged)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            if isinstance(v, str):
                yield v


def word_bucket(word):
    """Deterministic 0-99 split bucket for a word, stable across dialects and reruns."""
    digest = hashlib.sha256(word.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 100


def build_vocab(lexicons, kokoro_vocab):
    """Build the id layout: specials, then graphemes, then phonemes.

    `lexicons` maps (dialect, category) -> raw JSON dict. `kokoro_vocab`
    maps phoneme symbol -> Kokoro's own integer id (used only to order
    our phoneme ids deterministically, not reused as-is).
    """
    grapheme_chars = set()
    phoneme_chars = set()
    for d in lexicons.values():
        for key, value in d.items():
            grapheme_chars.update(key)
            for ps in all_phoneme_strings(value):
                phoneme_chars.update(ps)

    missing = sorted(phoneme_chars - set(kokoro_vocab))
    assert not missing, f"phoneme symbols missing from Kokoro vocab: {missing}"

    specials = {"pad": 0, "bos": 1, "sep": 2, "eos": 3, "us": 4, "gb": 5}
    next_id = len(specials)

    graphemes = {}
    for ch in sorted(grapheme_chars):
        graphemes[ch] = next_id
        next_id += 1

    phonemes = {}
    target_base = next_id
    for sym in sorted(phoneme_chars, key=lambda s: kokoro_vocab[s]):
        phonemes[sym] = next_id
        next_id += 1

    return {
        "format_version": FORMAT_VERSION,
        "specials": specials,
        "graphemes": graphemes,
        "phonemes": phonemes,
        "target_base": target_base,
        "eos": specials["eos"],
        "vocab_size": next_id,
        "counts": {
            "num_specials": len(specials),
            "num_graphemes": len(graphemes),
            "num_phonemes": len(phonemes),
        },
    }


def collect_pairs(lexicons):
    """Return dialect -> category -> ordered list of (word, phonemes) DEFAULT pairs."""
    pairs = {}
    for dialect in DIALECTS:
        pairs[dialect] = {}
        for category in CATEGORIES:
            out = []
            for word, value in lexicons[(dialect, category)].items():
                ps = resolve_default(value)
                if ps is None:
                    continue
                out.append((word, ps))
            pairs[dialect][category] = out
    return pairs


def split_dialect(gold_pairs, silver_pairs):
    """Split one dialect's gold+silver pairs into train/dev/test.

    Dev and test come only from gold, chosen by `word_bucket`. A silver
    word whose bucket lands in gold's held-out ranges is dropped, not
    trained on and not added to dev/test.
    """
    dev, test, train = [], [], []
    held_out_words = set()
    for word, ps in gold_pairs:
        bucket = word_bucket(word)
        if bucket < DEV_THRESHOLD:
            dev.append((word, ps))
            held_out_words.add(word)
        elif bucket < TEST_THRESHOLD:
            test.append((word, ps))
            held_out_words.add(word)
        else:
            train.append((word, ps))
    for word, ps in silver_pairs:
        if word in held_out_words:
            continue
        train.append((word, ps))
    return train, dev, test


def encode_sequence(vocab, dialect, word, phonemes):
    """Token ids for `<bos> <dialect> graphemes... <sep> phonemes... <eos>`."""
    specials = vocab["specials"]
    graphemes = vocab["graphemes"]
    phoneme_ids = vocab["phonemes"]
    seq = [specials["bos"], specials[dialect]]
    seq.extend(graphemes[ch] for ch in word)
    seq.append(specials["sep"])
    seq.extend(phoneme_ids[sym] for sym in phonemes)
    seq.append(specials["eos"])
    return seq


def pack_rows(examples, row_tokens, pad_id):
    """Pack `(word, sequence)` examples greedily into rows of `row_tokens` ids.

    Sequences keep their order. A sequence that does not fit in the current
    row's free space starts the next row, and the current row is filled to
    `row_tokens` with `pad_id`. Returns the rows and the pad count.
    """
    rows = []
    current = []
    pads = 0

    def close_row():
        nonlocal pads
        pads += row_tokens - len(current)
        rows.append(current + [pad_id] * (row_tokens - len(current)))

    for word, seq in examples:
        if len(seq) > row_tokens:
            raise ValueError(
                f"sequence for word {word!r} is {len(seq)} tokens, longer than "
                f"--row-tokens {row_tokens}. Raise --row-tokens and the training "
                f"config's model.max_seq_len (row_tokens - 1) together."
            )
        if len(current) + len(seq) > row_tokens:
            close_row()
            current = []
        current.extend(seq)
    if current:
        close_row()
    return rows, pads


def write_bin(path, rows):
    """Write rows into oxidizr's flat little-endian u32 format."""
    with open(path, "wb") as f:
        for row in rows:
            f.write(struct.pack(f"<{len(row)}I", *row))


def parse_args():
    parser = argparse.ArgumentParser(description="Build the g2p_lm training set.")
    parser.add_argument("output_dir", help="directory for vocab.json, train.bin, dev.jsonl, test.jsonl")
    parser.add_argument(
        "--row-tokens",
        type=int,
        default=DEFAULT_ROW_TOKENS,
        help="tokens per train.bin row; must equal the training config's max_seq_len + 1 "
        f"(default {DEFAULT_ROW_TOKENS})",
    )
    args = parser.parse_args()
    if args.row_tokens < 2:
        parser.error(f"--row-tokens {args.row_tokens} is below 2: a row needs one input and one target")
    return args


def main():
    args = parse_args()
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    misaki_data_dir = os.environ.get("MISAKI_DATA_DIR")
    kokoro_config_path = os.environ.get("KOKORO_CONFIG")
    if not kokoro_config_path:
        print("error: KOKORO_CONFIG must point at Kokoro-82M's config.json", file=sys.stderr)
        return 1

    lexicons = {}
    for dialect in DIALECTS:
        for category in CATEGORIES:
            lexicons[(dialect, category)] = load_lexicon(misaki_data_dir, f"{dialect}_{category}.json")

    with open(kokoro_config_path, encoding="utf-8") as f:
        kokoro_vocab = json.load(f)["vocab"]
    multi_char = [s for s in kokoro_vocab if len(s) != 1]
    assert not multi_char, f"Kokoro vocab has multi-char symbols, single-char split assumption breaks: {multi_char}"

    vocab = build_vocab(lexicons, kokoro_vocab)
    pairs = collect_pairs(lexicons)

    train_pairs = []  # (dialect, word, phonemes)
    dev_pairs = []
    test_pairs = []
    for dialect in DIALECTS:
        train, dev, test = split_dialect(pairs[dialect]["gold"], pairs[dialect]["silver"])
        train_pairs.extend((dialect, w, p) for w, p in train)
        dev_pairs.extend((dialect, w, p) for w, p in dev)
        test_pairs.extend((dialect, w, p) for w, p in test)

    examples = [(word, encode_sequence(vocab, dialect, word, ps)) for dialect, word, ps in train_pairs]
    random.Random(SHUFFLE_SEED).shuffle(examples)
    sequences = [seq for _, seq in examples]
    try:
        rows, pads = pack_rows(examples, args.row_tokens, vocab["specials"]["pad"])
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    max_len = max((len(s) for s in sequences), default=0)
    max_len = max(max_len, max((len(word) + len(ps) + 3 for _, word, ps in dev_pairs), default=0))
    max_len = max(max_len, max((len(word) + len(ps) + 3 for _, word, ps in test_pairs), default=0))

    vocab_path = os.path.join(out_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2, sort_keys=False)

    train_path = os.path.join(out_dir, "train.bin")
    write_bin(train_path, rows)

    dev_path = os.path.join(out_dir, "dev.jsonl")
    with open(dev_path, "w", encoding="utf-8") as f:
        for dialect, word, ps in sorted(dev_pairs, key=lambda t: (t[0], t[1])):
            f.write(json.dumps({"dialect": dialect, "word": word, "phonemes": ps}, ensure_ascii=False) + "\n")

    test_path = os.path.join(out_dir, "test.jsonl")
    with open(test_path, "w", encoding="utf-8") as f:
        for dialect, word, ps in sorted(test_pairs, key=lambda t: (t[0], t[1])):
            f.write(json.dumps({"dialect": dialect, "word": word, "phonemes": ps}, ensure_ascii=False) + "\n")

    print(f"vocab: {vocab_path} ({vocab['vocab_size']} ids, "
          f"{vocab['counts']['num_graphemes']} graphemes, {vocab['counts']['num_phonemes']} phonemes)")
    for dialect in DIALECTS:
        n_train = sum(1 for d, _, _ in train_pairs if d == dialect)
        n_dev = sum(1 for d, _, _ in dev_pairs if d == dialect)
        n_test = sum(1 for d, _, _ in test_pairs if d == dialect)
        print(f"{dialect}: train {n_train}, dev {n_dev}, test {n_test}")
    print(f"total: train {len(train_pairs)}, dev {len(dev_pairs)}, test {len(test_pairs)}")
    print(f"max sequence length: {max_len} tokens")
    row_total = len(rows) * args.row_tokens
    pad_fraction = pads / row_total if row_total else 0.0
    print(f"train.bin: {train_path} ({len(rows)} rows of {args.row_tokens} tokens, "
          f"{row_total} tokens, {sum(len(s) for s in sequences)} sequence tokens, "
          f"{os.path.getsize(train_path)} bytes)")
    print(f"padding: {pads} pad tokens, {pad_fraction:.2%} of train.bin")
    print(f"dev.jsonl: {dev_path} ({os.path.getsize(dev_path)} bytes)")
    print(f"test.jsonl: {test_path} ({os.path.getsize(test_path)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
