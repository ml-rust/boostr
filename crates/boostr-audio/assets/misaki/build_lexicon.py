#!/usr/bin/env python3
"""Build the embedded misaki English lexicons for `boostr_audio::g2p::misaki`.

Reads `{us,gb}_{gold,silver}.json` from the installed misaki 0.9.4 package
(Apache-2.0, https://github.com/hexgrad/misaki) and writes one zlib stream
per dialect: `en_us.lex.zlib` and `en_gb.lex.zlib`, beside this script.

The entries are stored as upstream ships them. The Rust loader applies
misaki's `Lexicon.grow_dictionary`. This script asserts every key is ASCII,
so ASCII case mapping in Rust reproduces Python's `lower` and `capitalize`.

Decompressed layout (all integers little-endian, `varint` = LEB128 u32):
    magic   b"MSKL" + u8 format version (1)
    tags    u8 count, then per tag: varint len + UTF-8 bytes
    gold    u32 entry count, then entries sorted by key bytes
    silver  u32 entry count, then entries sorted by key bytes
    entry   u8 bytes shared with the previous key in the section (0 for the
            first), varint suffix len + suffix bytes, u8 kind
            kind 0: varint len + phoneme bytes
            kind 1: u8 n, then n x (u8 tag index, u8 present,
                    [varint len + phoneme bytes when present == 1])

The output is deterministic for a fixed input and zlib build.

Usage:
    python3 crates/boostr-audio/assets/misaki/build_lexicon.py
Environment:
    MISAKI_DATA_DIR  directory holding the four JSON files (default: the
                     installed `misaki.data` package)
"""
import importlib.resources
import json
import os
import struct
import zlib

FORMAT_VERSION = 1


def varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def blob(s):
    b = s.encode("utf-8")
    return varint(len(b)) + b


def load(data_dir, name):
    if data_dir:
        with open(os.path.join(data_dir, name), encoding="utf-8") as f:
            return json.load(f)
    from misaki import data
    return json.loads(importlib.resources.files(data).joinpath(name).read_text(encoding="utf-8"))


def encode_section(d, tag_index):
    out = bytearray(struct.pack("<I", len(d)))
    prev = b""
    for key in sorted(d, key=lambda k: k.encode("utf-8")):
        assert key.isascii(), key
        v = d[key]
        kb = key.encode("utf-8")
        shared = 0
        while shared < min(len(prev), len(kb), 255) and prev[shared] == kb[shared]:
            shared += 1
        out.append(shared)
        out += varint(len(kb) - shared) + kb[shared:]
        prev = kb
        if isinstance(v, str):
            out.append(0)
            out += blob(v)
        else:
            out.append(1)
            out.append(len(v))
            for tag, ps in v.items():
                out.append(tag_index[tag])
                if ps is None:
                    out.append(0)
                else:
                    out.append(1)
                    out += blob(ps)
    return bytes(out)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.environ.get("MISAKI_DATA_DIR")
    for dialect in ("us", "gb"):
        gold = load(data_dir, f"{dialect}_gold.json")
        silver = load(data_dir, f"{dialect}_silver.json")
        tags = sorted({t for d in (gold, silver) for v in d.values() if isinstance(v, dict) for t in v})
        tag_index = {t: i for i, t in enumerate(tags)}
        raw = bytearray(b"MSKL")
        raw.append(FORMAT_VERSION)
        raw.append(len(tags))
        for t in tags:
            raw += blob(t)
        raw += encode_section(gold, tag_index)
        raw += encode_section(silver, tag_index)
        packed = zlib.compress(bytes(raw), 9)
        path = os.path.join(here, f"en_{dialect}.lex.zlib")
        with open(path, "wb") as f:
            f.write(packed)
        print(f"{path}: {len(gold)} gold, {len(silver)} silver, {len(raw)} raw bytes, {len(packed)} packed bytes")


if __name__ == "__main__":
    main()
