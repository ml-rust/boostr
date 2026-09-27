#!/usr/bin/env python3
"""Pack a trained g2p_lm checkpoint for `boostr_audio::g2p::neural`'s
`g2p-neural` feature (`src/g2p/neural/embedded.rs`, `include_bytes!`).

Reads oxidizr's checkpoint directory (`config.json` plus one or more
`*.safetensors` files, `oxidizr/src/trainer.rs` `save_blazr_config_if_needed`
and `save_checkpoint_named`) and a `vocab.json`
(`assets/g2p_lm/build_dataset.py`'s output), then writes, into the output
directory:

    model.safetensors.zlib   the checkpoint's SafeTensors bytes, zlib-compressed
    config.json              copied verbatim
    vocab.json                copied verbatim

Checkpoint selection, mirroring
`crates/boostr-audio/src/g2p/neural/checkpoint.rs`: the sole `*.safetensors`
file if there is only one, else the one named `*_final.safetensors`, else
the one with the highest `_step_<N>` suffix.

Usage:
    python3 crates/boostr-audio/assets/g2p_lm/pack_model.py \\
        <checkpoint_dir> <vocab.json> <out_dir>
"""
import glob
import os
import re
import shutil
import sys
import zlib


def pick_checkpoint(checkpoint_dir):
    """Return the path of the checkpoint to pack, or exit with every
    candidate named when the directory holds several and none is preferred."""
    candidates = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not candidates:
        sys.exit(f"error: no *.safetensors checkpoint in {checkpoint_dir}")
    if len(candidates) == 1:
        return candidates[0]

    finals = [p for p in candidates if p[: -len(".safetensors")].endswith("_final")]
    if finals:
        return finals[0]

    by_step = []
    for p in candidates:
        stem = p[: -len(".safetensors")]
        m = re.search(r"_step_(\d+)$", stem)
        if m:
            by_step.append((int(m.group(1)), p))
    if len(by_step) == len(candidates):
        return max(by_step, key=lambda pair: pair[0])[1]

    sys.exit(
        f"error: {checkpoint_dir} holds {len(candidates)} safetensors files and none "
        f"is named *_final or *_step_<N>: {', '.join(candidates)}"
    )


def main():
    if len(sys.argv) != 4:
        print("usage: pack_model.py <checkpoint_dir> <vocab.json> <out_dir>", file=sys.stderr)
        return 1
    checkpoint_dir, vocab_path, out_dir = sys.argv[1:4]

    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.isfile(config_path):
        sys.exit(f"error: no config.json in {checkpoint_dir}")
    if not os.path.isfile(vocab_path):
        sys.exit(f"error: no such vocab.json: {vocab_path}")

    checkpoint_path = pick_checkpoint(checkpoint_dir)

    os.makedirs(out_dir, exist_ok=True)
    with open(checkpoint_path, "rb") as f:
        raw = f.read()
    compressed = zlib.compress(raw, level=9)
    with open(os.path.join(out_dir, "model.safetensors.zlib"), "wb") as f:
        f.write(compressed)
    shutil.copyfile(config_path, os.path.join(out_dir, "config.json"))
    shutil.copyfile(vocab_path, os.path.join(out_dir, "vocab.json"))

    print(
        f"packed {checkpoint_path} ({len(raw)} bytes -> {len(compressed)} bytes zlib) "
        f"into {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
