#!/usr/bin/env python3
"""Build the tagger weights for `boostr_audio::g2p::spacy_tagger`, read at
run time by `EnglishTagger::from_dir`.

Loads the `tok2vec` and `tagger` components of an installed spaCy pipeline
`en_core_web_sm` 3.8.0 (spaCy 3.8.x, MIT, https://github.com/explosion/spaCy)
from its directory, reads the thinc parameters the pipeline deserializes from
`tok2vec/model` and `tagger/model`, and writes two files, into an output
directory (normally `$BOOSTR_MODELS_DIR/spacy-en-tagger`; not committed to
git):

  tagger.safetensors   plain (uncompressed) safetensors file
  tagger.json          network geometry, hash seeds, labels, symbols

Tensors (all F32, row-major, thinc's own layouts):
    embed.{i}.E            [rows_i, width]  HashEmbed table of attribute i
    embed.maxout.W         [width, pieces, width * n_attrs]
    embed.maxout.b         [width, pieces]
    embed.norm.G           [width]           LayerNorm gain
    embed.norm.b           [width]           LayerNorm bias
    encode.{d}.maxout.W    [width, pieces, width * (2 * window + 1)]
    encode.{d}.maxout.b    [width, pieces]
    encode.{d}.norm.G      [width]
    encode.{d}.norm.b      [width]
    softmax.W              [n_labels, width]
    softmax.b              [n_labels]

tagger.json:
    {
      "source":  str,             # spaCy and pipeline versions
      "attrs":   [str, ...],      # FeatureExtractor columns, table order
      "config":  {
        "width": int, "depth": int, "window_size": int,
        "embed_pieces": int, "encode_pieces": int,
        "seeds": [int, ...],      # HashEmbed seed per table
        "pad": int,               # with_array pad around the encoder input
        "layer_norm_eps": float   # thinc LayerNorm variance epsilon
      },
      "labels":  [str, ...],      # tagger labels, class index order
      "symbols": [[str, int], ...]  # spacy.symbols.IDS, sorted by id
    }

`symbols` is the table spaCy's `StringStore` consults before hashing: a
string listed there gets its symbol ID instead of `hash_string`.

The script asserts the model tree it walks: the layer names, the attribute
columns, the HashEmbed column order, dropout-only extra layers and the
tagger's unnormalized softmax. Any other architecture stops it.

The output is deterministic for a fixed spaCy and pipeline build.

Usage:
    python3 crates/boostr-audio/assets/spacy_en/convert_tagger.py \\
        <en_core_web_sm model dir> $BOOSTR_MODELS_DIR/spacy-en-tagger
The model dir is the one holding `config.cfg`, `tok2vec/` and `tagger/`
(`python -c "import en_core_web_sm; print(en_core_web_sm.load().path)"`).
Environment:
    SPACY_TAGGER_MODEL  model dir (overrides argv[1])
    SPACY_TAGGER_OUT    output dir (overrides argv[2])
"""
import json
import os
import struct
import sys

import numpy as np
import spacy
from spacy.symbols import IDS

# thinc `layers/layernorm.py` `_get_moments`: `X.var(axis=1) + 1e-08`.
LAYER_NORM_EPS = 1e-08


def fail(msg):
    sys.exit(f"convert_tagger.py: {msg}")


def expect(cond, msg):
    if not cond:
        fail(msg)


def param(layer, name):
    arr = layer.get_param(name)
    return np.ascontiguousarray(arr, dtype=np.float32)


def maxout_norm(chain_layer):
    """`maxout>>layernorm>>dropout` -> (maxout, layernorm)."""
    names = [layer.name for layer in chain_layer.walk()]
    expect(
        sorted(n for n in names if n in ("maxout", "layernorm", "dropout"))
        == ["dropout", "layernorm", "maxout"],
        f"expected maxout>>layernorm>>dropout, found {chain_layer.name}",
    )
    found = {layer.name: layer for layer in chain_layer.walk()}
    return found["maxout"], found["layernorm"]


def convert(model_dir):
    nlp = spacy.load(model_dir, enable=["tok2vec", "tagger"])
    source = f"spaCy {spacy.__version__}, {nlp.meta['lang']}_{nlp.meta['name']} {nlp.meta['version']}"
    t2v = nlp.get_pipe("tok2vec").model
    embed = t2v.get_ref("embed")
    encode = t2v.get_ref("encode")

    extractor = [n for n in embed.walk() if n.name == "extract_features"]
    expect(len(extractor) == 1, "no extract_features layer")
    attrs = list(extractor[0].attrs["columns"])

    tables = [n for n in embed.walk() if n.name == "hashembed"]
    expect(len(tables) == len(attrs), "one HashEmbed per attribute expected")
    for i, table in enumerate(tables):
        expect(table.attrs["column"] == i, f"HashEmbed {i} reads column {table.attrs['column']}")
    seeds = [int(t.attrs["seed"]) for t in tables]
    expect(all(0 <= s < 2**32 for s in seeds), "HashEmbed seeds must fit in u32")

    embed_mix = [n for n in embed.walk() if n.name == "maxout>>layernorm>>dropout"]
    expect(len(embed_mix) == 1, "no embedding Maxout mixing layer")
    embed_maxout, embed_norm = maxout_norm(embed_mix[0])

    encode_pad = encode.attrs.get("pad")
    expect(isinstance(encode_pad, int), "encoder with_array carries no pad")
    block_name = "residual(expand_window>>maxout>>layernorm>>dropout)"
    blocks = [n for n in encode.walk() if n.name == block_name]
    windows = [n for n in encode.walk() if n.name == "expand_window"]
    expect(len(blocks) == len(windows) and blocks, "residual(expand_window>>...) blocks expected")
    window_size = windows[0].attrs["window_size"]
    expect(all(w.attrs["window_size"] == window_size for w in windows), "mixed window sizes")

    tagger = nlp.get_pipe("tagger")
    softmax = tagger.model.get_ref("softmax")
    expect(softmax.name == "softmax", f"unexpected output layer {softmax.name}")
    expect(not softmax.attrs["softmax_normalize"], "tagger softmax normalizes at inference")
    expect(softmax.attrs["softmax_temperature"] == 1.0, "tagger softmax temperature != 1")
    expect(not tagger.cfg.get("overwrite", False), "tagger overwrite is set")

    tensors = {}
    for i, table in enumerate(tables):
        tensors[f"embed.{i}.E"] = param(table, "E")
    tensors["embed.maxout.W"] = param(embed_maxout, "W")
    tensors["embed.maxout.b"] = param(embed_maxout, "b")
    tensors["embed.norm.G"] = param(embed_norm, "G")
    tensors["embed.norm.b"] = param(embed_norm, "b")
    for d, block in enumerate(blocks):
        chain = [n for n in block.walk() if n.name == "maxout>>layernorm>>dropout"]
        expect(len(chain) == 1, f"encoder block {d} has no maxout>>layernorm>>dropout")
        maxout, norm = maxout_norm(chain[0])
        tensors[f"encode.{d}.maxout.W"] = param(maxout, "W")
        tensors[f"encode.{d}.maxout.b"] = param(maxout, "b")
        tensors[f"encode.{d}.norm.G"] = param(norm, "G")
        tensors[f"encode.{d}.norm.b"] = param(norm, "b")
    tensors["softmax.W"] = param(softmax, "W")
    tensors["softmax.b"] = param(softmax, "b")

    width = tensors["embed.norm.G"].shape[0]
    embed_pieces = tensors["embed.maxout.W"].shape[1]
    encode_pieces = tensors["encode.0.maxout.W"].shape[1]
    expect(tensors["embed.maxout.W"].shape == (width, embed_pieces, width * len(attrs)),
           "embedding Maxout shape")
    for d in range(len(blocks)):
        expect(tensors[f"encode.{d}.maxout.W"].shape
               == (width, encode_pieces, width * (2 * window_size + 1)), f"encoder {d} shape")
    labels = list(tagger.labels)
    expect(tensors["softmax.W"].shape == (len(labels), width), "softmax shape")

    config = {
        "width": int(width),
        "depth": len(blocks),
        "window_size": int(window_size),
        "embed_pieces": int(embed_pieces),
        "encode_pieces": int(encode_pieces),
        "seeds": seeds,
        "pad": int(encode_pad),
        "layer_norm_eps": LAYER_NORM_EPS,
    }
    symbols = sorted(IDS.items(), key=lambda kv: (kv[1], kv[0]))
    meta = {"source": source, "attrs": attrs, "config": config, "labels": labels,
            "symbols": [[s, int(i)] for s, i in symbols]}
    return tensors, meta


def safetensors_bytes(tensors, source):
    header = {"__metadata__": {"source": source}}
    offset = 0
    names = sorted(tensors)
    for name in names:
        arr = tensors[name]
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + arr.nbytes]}
        offset += arr.nbytes
    text = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    text += b" " * (-len(text) % 8)
    body = b"".join(tensors[name].astype("<f4").tobytes() for name in names)
    return struct.pack("<Q", len(text)) + text + body


def main():
    model_dir = os.environ.get("SPACY_TAGGER_MODEL") or (sys.argv[1] if len(sys.argv) > 1 else None)
    out_dir = os.environ.get("SPACY_TAGGER_OUT") or (sys.argv[2] if len(sys.argv) > 2 else None)
    if model_dir is None or out_dir is None:
        fail("usage: convert_tagger.py <model dir> <output dir>")
    tensors, meta = convert(model_dir)
    raw = safetensors_bytes(tensors, meta["source"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "tagger.safetensors"), "wb") as f:
        f.write(raw)
    with open(os.path.join(out_dir, "tagger.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
        f.write("\n")
    n_params = sum(a.size for a in tensors.values())
    print(f"wrote {out_dir}: {len(tensors)} tensors, {n_params} parameters, "
          f"{len(raw)} bytes")


if __name__ == "__main__":
    main()
