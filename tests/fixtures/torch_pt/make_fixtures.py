#!/usr/bin/env python3
"""Tiny `.pth` fixtures for the `torch_pt` reader tests in `tests/torch_pt_source.rs`.

Each file is a `torch.save` in the default zip format. Values are small
integers or exact binary fractions, so F16 and BF16 hold them without
rounding and the Rust test compares them exactly.

Files:
- `wrap_params_ema.pth`: `params_ema` and `params` both present, plus `metadata`.
- `wrap_params.pth`: `params` plus `metadata`.
- `wrap_state_dict.pth`: `state_dict` plus `metadata`.
- `wrap_root.pth`: tensors at the root, no wrapper.
- `strides.pth`: `bad` is `x.t()` with transposed strides. `ok_size1` has a
  size-1 dim whose stride is not the row-major one.
- `offset.pth`: `big[3:7]` saved alone, so the file holds all of `big` and
  the tensor starts at storage offset 3.
- `half.pth`: an F16 and a BF16 tensor.

Requires: torch.

Usage: python3 tests/fixtures/torch_pt/make_fixtures.py
"""
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent


def layer(scale):
    return {
        "conv.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3) * scale,
        "conv.bias": torch.tensor([1.0, -1.0]) * scale,
    }


def save(name, obj):
    torch.save(obj, OUT / name)


def main():
    metadata = {"in_nc": 3, "name": "tiny"}
    save("wrap_params_ema.pth", {"params_ema": layer(1.0), "params": layer(2.0), "metadata": metadata})
    save("wrap_params.pth", {"params": layer(2.0), "metadata": metadata})
    save("wrap_state_dict.pth", {"state_dict": layer(3.0), "metadata": metadata})
    save("wrap_root.pth", layer(4.0))

    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    bad = x.t()
    assert bad.stride() == (1, 3) and not bad.is_contiguous()
    y = torch.arange(3, dtype=torch.float32).reshape(3, 1)
    ok_size1 = y.t()
    assert ok_size1.shape == (1, 3) and ok_size1.stride() == (1, 1)
    save("strides.pth", {"bad": bad, "ok_size1": ok_size1})

    big = torch.arange(10, dtype=torch.float32) * 0.5
    view = big[3:7]
    assert view.storage_offset() == 3
    save("offset.pth", {"view": view})

    values = torch.tensor([[1.0, -2.5, 0.125], [3.0, 0.0, -0.75]])
    save("half.pth", {"f16": values.to(torch.float16), "bf16": values.to(torch.bfloat16)})


if __name__ == "__main__":
    main()
