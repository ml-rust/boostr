#!/usr/bin/env python3
"""Real-checkpoint reference dump for the SRVGGNetCompact parity test.

Loads `realesr-general-x4v3.pth` into the video-remaster PoC architecture,
runs one deterministic synthetic 3x67x45 frame through it in torch CPU
float32 eval mode, and writes:

  <out>/io.safetensors   `input` [1,3,67,45], `raw` [1,3,268,180]
                         (forward_raw), `resized` [1,3,134,90] (forward)
  <out>/index.json       torch version, checkpoint path and sha256, shapes,
                         and torch's own CUDA F16 error against the F32
                         golden when a GPU is present

The checkpoint path comes from `REALESR_GENERAL_X4V3`, else
`$BOOSTR_MODELS_DIR/realesr-general-x4v3/realesr-general-x4v3.pth`. The
output directory defaults to `$BOOSTR_MODELS_DIR/sr_reference/compact`.
Nothing here is checked in.

Requires torch and safetensors. Usage:
  python3 sr_reference_dump.py [<output_dir>]
"""

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional

SEED = 4242
SHAPE = (1, 3, 67, 45)


class CompactRestorer(nn.Module):
    """Copy of the video-remaster PoC `CompactRestorer`."""

    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(3, 64, 3, padding=1), nn.PReLU(64)]
        for _ in range(32):
            layers.extend([nn.Conv2d(64, 64, 3, padding=1), nn.PReLU(64)])
        layers.append(nn.Conv2d(64, 48, 3, padding=1))
        self.body = nn.Sequential(*layers)
        self.upsampler = nn.PixelShuffle(4)

    def forward_raw(self, image):
        enlarged = self.upsampler(self.body(image))
        return enlarged + functional.interpolate(image, scale_factor=4, mode="nearest")

    def forward(self, image):
        height, width = image.shape[-2:]
        enlarged = self.forward_raw(image)
        return functional.interpolate(
            enlarged.float(),
            size=(height * 2, width * 2),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).to(enlarged.dtype)


def models_dir() -> Path:
    root = os.environ.get("BOOSTR_MODELS_DIR")
    if not root:
        sys.exit("set BOOSTR_MODELS_DIR, or pass the output directory and REALESR_GENERAL_X4V3")
    return Path(root)


def checkpoint_path() -> Path:
    env = os.environ.get("REALESR_GENERAL_X4V3")
    if env:
        return Path(env)
    return models_dir() / "realesr-general-x4v3" / "realesr-general-x4v3.pth"


def synthetic_frame() -> torch.Tensor:
    """Seeded noise over per-channel gradients, clamped to [0, 1]."""
    _, channels, height, width = SHAPE
    ys = torch.linspace(0.0, 1.0, height).view(height, 1)
    xs = torch.linspace(0.0, 1.0, width).view(1, width)
    gradients = torch.stack(
        [
            0.15 + 0.7 * xs.expand(height, width),
            0.15 + 0.7 * ys.expand(height, width),
            0.5 + 0.35 * torch.sin(6.0 * (xs + ys)),
        ]
    )
    generator = torch.Generator().manual_seed(SEED)
    noise = 0.12 * torch.randn((channels, height, width), generator=generator)
    return (gradients + noise).clamp(0.0, 1.0).unsqueeze(0).contiguous()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantise_u8(x: torch.Tensor) -> torch.Tensor:
    # torch.round rounds half to even.
    return torch.round(x.clamp(0.0, 1.0) * 255.0)


def half_error(model, frame, golden_raw, golden_resized, dtype):
    half = CompactRestorer().eval()
    half.load_state_dict(model.state_dict())
    half = half.to("cuda", dtype)
    with torch.no_grad():
        image = frame.to("cuda", dtype)
        raw = half.forward_raw(image).float().cpu()
        resized = half(image).float().cpu()
    report = {}
    for name, ours, gold in (("raw", raw, golden_raw), ("resized", resized, golden_resized)):
        mse = (ours - gold).pow(2).mean().item()
        u8_diff = (quantise_u8(ours) - quantise_u8(gold)).abs()
        report[name] = {
            "max_abs": (ours - gold).abs().max().item(),
            "psnr_db": 10.0 * math.log10(1.0 / mse) if mse > 0 else float("inf"),
            "u8_exact_fraction": (u8_diff == 0).float().mean().item(),
            "u8_max_diff": u8_diff.max().item(),
        }
    return report


def main():
    if len(sys.argv) > 2:
        sys.exit("usage: sr_reference_dump.py [<output_dir>]")
    out = Path(sys.argv[1]) if len(sys.argv) == 2 else models_dir() / "sr_reference" / "compact"
    ckpt = checkpoint_path()
    if not ckpt.is_file():
        sys.exit(f"checkpoint {ckpt} not found; set REALESR_GENERAL_X4V3 or BOOSTR_MODELS_DIR")

    checkpoint = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    model = CompactRestorer().eval()
    model.load_state_dict(state, strict=True)

    frame = synthetic_frame()
    with torch.no_grad():
        raw = model.forward_raw(frame).contiguous()
        resized = model(frame).contiguous()

    out.mkdir(parents=True, exist_ok=True)
    save_file({"input": frame, "raw": raw, "resized": resized}, str(out / "io.safetensors"))

    index = {
        "torch_version": torch.__version__,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256(ckpt),
        "seed": SEED,
        "shapes": {
            "input": list(frame.shape),
            "raw": list(raw.shape),
            "resized": list(resized.shape),
        },
        "ranges": {
            "raw": [raw.min().item(), raw.max().item()],
            "resized": [resized.min().item(), resized.max().item()],
        },
    }
    if torch.cuda.is_available():
        index["torch_cuda_half_vs_cpu_f32"] = {
            "f16": half_error(model, frame, raw, resized, torch.float16),
            "bf16": half_error(model, frame, raw, resized, torch.bfloat16),
        }
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
