#!/usr/bin/env python3
"""Real-checkpoint reference dumps for the super-resolution parity tests.

Loads one real checkpoint into its video-remaster PoC architecture, runs one
deterministic synthetic 3x67x45 frame through it in torch CPU float32 eval
mode, and writes:

  <out>/io.safetensors   `input` [1,3,67,45] and the network's outputs
  <out>/index.json       torch version, checkpoint path and sha256, shapes,
                         ranges, and torch's own error against the CPU F32
                         golden: CPU F64, plus CUDA F32 with TF32 off, CUDA
                         F16 and CUDA BF16 when a GPU is present

Networks:

  compact            `realesr-general-x4v3.pth` in the PoC `CompactRestorer`.
                     Outputs `raw` [1,3,268,180] (forward_raw) and `resized`
                     [1,3,134,90] (forward). Checkpoint: `REALESR_GENERAL_X4V3`,
                     else `$BOOSTR_MODELS_DIR/realesr-general-x4v3/realesr-general-x4v3.pth`.
  realesrgan_x2plus  `RealESRGAN_x2plus.pth` in the PoC `RRDBNet`. Output
                     `out` [1,3,134,90]. Checkpoint: `REALESRGAN_X2PLUS_PTH`,
                     else `$BOOSTR_MODELS_DIR/RealESRGAN_x2plus/RealESRGAN_x2plus.pth`.
  purevision         `2x_PureVision.pth` in the PoC `RRDBNet`, through the
                     PoC legacy ESRGAN key remap. Output `out` [1,3,134,90].
                     Checkpoint: `PUREVISION_2X_PTH`, else
                     `$BOOSTR_MODELS_DIR/2x_PureVision/2x_PureVision.pth`.

The output directory defaults to `$BOOSTR_MODELS_DIR/sr_reference/<network>`.
Nothing here is checked in.

Requires torch and safetensors. Usage:
  python3 sr_reference_dump.py <network> [<output_dir>]
"""

import hashlib
import json
import math
import os
import re
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


class ResidualDenseBlock(nn.Module):
    """Copy of the video-remaster PoC `ResidualDenseBlock`."""

    def __init__(self, features=64, growth=32):
        super().__init__()
        for index in range(1, 6):
            output = features if index == 5 else growth
            setattr(
                self,
                f"conv{index}",
                nn.Conv2d(features + (index - 1) * growth, output, 3, padding=1),
            )

    def forward(self, image):
        features = [image]
        for index in range(1, 5):
            value = getattr(self, f"conv{index}")(torch.cat(features, dim=1))
            features.append(functional.leaky_relu(value, negative_slope=0.2))
        return image + self.conv5(torch.cat(features, dim=1)) * 0.2


class ResidualInResidualBlock(nn.Module):
    """Copy of the video-remaster PoC `ResidualInResidualBlock`."""

    def __init__(self):
        super().__init__()
        self.rdb1 = ResidualDenseBlock()
        self.rdb2 = ResidualDenseBlock()
        self.rdb3 = ResidualDenseBlock()

    def forward(self, image):
        return image + self.rdb3(self.rdb2(self.rdb1(image))) * 0.2


class RRDBNet(nn.Module):
    """Copy of the video-remaster PoC `RRDBNet`, the x2 network."""

    def __init__(self):
        super().__init__()
        self.conv_first = nn.Conv2d(12, 64, 3, padding=1)
        self.body = nn.Sequential(*(ResidualInResidualBlock() for _ in range(23)))
        self.conv_body = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_up1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_up2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_hr = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_last = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, image):
        height, width = image.shape[-2:]
        if height % 2 or width % 2:
            image = functional.pad(image, (0, width % 2, 0, height % 2), mode="replicate")
        features = self.conv_first(functional.pixel_unshuffle(image, 2))
        features = features + self.conv_body(self.body(features))
        for convolution in (self.conv_up1, self.conv_up2):
            features = convolution(functional.interpolate(features, scale_factor=2, mode="nearest"))
            features = functional.leaky_relu(features, negative_slope=0.2)
        features = functional.leaky_relu(self.conv_hr(features), negative_slope=0.2)
        return self.conv_last(features)[..., : height * 2, : width * 2]


def legacy_esrgan_state(state, path):
    """Copy of the video-remaster PoC `load_weights` legacy remap."""
    if "model.0.weight" not in state:
        return state
    names = {
        "model.0": "conv_first",
        "model.1.sub.23": "conv_body",
        "model.3": "conv_up1",
        "model.6": "conv_up2",
        "model.8": "conv_hr",
        "model.10": "conv_last",
    }
    converted = {}
    for key, value in state.items():
        match = re.fullmatch(
            r"model\.1\.sub\.(\d+)\.RDB([123])\.conv([12345])\.0\.(weight|bias)",
            key,
        )
        if match:
            block, dense, convolution, parameter = match.groups()
            target = f"body.{block}.rdb{dense}.conv{convolution}.{parameter}"
        else:
            prefix, separator, parameter = key.rpartition(".")
            if prefix not in names or parameter not in ("weight", "bias"):
                raise RuntimeError(f"Unsupported checkpoint parameter {key} in {path}.")
            target = names[prefix] + separator + parameter
        if target in converted:
            raise RuntimeError(f"Duplicate checkpoint parameter {target} in {path}.")
        converted[target] = value
    return converted


def compact_outputs(model, image):
    return {"raw": model.forward_raw(image), "resized": model(image)}


def rrdb_outputs(model, image):
    return {"out": model(image)}


# name -> (checkpoint env var, path under BOOSTR_MODELS_DIR, model class,
# state remap, outputs).
NETWORKS = {
    "compact": (
        "REALESR_GENERAL_X4V3",
        "realesr-general-x4v3/realesr-general-x4v3.pth",
        CompactRestorer,
        lambda state, path: state,
        compact_outputs,
    ),
    "realesrgan_x2plus": (
        "REALESRGAN_X2PLUS_PTH",
        "RealESRGAN_x2plus/RealESRGAN_x2plus.pth",
        RRDBNet,
        legacy_esrgan_state,
        rrdb_outputs,
    ),
    "purevision": (
        "PUREVISION_2X_PTH",
        "2x_PureVision/2x_PureVision.pth",
        RRDBNet,
        legacy_esrgan_state,
        rrdb_outputs,
    ),
}


def models_dir() -> Path:
    root = os.environ.get("BOOSTR_MODELS_DIR")
    if not root:
        sys.exit("set BOOSTR_MODELS_DIR, or pass the output directory and the checkpoint variable")
    return Path(root)


def checkpoint_path(var, relative) -> Path:
    env = os.environ.get(var)
    if env:
        return Path(env)
    return models_dir() / relative


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


def error_report(ours, golden):
    """Max abs, PSNR, u8-exact fraction and max u8 diff of each output."""
    report = {}
    for name, gold in golden.items():
        mine = ours[name].double()
        gold = gold.double()
        mse = (mine - gold).pow(2).mean().item()
        u8_diff = (quantise_u8(mine) - quantise_u8(gold)).abs()
        report[name] = {
            "max_abs": (mine - gold).abs().max().item(),
            "psnr_db": 10.0 * math.log10(1.0 / mse) if mse > 0 else float("inf"),
            "u8_exact_fraction": (u8_diff == 0).double().mean().item(),
            "u8_max_diff": u8_diff.max().item(),
        }
    return report


def rerun(model_class, state, outputs, frame, device, dtype):
    """The network on `device` in `dtype`, outputs as F32 on CPU."""
    model = model_class().eval()
    model.load_state_dict(state, strict=True)
    model = model.to(device, dtype)
    with torch.no_grad():
        result = outputs(model, frame.to(device, dtype))
    return {k: v.float().cpu() for k, v in result.items()}


def calibration(model_class, state, outputs, frame, golden):
    """Torch's own error against the CPU F32 golden on other precisions."""
    report = {
        "f64_cpu": error_report(
            rerun(model_class, state, outputs, frame, "cpu", torch.float64), golden
        )
    }
    if not torch.cuda.is_available():
        return report
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        report["f32_cuda_no_tf32"] = error_report(
            rerun(model_class, state, outputs, frame, "cuda", torch.float32), golden
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    for name, dtype in (("f16_cuda", torch.float16), ("bf16_cuda", torch.bfloat16)):
        report[name] = error_report(
            rerun(model_class, state, outputs, frame, "cuda", dtype), golden
        )
    return report


def main():
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in NETWORKS:
        sys.exit(f"usage: sr_reference_dump.py {{{'|'.join(NETWORKS)}}} [<output_dir>]")
    network = sys.argv[1]
    var, relative, model_class, remap, outputs = NETWORKS[network]
    out = Path(sys.argv[2]) if len(sys.argv) == 3 else models_dir() / "sr_reference" / network
    ckpt = checkpoint_path(var, relative)
    if not ckpt.is_file():
        sys.exit(f"checkpoint {ckpt} not found; set {var} or BOOSTR_MODELS_DIR")

    checkpoint = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    state = remap(checkpoint.get("params_ema", checkpoint.get("params", checkpoint)), ckpt)
    model = model_class().eval()
    model.load_state_dict(state, strict=True)

    frame = synthetic_frame()
    with torch.no_grad():
        golden = {k: v.contiguous() for k, v in outputs(model, frame).items()}

    out.mkdir(parents=True, exist_ok=True)
    save_file({"input": frame, **golden}, str(out / "io.safetensors"))

    index = {
        "network": network,
        "torch_version": torch.__version__,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256(ckpt),
        "seed": SEED,
        "shapes": {"input": list(frame.shape), **{k: list(v.shape) for k, v in golden.items()}},
        "ranges": {k: [v.min().item(), v.max().item()] for k, v in golden.items()},
        "torch_vs_cpu_f32": calibration(model_class, state, outputs, frame, golden),
    }
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
