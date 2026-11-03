#!/usr/bin/env python3
"""Oracle fixtures for the SRVGGNetCompact parity test.

Builds a seeded tiny SRVGGNetCompact (num_feat 8, num_conv 2, upscale 4) on
CPU in float32 eval mode and dumps:

  <out>/config.json           network config and input shapes
  <out>/weights.safetensors   state dict under `body.*` keys
  <out>/weights.pth           the same state dict, saved as `{"params": ...}`
  <out>/io.safetensors        per input `a` and `b`: `input_<k>`,
                              `raw_<k>` (forward_raw), `resized_<k>` (forward)

The network is a self-contained copy of the video-remaster PoC
`CompactRestorer`, generalised over the config. That class is the
checkpoint-compatible SRVGG architecture from Real-ESRGAN
(`realesrgan/archs/srvgg_arch.py`).

Requires torch and safetensors. Usage:
  python3 sr_blocks_oracle.py <output_dir>
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional

SEED = 20261104
NUM_IN_CH = 3
NUM_OUT_CH = 3
NUM_FEAT = 8
NUM_CONV = 2
UPSCALE = 4
INPUTS = {"a": (1, 3, 13, 11), "b": (1, 3, 8, 8)}


class SrVggCompact(nn.Module):
    """SRVGGNetCompact with PReLU, the PoC `CompactRestorer` over a config."""

    def __init__(self, num_in_ch, num_out_ch, num_feat, num_conv, upscale):
        super().__init__()
        self.upscale = upscale
        layers = [nn.Conv2d(num_in_ch, num_feat, 3, padding=1), nn.PReLU(num_feat)]
        for _ in range(num_conv):
            layers.extend([nn.Conv2d(num_feat, num_feat, 3, padding=1), nn.PReLU(num_feat)])
        layers.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, padding=1))
        self.body = nn.Sequential(*layers)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward_raw(self, image):
        enlarged = self.upsampler(self.body(image))
        return enlarged + functional.interpolate(image, scale_factor=self.upscale, mode="nearest")

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


def build_model():
    model = SrVggCompact(NUM_IN_CH, NUM_OUT_CH, NUM_FEAT, NUM_CONV, UPSCALE).eval()
    with torch.no_grad():
        # The default PReLU slope is one shared 0.25. Distinct per-channel
        # slopes catch a channel-order or broadcast error.
        for layer in model.body:
            if isinstance(layer, nn.PReLU):
                layer.weight.uniform_(0.05, 0.45)
    return model


def make_input(shape, generator):
    # Values in [0, 1], the range of a decoded frame.
    return torch.rand(shape, generator=generator, dtype=torch.float32)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: sr_blocks_oracle.py <output_dir>")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    model = build_model()
    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(out / "weights.safetensors"))
    torch.save({"params": state}, str(out / "weights.pth"))

    generator = torch.Generator().manual_seed(SEED + 1)
    io = {}
    with torch.no_grad():
        for key, shape in INPUTS.items():
            image = make_input(shape, generator)
            io[f"input_{key}"] = image
            io[f"raw_{key}"] = model.forward_raw(image).contiguous()
            io[f"resized_{key}"] = model(image).contiguous()
    save_file(io, str(out / "io.safetensors"))

    config = {
        "num_in_ch": NUM_IN_CH,
        "num_out_ch": NUM_OUT_CH,
        "num_feat": NUM_FEAT,
        "num_conv": NUM_CONV,
        "upscale": UPSCALE,
        "inputs": {k: list(v) for k, v in INPUTS.items()},
        "seed": SEED,
        "torch_version": torch.__version__,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for key in INPUTS:
        raw = io[f"raw_{key}"]
        resized = io[f"resized_{key}"]
        print(
            f"{key}: raw {tuple(raw.shape)} [{raw.min():.4f}, {raw.max():.4f}], "
            f"resized {tuple(resized.shape)} [{resized.min():.4f}, {resized.max():.4f}]"
        )

    # Half-precision calibration: torch's own CUDA F16 / BF16 error against
    # the F32 golden. Printed only; the Rust test states its own bound.
    if torch.cuda.is_available():
        for dtype in (torch.float16, torch.bfloat16):
            half = build_model_like(model).to("cuda", dtype)
            worst_raw = 0.0
            worst_resized = 0.0
            with torch.no_grad():
                for key in INPUTS:
                    image = io[f"input_{key}"].to("cuda", dtype)
                    raw = half.forward_raw(image).float().cpu()
                    resized = half(image).float().cpu()
                    worst_raw = max(worst_raw, (raw - io[f"raw_{key}"]).abs().max().item())
                    worst_resized = max(
                        worst_resized, (resized - io[f"resized_{key}"]).abs().max().item()
                    )
            print(f"torch cuda {dtype}: max abs raw {worst_raw:.3e}, resized {worst_resized:.3e}")


def build_model_like(model):
    copy = SrVggCompact(NUM_IN_CH, NUM_OUT_CH, NUM_FEAT, NUM_CONV, UPSCALE).eval()
    copy.load_state_dict(model.state_dict())
    return copy


if __name__ == "__main__":
    main()
