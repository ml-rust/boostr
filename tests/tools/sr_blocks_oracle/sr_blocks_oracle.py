#!/usr/bin/env python3
"""Oracle fixtures for the super-resolution parity tests.

`compact` builds a seeded tiny SRVGGNetCompact (num_feat 8, num_conv 2,
upscale 4) on CPU in float32 eval mode and dumps:

  <out>/config.json           network config and input shapes
  <out>/weights.safetensors   state dict under `body.*` keys
  <out>/weights.pth           the same state dict, saved as `{"params": ...}`
  <out>/io.safetensors        per input `a` and `b`: `input_<k>`,
                              `raw_<k>` (forward_raw), `resized_<k>` (forward)

The network is a self-contained copy of the video-remaster PoC
`CompactRestorer`, generalised over the config. That class is the
checkpoint-compatible SRVGG architecture from Real-ESRGAN
(`realesrgan/archs/srvgg_arch.py`).

`rrdb` builds a seeded tiny x2 RRDBNet (num_feat 8, num_grow_ch 4,
num_block 1) on CPU in float32 eval mode and dumps:

  <out>/config.json                      network config and input shapes
  <out>/weights.pth                      modern keys, as `{"params_ema": ...}`
  <out>/weights_legacy.pth               legacy ESRGAN `model.*` keys plus
                                         `metadata.*`, as PureVision ships
  <out>/weights_legacy_duplicate.pth     legacy keys plus a second key that
                                         maps to `body.0.rdb1.conv1.weight`
  <out>/weights_legacy_bad_metadata.pth  legacy keys with the metadata of an
                                         x4 network
  <out>/weights_unexpected.pth           modern keys plus one unknown key
  <out>/io.safetensors                   per input `a` and `b`: `input_<k>`,
                                         `out_<k>` (forward)

The RRDB network is the video-remaster PoC `RRDBNet`, generalised over the
config. That class is BasicSR's checkpoint-compatible RRDBNet
(`basicsr/archs/rrdbnet_arch.py`). The legacy remap is the PoC
`load_weights`.

Requires torch and safetensors. Usage:
  python3 sr_blocks_oracle.py compact <output_dir>
  python3 sr_blocks_oracle.py rrdb <output_dir>
"""

import json
import re
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

RRDB_SEED = 20261105
RRDB_NUM_FEAT = 8
RRDB_NUM_GROW_CH = 4
RRDB_NUM_BLOCK = 1
RRDB_SCALE = 2
RRDB_WEIGHT_GAIN = 1.7
# Legacy ESRGAN names of the convs outside the body, as the PoC maps them.
# `conv_body` is `model.1.sub.<num_block>`.
LEGACY_FIXED = {
    "conv_first": "model.0",
    "conv_up1": "model.3",
    "conv_up2": "model.6",
    "conv_hr": "model.8",
    "conv_last": "model.10",
}


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
    networks = {"compact": compact_main, "rrdb": rrdb_main}
    if len(sys.argv) != 3 or sys.argv[1] not in networks:
        sys.exit("usage: sr_blocks_oracle.py {compact|rrdb} <output_dir>")
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    networks[sys.argv[1]](out)


def compact_main(out):
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


class ResidualDenseBlock(nn.Module):
    """The PoC `ResidualDenseBlock` over `features` and `growth`."""

    def __init__(self, features, growth):
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
    """The PoC `ResidualInResidualBlock` over `features` and `growth`."""

    def __init__(self, features, growth):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(features, growth)
        self.rdb2 = ResidualDenseBlock(features, growth)
        self.rdb3 = ResidualDenseBlock(features, growth)

    def forward(self, image):
        return image + self.rdb3(self.rdb2(self.rdb1(image))) * 0.2


class RrdbNet(nn.Module):
    """The PoC x2 `RRDBNet` over the config."""

    def __init__(self, num_in_ch, num_out_ch, num_feat, num_block, num_grow_ch):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch * 4, num_feat, 3, padding=1)
        self.body = nn.Sequential(
            *(ResidualInResidualBlock(num_feat, num_grow_ch) for _ in range(num_block))
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, padding=1)

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


def build_rrdb():
    return RrdbNet(
        NUM_IN_CH, NUM_OUT_CH, RRDB_NUM_FEAT, RRDB_NUM_BLOCK, RRDB_NUM_GROW_CH
    ).eval()


def seeded_rrdb():
    """`build_rrdb` with its conv weights scaled by `RRDB_WEIGHT_GAIN`.

    The default init shrinks each conv's output, so the output of the tiny
    network sits near 0. The gain moves it to about [-1, 1].
    """
    model = build_rrdb()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.mul_(RRDB_WEIGHT_GAIN)
    return model


def legacy_name(key, num_block):
    """The legacy ESRGAN key of modern key `key`."""
    match = re.fullmatch(r"body\.(\d+)\.rdb([123])\.conv([12345])\.(weight|bias)", key)
    if match:
        block, dense, convolution, parameter = match.groups()
        return f"model.1.sub.{block}.RDB{dense}.conv{convolution}.0.{parameter}"
    prefix, _, parameter = key.rpartition(".")
    if prefix == "conv_body":
        return f"model.1.sub.{num_block}.{parameter}"
    return f"{LEGACY_FIXED[prefix]}.{parameter}"


def legacy_to_modern(state, num_block):
    """The PoC `load_weights` remap, with `model.1.sub.23` as
    `model.1.sub.<num_block>`. Unknown keys and duplicate targets raise."""
    names = {v: k for k, v in LEGACY_FIXED.items()}
    names[f"model.1.sub.{num_block}"] = "conv_body"
    converted = {}
    for key, value in state.items():
        match = re.fullmatch(
            r"model\.1\.sub\.(\d+)\.RDB([123])\.conv([12345])\.0\.(weight|bias)", key
        )
        if match:
            block, dense, convolution, parameter = match.groups()
            target = f"body.{block}.rdb{dense}.conv{convolution}.{parameter}"
        else:
            prefix, separator, parameter = key.rpartition(".")
            if prefix not in names or parameter not in ("weight", "bias"):
                raise RuntimeError(f"unsupported legacy key {key}")
            target = names[prefix] + separator + parameter
        if target in converted:
            raise RuntimeError(f"duplicate target {target}")
        converted[target] = value
    return converted


def rrdb_metadata(scale):
    """`metadata` as PureVision ships it, for an RRDBNet of `scale / 2`."""
    return {
        "in_nc": NUM_IN_CH * 4,
        "out_nc": NUM_OUT_CH,
        "num_filters": RRDB_NUM_FEAT,
        "num_blocks": RRDB_NUM_BLOCK,
        "scale": scale,
        "plus": False,
        "shuffle_factor": 2,
        "norm": None,
        "act": "leakyrelu",
        "upsampler": "upconv",
        "mode": "CNA",
    }


def rrdb_main(out):
    torch.manual_seed(RRDB_SEED)
    model = seeded_rrdb()
    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    legacy = {legacy_name(k, RRDB_NUM_BLOCK): v for k, v in state.items()}

    # The legacy dict must round-trip through the PoC remap to the same
    # state, so the legacy fixture holds the weights under test.
    back = legacy_to_modern(legacy, RRDB_NUM_BLOCK)
    if back.keys() != state.keys() or any(not torch.equal(back[k], state[k]) for k in state):
        sys.exit("legacy remap does not round-trip")

    torch.save({"params_ema": state}, str(out / "weights.pth"))
    torch.save(
        {"metadata": rrdb_metadata(2 * RRDB_SCALE), "params_ema": legacy},
        str(out / "weights_legacy.pth"),
    )
    duplicate = dict(legacy)
    duplicate["model.1.sub.00.RDB1.conv1.0.weight"] = legacy["model.1.sub.0.RDB1.conv1.0.weight"].clone()
    torch.save(
        {"metadata": rrdb_metadata(2 * RRDB_SCALE), "params_ema": duplicate},
        str(out / "weights_legacy_duplicate.pth"),
    )
    torch.save(
        {"metadata": rrdb_metadata(4 * RRDB_SCALE), "params_ema": legacy},
        str(out / "weights_legacy_bad_metadata.pth"),
    )
    unexpected = dict(state)
    unexpected["body.0.rdb1.conv6.weight"] = state["body.0.rdb1.conv5.bias"].clone()
    torch.save({"params_ema": unexpected}, str(out / "weights_unexpected.pth"))

    generator = torch.Generator().manual_seed(RRDB_SEED + 1)
    io = {}
    with torch.no_grad():
        for key, shape in INPUTS.items():
            image = make_input(shape, generator)
            io[f"input_{key}"] = image
            io[f"out_{key}"] = model(image).contiguous()
    save_file(io, str(out / "io.safetensors"))

    config = {
        "num_in_ch": NUM_IN_CH,
        "num_out_ch": NUM_OUT_CH,
        "num_feat": RRDB_NUM_FEAT,
        "num_block": RRDB_NUM_BLOCK,
        "num_grow_ch": RRDB_NUM_GROW_CH,
        "scale": RRDB_SCALE,
        "inputs": {k: list(v) for k, v in INPUTS.items()},
        "seed": RRDB_SEED,
        "torch_version": torch.__version__,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for key in INPUTS:
        result = io[f"out_{key}"]
        print(f"{key}: out {tuple(result.shape)} [{result.min():.4f}, {result.max():.4f}]")

    # Half-precision calibration: torch's own CUDA F16 / BF16 error against
    # the F32 golden. Printed only; the Rust test states its own bound.
    if torch.cuda.is_available():
        for dtype in (torch.float16, torch.bfloat16):
            half = build_rrdb()
            half.load_state_dict(state)
            half = half.to("cuda", dtype)
            worst = 0.0
            with torch.no_grad():
                for key in INPUTS:
                    image = io[f"input_{key}"].to("cuda", dtype)
                    result = half(image).float().cpu()
                    worst = max(worst, (result - io[f"out_{key}"]).abs().max().item())
            print(f"torch cuda {dtype}: max abs out {worst:.3e}")


if __name__ == "__main__":
    main()
