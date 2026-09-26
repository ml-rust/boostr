#!/usr/bin/env python3
"""Oracle fixtures for VITS building-block parity tests.

Builds HF `transformers.models.vits.modeling_vits.{VitsAttention,
VitsDilatedDepthSeparableConv, VitsWaveNet}` with a small config, under a
fixed seed, in eval mode (dropout off), and dumps:

  <out>/config.json           shared numeric config for all three modules
  <out>/attention/weights.safetensors   HF state_dict key names
  <out>/attention/io.safetensors        input, attention_mask, output, output_masked
  <out>/dds_conv/weights.safetensors
  <out>/dds_conv/io.safetensors         input, padding_mask, output, output_masked
  <out>/wavenet/weights.safetensors     speaker_embedding_size=0 (no cond_layer)
  <out>/wavenet/io.safetensors          input, padding_mask, output, output_masked
  <out>/wavenet_cond/weights.safetensors  speaker_embedding_size=4 (has cond_layer)
  <out>/wavenet_cond/io.safetensors       input, padding_mask, global_conditioning, output

Requires: torch, transformers, safetensors. Usage:
  python3 vits_blocks_oracle.py <output_dir>
"""
import json
import sys

import torch
from safetensors.torch import save_file
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.models.vits.modeling_vits import (
    VitsAttention,
    VitsConfig,
    VitsDilatedDepthSeparableConv,
    VitsWaveNet,
)

HIDDEN = 8
HEADS = 2
WINDOW = 2
KERNEL = 3
DDS_LAYERS = 3
WAVENET_LAYERS = 3
DILATION_RATE = 2
SPEAKER_DIM = 4
BATCH = 2
SEQ = 5


def base_config(**overrides):
    kwargs = dict(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        window_size=WINDOW,
        use_bias=True,
        attention_dropout=0.0,
        duration_predictor_kernel_size=KERNEL,
        depth_separable_num_layers=DDS_LAYERS,
        wavenet_kernel_size=KERNEL,
        wavenet_dilation_rate=DILATION_RATE,
        wavenet_dropout=0.0,
        speaker_embedding_size=0,
    )
    kwargs.update(overrides)
    return VitsConfig(**kwargs)


def padding_mask_2d():
    """[BATCH, SEQ] boolean-as-float mask: batch 1's last two positions are padding."""
    m = torch.ones(BATCH, SEQ)
    m[1, SEQ - 2 :] = 0.0
    return m


def dump(dir_path, weights: dict, io: dict):
    dir_path.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in weights.items()}, dir_path / "weights.safetensors")
    save_file({k: v.contiguous() for k, v in io.items()}, dir_path / "io.safetensors")


def run_attention(out_dir):
    torch.manual_seed(0)
    cfg = base_config()
    attn = VitsAttention(cfg)
    attn.eval()

    hidden_states = torch.randn(BATCH, SEQ, HIDDEN)
    mask_2d = padding_mask_2d()
    attention_mask = _prepare_4d_attention_mask(mask_2d, hidden_states.dtype)

    with torch.no_grad():
        output, _ = attn(hidden_states, attention_mask=None)
        output_masked, _ = attn(hidden_states, attention_mask=attention_mask)

    dump(
        out_dir / "attention",
        dict(attn.state_dict()),
        dict(
            input=hidden_states,
            attention_mask=attention_mask,
            output=output,
            output_masked=output_masked,
        ),
    )


def run_dds_conv(out_dir):
    torch.manual_seed(1)
    cfg = base_config()
    dds = VitsDilatedDepthSeparableConv(cfg, dropout_rate=0.0)
    dds.eval()

    inputs = torch.randn(BATCH, HIDDEN, SEQ)
    full_mask = torch.ones(BATCH, 1, SEQ)
    real_mask = padding_mask_2d().unsqueeze(1)

    with torch.no_grad():
        output = dds(inputs, full_mask)
        output_masked = dds(inputs, real_mask)

    dump(
        out_dir / "dds_conv",
        dict(dds.state_dict()),
        dict(
            input=inputs,
            padding_mask=real_mask,
            output=output,
            output_masked=output_masked,
        ),
    )


def run_wavenet(out_dir):
    torch.manual_seed(2)
    cfg = base_config(speaker_embedding_size=0)
    wn = VitsWaveNet(cfg, num_layers=WAVENET_LAYERS)
    wn.eval()

    inputs = torch.randn(BATCH, HIDDEN, SEQ)
    full_mask = torch.ones(BATCH, 1, SEQ)
    real_mask = padding_mask_2d().unsqueeze(1)

    with torch.no_grad():
        output = wn(inputs, full_mask, global_conditioning=None)
        output_masked = wn(inputs, real_mask, global_conditioning=None)

    dump(
        out_dir / "wavenet",
        dict(wn.state_dict()),
        dict(
            input=inputs,
            padding_mask=real_mask,
            output=output,
            output_masked=output_masked,
        ),
    )


def run_wavenet_cond(out_dir):
    torch.manual_seed(3)
    cfg = base_config(speaker_embedding_size=SPEAKER_DIM)
    wn = VitsWaveNet(cfg, num_layers=WAVENET_LAYERS)
    wn.eval()

    inputs = torch.randn(BATCH, HIDDEN, SEQ)
    full_mask = torch.ones(BATCH, 1, SEQ)
    global_conditioning = torch.randn(BATCH, SPEAKER_DIM, 1)

    with torch.no_grad():
        output = wn(inputs, full_mask, global_conditioning=global_conditioning)

    dump(
        out_dir / "wavenet_cond",
        dict(wn.state_dict()),
        dict(
            input=inputs,
            padding_mask=full_mask,
            global_conditioning=global_conditioning,
            output=output,
        ),
    )


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <output_dir>", file=sys.stderr)
        sys.exit(1)
    from pathlib import Path

    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    config = dict(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        window_size=WINDOW,
        kernel_size=KERNEL,
        dds_layers=DDS_LAYERS,
        wavenet_layers=WAVENET_LAYERS,
        dilation_rate=DILATION_RATE,
        speaker_dim=SPEAKER_DIM,
        batch=BATCH,
        seq=SEQ,
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    run_attention(out_dir)
    run_dds_conv(out_dir)
    run_wavenet(out_dir)
    run_wavenet_cond(out_dir)
    print(f"wrote fixtures to {out_dir}")


if __name__ == "__main__":
    main()
