#!/usr/bin/env python3
"""Whole-model reference dump for VITS (MMS-TTS) parity tests.

Loads HuggingFace's `VitsModel` + `VitsTokenizer` from a real checkpoint
directory, runs each of a few Malay sentences through the model in eval mode
(CPU, float32), and records every intermediate later VITS units need: the
text encoder's outputs, the duration predictor's noise draw and predicted
durations, the monotonic alignment path, the expanded prior distribution, the
prior-latent noise draw, the flow output, and the final waveform. Every
Gaussian noise draw (`torch.randn` / `torch.randn_like`) is captured by
monkeypatching those two functions inside `torch`, so the Rust side replays
the exact same noise instead of needing a matching RNG.

Usage:
  python3 vits_reference_dump.py <output_dir>

Reads the checkpoint directory from `MMS_TTS_ZLM_DIR`, else
`$BOOSTR_MODELS_DIR/facebook/mms-tts-zlm`. Requires torch, transformers,
safetensors.

Writes, per sentence `i` (0-indexed): `<output_dir>/sentence_<i>.safetensors`
plus a shared `<output_dir>/index.json` describing the sentences, the
checkpoint's config values every unit needs, and the torch/transformers
versions the dump was produced with. Not checked in: run this script once and
point `VITS_REF_DIR` (or `BOOSTR_MODELS_DIR`) at the output.
"""

import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

SENTENCES = [
    # Short: no punctuation, few tokens.
    "Selamat pagi.",
    # Medium: a question, mixed-case words, one piece of punctuation.
    "Bagaimana keadaan awak hari ini?",
    # Longer, with an internal comma plus a final full stop — exercises a
    # longer duration-predictor sequence and a wider alignment matrix.
    "Kerajaan Malaysia telah mengumumkan pelbagai inisiatif baharu, "
    "termasuk peningkatan infrastruktur di seluruh negara.",
]


def checkpoint_dir() -> Path:
    env = os.environ.get("MMS_TTS_ZLM_DIR")
    if env:
        return Path(env)
    root = os.environ.get("BOOSTR_MODELS_DIR")
    if not root:
        print(
            "error: set MMS_TTS_ZLM_DIR or BOOSTR_MODELS_DIR to locate the "
            "facebook/mms-tts-zlm checkpoint",
            file=sys.stderr,
        )
        sys.exit(1)
    return Path(root) / "facebook" / "mms-tts-zlm"


class NoiseCapture:
    """Monkeypatches `torch.randn`/`torch.randn_like` for the duration of a
    `with` block, recording every draw (in call order) as
    `{tag}_{index}` -> tensor, alongside the shape and originating function
    name. Restores the originals on exit even if the wrapped call raises.
    """

    def __init__(self, tag: str):
        self.tag = tag
        self.calls = []
        self._orig_randn = torch.randn
        self._orig_randn_like = torch.randn_like

    def __enter__(self):
        def traced_randn(*args, **kwargs):
            out = self._orig_randn(*args, **kwargs)
            self.calls.append(("randn", out.shape, out.clone()))
            return out

        def traced_randn_like(*args, **kwargs):
            out = self._orig_randn_like(*args, **kwargs)
            self.calls.append(("randn_like", out.shape, out.clone()))
            return out

        torch.randn = traced_randn
        torch.randn_like = traced_randn_like
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.randn = self._orig_randn
        torch.randn_like = self._orig_randn_like

    def named_tensors(self):
        out = {}
        for i, (fn_name, _shape, tensor) in enumerate(self.calls):
            out[f"{self.tag}_{fn_name}_{i}"] = tensor
        return out


def run_one(model, tokenizer, text: str, seed: int):
    """Mirrors `VitsModel.forward` step by step (same submodule calls, same
    order) so every intermediate is reachable — `VitsModel.forward` itself
    returns only the waveform, sequence length and spectrogram.
    """
    import numpy as np

    torch.manual_seed(seed)
    inputs = tokenizer(text=text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    tensors = {
        "input_ids": input_ids.to(torch.int64),
        "attention_mask": attention_mask.to(torch.float32),
    }

    with NoiseCapture("noise") as capture:
        with torch.no_grad():
            mask_dtype = model.text_encoder.embed_tokens.weight.dtype
            input_padding_mask = attention_mask.unsqueeze(-1).to(mask_dtype)

            text_encoder_output = model.text_encoder(
                input_ids=input_ids,
                padding_mask=input_padding_mask,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden_states = text_encoder_output.last_hidden_state
            tensors["text_encoder_last_hidden_state"] = hidden_states.clone()
            tensors["text_encoder_prior_means"] = text_encoder_output.prior_means.clone()
            tensors["text_encoder_prior_log_variances"] = (
                text_encoder_output.prior_log_variances.clone()
            )

            hidden_states = hidden_states.transpose(1, 2)
            input_padding_mask_t = input_padding_mask.transpose(1, 2)
            prior_means = text_encoder_output.prior_means
            prior_log_variances = text_encoder_output.prior_log_variances

            log_duration = model.duration_predictor(
                hidden_states,
                input_padding_mask_t,
                None,
                reverse=True,
                noise_scale=model.noise_scale_duration,
            )
            tensors["log_duration"] = log_duration.clone()

            length_scale = 1.0 / model.speaking_rate
            duration = torch.ceil(torch.exp(log_duration) * input_padding_mask_t * length_scale)
            tensors["duration"] = duration.clone()
            predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()
            tensors["predicted_lengths"] = predicted_lengths.clone().to(torch.int64)

            indices = torch.arange(
                predicted_lengths.max(), dtype=predicted_lengths.dtype, device=predicted_lengths.device
            )
            output_padding_mask = indices.unsqueeze(0) < predicted_lengths.unsqueeze(1)
            output_padding_mask = output_padding_mask.unsqueeze(1).to(input_padding_mask_t.dtype)
            tensors["output_padding_mask"] = output_padding_mask.clone()

            attn_mask = torch.unsqueeze(input_padding_mask_t, 2) * torch.unsqueeze(
                output_padding_mask, -1
            )
            batch_size, _, output_length, input_length = attn_mask.shape
            cum_duration = torch.cumsum(duration, -1).view(batch_size * input_length, 1)
            out_indices = torch.arange(output_length, dtype=duration.dtype, device=duration.device)
            valid_indices = out_indices.unsqueeze(0) < cum_duration
            valid_indices = valid_indices.to(attn_mask.dtype).view(
                batch_size, input_length, output_length
            )
            padded_indices = valid_indices - torch.nn.functional.pad(
                valid_indices, [0, 0, 1, 0, 0, 0]
            )[:, :-1]
            attn = padded_indices.unsqueeze(1).transpose(2, 3) * attn_mask
            tensors["attn"] = attn.clone()

            expanded_prior_means = torch.matmul(attn.squeeze(1), prior_means).transpose(1, 2)
            expanded_prior_log_variances = torch.matmul(attn.squeeze(1), prior_log_variances).transpose(
                1, 2
            )
            tensors["expanded_prior_means"] = expanded_prior_means.clone()
            tensors["expanded_prior_log_variances"] = expanded_prior_log_variances.clone()

            prior_latents = (
                expanded_prior_means
                + torch.randn_like(expanded_prior_means)
                * torch.exp(expanded_prior_log_variances)
                * model.noise_scale
            )
            tensors["prior_latents"] = prior_latents.clone()

            latents = model.flow(
                prior_latents, output_padding_mask, global_conditioning=None, reverse=True
            )
            tensors["flow_output"] = latents.clone()

            spectrogram = latents * output_padding_mask
            waveform = model.decoder(spectrogram, None)
            waveform = waveform.squeeze(1)
            tensors["waveform"] = waveform.clone()
            sequence_lengths = predicted_lengths * int(np.prod(model.config.upsample_rates))
            tensors["sequence_lengths"] = sequence_lengths.clone().to(torch.int64)

    tensors.update(capture.named_tensors())
    return {k: v.contiguous() for k, v in tensors.items()}


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <output_dir>", file=sys.stderr)
        sys.exit(1)
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    import transformers
    from transformers import VitsModel, VitsTokenizer

    ckpt = checkpoint_dir()
    if not ckpt.exists():
        print(f"error: checkpoint directory not found: {ckpt}", file=sys.stderr)
        sys.exit(1)

    tokenizer = VitsTokenizer.from_pretrained(str(ckpt))
    model = VitsModel.from_pretrained(str(ckpt), torch_dtype=torch.float32)
    model.eval()

    sentences_meta = []
    for i, text in enumerate(SENTENCES):
        tensors = run_one(model, tokenizer, text, seed=1000 + i)
        file_name = f"sentence_{i}.safetensors"
        save_file(tensors, out_dir / file_name)
        sentences_meta.append(
            {
                "index": i,
                "text": text,
                "file": file_name,
                "input_len": int(tensors["input_ids"].shape[1]),
                "waveform_len": int(tensors["waveform"].shape[1]),
            }
        )
        print(f"wrote {file_name}: {text!r}")

    index = {
        "checkpoint_dir": str(ckpt),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "config": {
            "noise_scale": model.noise_scale,
            "noise_scale_duration": model.noise_scale_duration,
            "speaking_rate": model.speaking_rate,
            "hidden_size": model.config.hidden_size,
            "flow_size": model.config.flow_size,
            "vocab_size": model.config.vocab_size,
            "num_hidden_layers": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
            "window_size": model.config.window_size,
            "sampling_rate": model.config.sampling_rate,
            "upsample_rates": list(model.config.upsample_rates),
        },
        "sentences": sentences_meta,
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"wrote index.json ({len(sentences_meta)} sentences) to {out_dir}")


if __name__ == "__main__":
    main()
