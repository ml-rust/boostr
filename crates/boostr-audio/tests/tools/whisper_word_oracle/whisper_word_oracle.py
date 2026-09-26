#!/usr/bin/env python3
"""Oracle generator for `crates/boostr-audio/tests/whisper_word_oracle.rs`.

Produces `whisper_word_oracle/{model}__{clip}.json` for `model` in
`whisper-tiny`, `whisper-base`, `whisper-small` and `clip` in `en_speech`,
`ms_codeswitch` (6 files total). Each file carries: `model`, `clip`,
`sample_rate`, `language`, `alignment_heads`, `median_filter_width`,
`time_precision` (+ `time_precision_derivation`), `num_frames` (already
halved: raw mel frames / 2), `prompt_ids`, `generated_ids` (HF's raw
`generate()` output, including the trailing EOS), `cross_attn_raw` (one
`[T_out, num_frames]` plane per selected head, cropped to `num_frames` and
with the prompt-prefix rows already dropped), `matrix_after_filter` (the
same crop, normalized/median-filtered/head-averaged), `dtw_text_indices` /
`dtw_time_indices` (`_dynamic_time_warping`'s backtrace), `token_timestamps`
(`prompt_ids` zeros + one entry per generated token + one trailing
duplicate), `words_from_generate` (`tokenizer._decode_asr` run on that same
notimestamps decode), `words` (a SEPARATE decode through the
`automatic-speech-recognition` pipeline with `return_timestamps="word"`),
`input_features_stats` / `encoder_last_hidden_state_stats` (`{shape, mean,
std, min, max}` in float64, full precision), and `notes`.

Every float other than the stats dicts (which stay float64) is rounded to 7
significant digits.

Run (offline, CPU, float32):

    HF_HUB_OFFLINE=1 BOOSTR_MODELS_DIR=/path/to/models \\
        python3 whisper_word_oracle.py /path/to/output_dir

Env vars (mirrors `crates/boostr-audio/tests/whisper_word_oracle.rs` /
`tests/common/mod.rs`):

  * `BOOSTR_MODELS_DIR` (required) - root models directory.
  * `WHISPER_TINY_DIR` (optional) - overrides `$BOOSTR_MODELS_DIR/whisper-tiny`.
  * `WHISPER_BASE_DIR` (optional) - overrides `$BOOSTR_MODELS_DIR/whisper-base`.
  * `WHISPER_SMALL_DIR` (optional) - overrides `$BOOSTR_MODELS_DIR/whisper-small`;
    with neither set nor present on disk, falls back to the HF hub id
    `openai/whisper-small` resolved from the local hub cache (requires
    `HF_HUB_OFFLINE=1` and that repo already cached).
  * `WHISPER_WORD_ORACLE_EN_WAV` (optional) - overrides
    `$BOOSTR_MODELS_DIR/librispeech_en_sample.wav` (the `en_speech` clip,
    used whole, language `en`).
  * `WHISPER_MEL_FIXTURE` (optional) - overrides
    `$BOOSTR_MODELS_DIR/whisper_mel_fixture.safetensors`; its `input` tensor,
    cut to the first 320000 samples (20 s at 16 kHz), is the `ms_codeswitch`
    clip, language `ms`.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.numpy import load_file as load_safetensors

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import (  # noqa: E402
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)
from transformers.models.whisper.generation_whisper import (  # noqa: E402
    _dynamic_time_warping,
    _median_filter,
)

SAMPLE_RATE = 16_000
MS_CLIP_SAMPLES = 320_000
TIME_PRECISION = 0.02
TIME_PRECISION_DERIVATION = (
    "HF's fixed default (not derived from config): one decoder-side token step "
    "covers 2 encoder frames (conv2 stride 2), each encoder frame covers a 10ms "
    "mel hop, so 2 * 0.010 = 0.02s per token."
)
MODELS = ["whisper-tiny", "whisper-base", "whisper-small"]
CLIP_LANGUAGE = {"en_speech": "en", "ms_codeswitch": "ms"}


def round_sig(x: float, sig: int = 7) -> float:
    """Round `x` to `sig` significant digits."""
    if x == 0.0 or not math.isfinite(x):
        return x
    digits = sig - int(math.floor(math.log10(abs(x)))) - 1
    return round(x, digits)


def round_floats(obj):
    """Recursively round every float in `obj` (nested lists/dicts) to 7 sig figs."""
    if isinstance(obj, float):
        return round_sig(obj)
    if isinstance(obj, dict):
        return {k: round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v) for v in obj]
    return obj


def require_env_or_default(specific_var: str, models_dir: Path, relative: str) -> Path:
    """Resolve `$<specific_var>`, else `models_dir / relative`. Errors clearly
    when the resolved path does not exist."""
    override = os.environ.get(specific_var)
    path = Path(override) if override else (models_dir / relative)
    if not path.exists():
        raise SystemExit(
            f"error: {path} does not exist; set {specific_var} or BOOSTR_MODELS_DIR"
        )
    return path


def resolve_model_source(model: str, models_dir: Path):
    """Return a path or HF hub id `from_pretrained` can load for `model`."""
    if model == "whisper-tiny":
        return require_env_or_default("WHISPER_TINY_DIR", models_dir, "whisper-tiny")
    if model == "whisper-base":
        return require_env_or_default("WHISPER_BASE_DIR", models_dir, "whisper-base")
    if model == "whisper-small":
        override = os.environ.get("WHISPER_SMALL_DIR")
        if override:
            path = Path(override)
            if not path.exists():
                raise SystemExit(f"error: WHISPER_SMALL_DIR={path} does not exist")
            return path
        local = models_dir / "whisper-small"
        if local.exists():
            return local
        # No local copy: resolve from the HF hub cache (offline).
        return "openai/whisper-small"
    raise SystemExit(f"error: unknown model {model!r}")


def load_en_clip(models_dir: Path) -> np.ndarray:
    path = require_env_or_default(
        "WHISPER_WORD_ORACLE_EN_WAV", models_dir, "librispeech_en_sample.wav"
    )
    audio, sr = sf.read(str(path), dtype="float32")
    if sr != SAMPLE_RATE:
        raise SystemExit(f"error: {path} is {sr} Hz, expected {SAMPLE_RATE} Hz")
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio


def load_ms_clip(models_dir: Path) -> np.ndarray:
    path = require_env_or_default(
        "WHISPER_MEL_FIXTURE", models_dir, "whisper_mel_fixture.safetensors"
    )
    tensors = load_safetensors(str(path))
    if "input" not in tensors:
        raise SystemExit(f"error: {path} is missing the `input` tensor")
    full = tensors["input"]
    if full.shape[0] < MS_CLIP_SAMPLES:
        raise SystemExit(
            f"error: {path}'s `input` tensor ({full.shape[0]} samples) is shorter "
            f"than the ms_codeswitch cut ({MS_CLIP_SAMPLES})"
        )
    return full[:MS_CLIP_SAMPLES].astype(np.float32)


def load_clip(clip: str, models_dir: Path) -> np.ndarray:
    if clip == "en_speech":
        return load_en_clip(models_dir)
    if clip == "ms_codeswitch":
        return load_ms_clip(models_dir)
    raise SystemExit(f"error: unknown clip {clip!r}")


def tensor_stats(data: np.ndarray) -> dict:
    """Shape + summary stats in float64, matching
    `common::TensorStats::from_data` on the Rust side."""
    flat = data.astype(np.float64).reshape(-1)
    return {
        "shape": list(data.shape),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
    }


class _GenerateOutputsView:
    """Wraps the plain `dict` `model.generate(...)` returns (forced by
    `return_token_timestamps=True`, see `_stack_split_outputs`) with the
    attribute access `_extract_token_timestamps` expects, plus the
    `"beam_indices" in generate_outputs` membership check it makes (always
    `False` here: greedy decoding, no beam search)."""

    def __init__(self, outputs: dict):
        self.sequences = outputs["sequences"]
        self.cross_attentions = outputs["cross_attentions"]

    def __contains__(self, key):
        return False


def run_case(model_name: str, clip: str, models_dir: Path) -> dict:
    language = CLIP_LANGUAGE[clip]
    source = resolve_model_source(model_name, models_dir)
    processor = WhisperProcessor.from_pretrained(source)
    # Default (not forced-eager) attention for the encoder: `generate()` itself
    # flips `model.config._attn_implementation` to `"eager"` internally, only
    # once `return_token_timestamps=True` is seen (see generation_whisper.py),
    # so encoding ahead of that call must NOT pre-force eager - doing so
    # changes the encoder's floating-point summation order enough to move
    # `encoder_last_hidden_state_stats` by ~1e-6, versus bit-identical (~1e-14)
    # when the encoder runs under its default attention implementation.
    model = WhisperForConditionalGeneration.from_pretrained(source, dtype=torch.float32)
    model.eval()

    audio = load_clip(clip, models_dir)

    processor.tokenizer.set_prefix_tokens(
        language=language, task="transcribe", predict_timestamps=False
    )
    prompt_ids = list(processor.tokenizer.prefix_tokens)
    prefix_len = len(prompt_ids)

    inputs = processor(
        audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True
    )
    input_features = inputs.input_features
    raw_frames = int(inputs.attention_mask.sum(-1).item())
    num_frames = raw_frames // 2

    with torch.no_grad():
        encoder_outputs = model.get_encoder()(input_features)
        outputs = model.generate(
            input_features,
            language=language,
            task="transcribe",
            return_timestamps=False,
            return_token_timestamps=True,
            return_dict_in_generate=True,
        )
    if not isinstance(outputs, dict):
        raise RuntimeError(
            "expected model.generate(...) to return a plain dict (forced by "
            "return_token_timestamps=True); got "
            f"{type(outputs)!r} - transformers version mismatch?"
        )

    sequences = outputs["sequences"]
    generated_ids = sequences[0, prefix_len:].tolist()
    eos_token_id = processor.tokenizer.eos_token_id
    if generated_ids[-1] != eos_token_id:
        raise RuntimeError(
            f"{model_name}__{clip}: generate() sequence does not end in the "
            f"configured eos id {eos_token_id}, ends in {generated_ids[-1]}"
        )

    alignment_heads = [list(pair) for pair in model.generation_config.alignment_heads]
    median_filter_width = model.config.median_filter_width

    # Replicates `_extract_token_timestamps`'s int-`num_frames`, no-beam-search
    # path (see transformers/models/whisper/generation_whisper.py) far enough
    # to capture the raw cropped per-head planes and the post-filter matrix,
    # neither of which that method exposes on its own.
    per_layer = [
        torch.cat([step[layer] for step in outputs["cross_attentions"]], dim=2)
        for layer in range(model.config.decoder_layers)
    ]
    weights = torch.stack([per_layer[l][:, h] for l, h in alignment_heads])
    weights = weights.permute([1, 0, 2, 3])
    weights = weights[..., :num_frames]
    weights = weights[:, :, prefix_len:, :]
    raw_planes = weights  # [1, num_heads, T_out, num_frames]

    std = torch.std(weights, dim=-2, keepdim=True, unbiased=False)
    mean = torch.mean(weights, dim=-2, keepdim=True)
    filtered = _median_filter((weights - mean) / std, median_filter_width)
    matrix_after_filter = filtered.mean(dim=1)[0]  # [T_out, num_frames]

    text_indices, time_indices = _dynamic_time_warping(
        -matrix_after_filter.cpu().double().numpy()
    )

    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps] * TIME_PRECISION
    token_timestamps = np.concatenate(
        [np.zeros(prefix_len), jump_times, [jump_times[-1]]]
    )

    # Cross-check against HF's own `_extract_token_timestamps` (the method
    # this replicates the internals of), called directly on the same output.
    official = model._extract_token_timestamps(
        _GenerateOutputsView(outputs),
        model.generation_config.alignment_heads,
        time_precision=TIME_PRECISION,
        num_frames=raw_frames,
        num_input_ids=prefix_len,
    )
    max_diff = float(np.abs(official[0].numpy() - token_timestamps).max())
    if max_diff >= 1e-4:
        raise RuntimeError(
            f"{model_name}__{clip}: manual token_timestamps diverge from "
            f"model._extract_token_timestamps by {max_diff}"
        )

    # `words_from_generate`: HF's own `_decode_asr`, run on the SAME
    # notimestamps decode (no timestamp tokens -> the whole sequence resolves
    # as one leftover chunk; expected, not an error).
    asr_outputs = [
        {
            "tokens": sequences,
            "token_timestamps": torch.tensor(
                token_timestamps, dtype=torch.float32
            ).unsqueeze(0),
        }
    ]
    _, asr_info = processor.tokenizer._decode_asr(
        asr_outputs, return_timestamps="word", return_language=False, time_precision=TIME_PRECISION
    )
    words_from_generate = [
        {"text": c["text"], "start": c["timestamp"][0], "end": c["timestamp"][1]}
        for c in asr_info["chunks"]
    ]

    # `words`: the pipeline's own product surface, from a SEPARATE decode WITH
    # timestamp tokens (`return_timestamps="word"` -> `return_timestamps=True`
    # internally). CPU-forced (device=-1): the harness must not touch a GPU.
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=-1,
    )
    pipeline_result = asr_pipeline(
        audio, return_timestamps="word", generate_kwargs={"language": language, "task": "transcribe"}
    )
    words = [
        {"text": c["text"], "start": c["timestamp"][0], "end": c["timestamp"][1]}
        for c in pipeline_result["chunks"]
    ]

    cross_attn_raw = {}
    for idx, (layer, head) in enumerate(alignment_heads):
        plane = raw_planes[0, idx].cpu().numpy().tolist()
        cross_attn_raw[f"layer{layer}_head{head}"] = plane

    fixture = {
        "model": model_name,
        "clip": clip,
        "sample_rate": SAMPLE_RATE,
        "language": language,
        "alignment_heads": alignment_heads,
        "median_filter_width": median_filter_width,
        "time_precision": TIME_PRECISION,
        "time_precision_derivation": TIME_PRECISION_DERIVATION,
        "num_frames": num_frames,
        "prompt_ids": prompt_ids,
        "generated_ids": generated_ids,
        "cross_attn_raw": cross_attn_raw,
        "matrix_after_filter": matrix_after_filter.cpu().numpy().tolist(),
        "dtw_text_indices": text_indices.tolist(),
        "dtw_time_indices": time_indices.tolist(),
        "token_timestamps": token_timestamps.tolist(),
        "words": words,
        "words_from_generate": words_from_generate,
        "notes": (
            "generate_ids/token_timestamps/words_from_generate come from a decode "
            "with return_timestamps=False (notimestamps forced, matching the Rust "
            "decoder). words (pipeline) comes from a SEPARATE decode WITH "
            "timestamp tokens (return_timestamps='word' -> return_timestamps=True "
            "internally). See module docstring for why these two differ."
        ),
    }
    fixture = round_floats(fixture)
    # Stats stay full float64 precision - never rounded to 7 sig figs.
    fixture["input_features_stats"] = tensor_stats(input_features.numpy())
    fixture["encoder_last_hidden_state_stats"] = tensor_stats(
        encoder_outputs.last_hidden_state.numpy()
    )
    return fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="directory to write {model}__{clip}.json into")
    args = parser.parse_args()

    models_dir_str = os.environ.get("BOOSTR_MODELS_DIR")
    if not models_dir_str:
        raise SystemExit("error: BOOSTR_MODELS_DIR is not set")
    models_dir = Path(models_dir_str)
    if not models_dir.exists():
        raise SystemExit(f"error: BOOSTR_MODELS_DIR={models_dir} does not exist")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in MODELS:
        for clip in CLIP_LANGUAGE:
            print(f"running {model_name}__{clip} ...", file=sys.stderr)
            fixture = run_case(model_name, clip, models_dir)
            out_path = output_dir / f"{model_name}__{clip}.json"
            out_path.write_text(json.dumps(fixture))
            print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
