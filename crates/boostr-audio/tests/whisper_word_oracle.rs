//! Oracle test for `boostr::model::audio::alignment_matrix`: a real greedy
//! decode's captured cross-attention, run through `alignment_matrix`, compared
//! against HuggingFace's own `_extract_token_timestamps` intermediates.
//!
//! Fixtures (all under `$BOOSTR_MODELS_DIR`, each overridable by its own var):
//! * `whisper_word_oracle/{model}__{clip}.json` — HF's `alignment_heads`,
//!   `median_filter_width`, `num_frames`, `prompt_ids`, `generated_ids`, the
//!   raw per-head cropped cross-attention planes (`cross_attn_raw`), the
//!   post-normalize/filter matrix (`matrix_after_filter`), and (added by
//!   `scripts/whisper_word_oracle.py`, run once against the same clips)
//!   `input_features_stats` / `encoder_last_hidden_state_stats` — HF's mel
//!   front end and encoder output stats, for isolating whether a raw-plane
//!   mismatch is float drift or a systematic mel/encoder difference.
//! * `librispeech_en_sample.wav` — the `en_speech` clip, used whole.
//! * `whisper_mel_fixture.safetensors`'s `input` tensor, cut to its first
//!   320000 samples (20 s) — the `ms_codeswitch` clip.
//! * `whisper-tiny`, `whisper-base` directories; `whisper-small` resolved from
//!   the HF hub cache the way `huggingface_hub` lays it out, since no local
//!   copy of it exists under `$BOOSTR_MODELS_DIR`.
//!
//! Every test is fixture-gated and skips when a fixture is absent.
//!
//! ## `num_frames` convention
//!
//! This fixture's `num_frames` field is already the crop width
//! `alignment_matrix` computes internally as `num_frames_arg / 2` — i.e. it
//! is HF's raw mel-frame count already halved, not the raw count itself.
//! Confirmed independently from the `ms_codeswitch` clip: it is exactly
//! 320000 samples / 16000 Hz = 20 s, giving 2000 raw 10ms mel frames, and
//! this fixture's `num_frames` is 1000 — exactly half. This test therefore
//! passes `2 * fixture.num_frames` as `alignment_matrix`'s `num_frames`
//! argument, so its internal `/ 2` recovers this fixture's crop width.
//!
//! ## EOS convention
//!
//! `WhisperModel::generate` (and `generate_with_cross_attentions`) never
//! returns an EOS id: `generate_impl` stops before appending it. This
//! fixture's `generated_ids` is HF's raw output, which DOES end in EOS. This
//! test strips that trailing id before comparing, and asserts it really is a
//! configured eos id rather than silently dropping a real token.
//!
//! ## Row layout
//!
//! `alignment_matrix` takes `skip_rows` (HF's `num_input_ids`, i.e.
//! `prompt_ids.len()`) and drops those prefix rows itself, before
//! normalizing — see its doc comment for why this must happen before, not
//! after. Its captured `T_total` (prefix rows plus one row per decode step)
//! equals `prompt_ids.len() + tokens.len()` exactly: EOS is never fed back
//! into the decoder, so every one of this crate's returned tokens (which
//! excludes EOS) has exactly one captured row, and there is no extra
//! "final uncaptured token" to account for. This test asserts that
//! identity, then asserts `alignment_matrix`'s output row count
//! (`T_total - skip_rows`) matches this fixture's `matrix_after_filter`
//! row count.
//!
//! ## Reporting order
//!
//! Every diagnostic (tokens, row counts, per-head raw planes, the alignment
//! matrix, and the mel/encoder stats) is computed and printed BEFORE any of
//! them is asserted: failures are collected into one list and asserted once
//! at the end, so a raw-plane mismatch can never prevent the alignment
//! matrix — the actual product surface — from being computed, printed, and
//! checked in the same run.

use std::collections::BTreeMap;
use std::path::PathBuf;

use boostr::model::audio::mel::{MelOptions, compute_mel_spectrogram_with};
use boostr::model::audio::{GenerateOptions, alignment_matrix};
use boostr::nn::VarMap;
use boostr_audio::WhisperBundle;
use boostr_audio::wav::{decode_wav, to_mono};
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use serde::Deserialize;

mod common;
use common::{cpu_setup, load_json, model_fixture, skip_notice};

const SAMPLE_RATE: usize = 16_000;
const MS_CLIP_SAMPLES: usize = 320_000;
/// Absolute tolerance for the raw (unnormalized) per-head planes: both sides
/// carry the same numbers through the same softmax, so drift is float noise.
const RAW_TOL: f32 = 1e-4;
/// Absolute tolerance for the post-normalize/median-filter matrix: mean/std
/// accumulate over the token axis, so this is looser than `RAW_TOL`.
const MATRIX_TOL: f32 = 2e-3;

#[derive(Debug, Deserialize)]
struct Oracle {
    alignment_heads: Vec<(usize, usize)>,
    median_filter_width: usize,
    num_frames: usize,
    prompt_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    cross_attn_raw: BTreeMap<String, Vec<Vec<f32>>>,
    matrix_after_filter: Vec<Vec<f32>>,
    /// HF's `input_features` stats — absent until `whisper_word_oracle.py`
    /// has been run against this fixture.
    #[serde(default)]
    input_features_stats: Option<TensorStats>,
    /// HF's `encoder_last_hidden_state` stats — same availability.
    #[serde(default)]
    encoder_last_hidden_state_stats: Option<TensorStats>,
}

/// Shape + summary statistics for a tensor, computed in f64. Comparable
/// across HF (`whisper_word_oracle.py`) and this crate regardless of which
/// side produced it.
#[derive(Debug, Clone, Deserialize)]
struct TensorStats {
    shape: Vec<usize>,
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
}

impl TensorStats {
    fn from_data(shape: &[usize], data: &[f32]) -> Self {
        let n = data.len().max(1) as f64;
        let sum: f64 = data.iter().map(|&x| x as f64).sum();
        let mean = sum / n;
        let var: f64 = data.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
        let min = data.iter().copied().fold(f32::INFINITY, f32::min) as f64;
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        Self {
            shape: shape.to_vec(),
            mean,
            std: var.sqrt(),
            min,
            max,
        }
    }
}

/// `layer{L}_head{H}` -> `(L, H)`.
fn parse_head_key(key: &str) -> (usize, usize) {
    let (layer_part, head_part) = key
        .split_once("_head")
        .unwrap_or_else(|| panic!("cross_attn_raw key `{key}` is not `layer<N>_head<M>`"));
    let layer: usize = layer_part
        .strip_prefix("layer")
        .unwrap_or_else(|| panic!("cross_attn_raw key `{key}` has no `layer` prefix"))
        .parse()
        .unwrap_or_else(|e| panic!("cross_attn_raw key `{key}`: bad layer index: {e}"));
    let head: usize = head_part
        .parse()
        .unwrap_or_else(|e| panic!("cross_attn_raw key `{key}`: bad head index: {e}"));
    (layer, head)
}

fn oracle_fixture(model: &str, clip: &str) -> Option<Oracle> {
    let path = model_fixture(
        "WHISPER_WORD_ORACLE_DIR",
        &format!("whisper_word_oracle/{model}__{clip}.json"),
    )?;
    Some(load_json::<Oracle>(&path))
}

/// `whisper-small` ships in no local models tree here; fall back to the HF
/// hub cache layout `huggingface_hub.snapshot_download` produces.
fn whisper_small_dir() -> Option<PathBuf> {
    if let Some(dir) = model_fixture("WHISPER_SMALL_DIR", "whisper-small") {
        return Some(dir);
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        PathBuf::from(home).join(".cache/huggingface/hub/models--openai--whisper-small/snapshots");
    let mut entries = std::fs::read_dir(&snapshots).ok()?;
    entries.find_map(|e| e.ok()).map(|e| e.path())
}

fn model_dir(model: &str) -> Option<PathBuf> {
    match model {
        "whisper-tiny" => model_fixture("WHISPER_TINY_DIR", "whisper-tiny"),
        "whisper-base" => model_fixture("WHISPER_BASE_DIR", "whisper-base"),
        "whisper-small" => whisper_small_dir(),
        other => panic!("unknown oracle model `{other}`"),
    }
}

/// The env var [`model_dir`] checks first for `model`, named for the skip
/// notice so it tells the reader exactly which var enables the test.
fn model_dir_var(model: &str) -> &'static str {
    match model {
        "whisper-tiny" => "WHISPER_TINY_DIR",
        "whisper-base" => "WHISPER_BASE_DIR",
        "whisper-small" => "WHISPER_SMALL_DIR",
        other => panic!("unknown oracle model `{other}`"),
    }
}

fn en_clip_samples() -> Option<Vec<f32>> {
    let path = model_fixture("WHISPER_WORD_ORACLE_EN_WAV", "librispeech_en_sample.wav")?;
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let wav = decode_wav(&bytes).unwrap_or_else(|e| panic!("decoding {}: {e}", path.display()));
    assert_eq!(
        wav.sample_rate as usize, SAMPLE_RATE,
        "en_speech clip is not 16 kHz"
    );
    Some(to_mono(&wav.samples, wav.channels).expect("mono-mixing the en_speech clip"))
}

fn ms_clip_samples(device: &CpuDevice) -> Option<Vec<f32>> {
    let path = model_fixture("WHISPER_MEL_FIXTURE", "whisper_mel_fixture.safetensors")?;
    let map = VarMap::<CpuRuntime>::from_safetensors(&path, device)
        .unwrap_or_else(|e| panic!("loading {}: {e}", path.display()));
    let full: Vec<f32> = map
        .get_tensor("input")
        .unwrap_or_else(|e| panic!("fixture is missing `input`: {e}"))
        .to_vec();
    assert!(
        full.len() >= MS_CLIP_SAMPLES,
        "whisper_mel_fixture `input` is shorter than the ms_codeswitch cut"
    );
    Some(full[..MS_CLIP_SAMPLES].to_vec())
}

fn clip_samples(clip: &str, device: &CpuDevice) -> Option<Vec<f32>> {
    match clip {
        "en_speech" => en_clip_samples(),
        "ms_codeswitch" => ms_clip_samples(device),
        other => panic!("unknown oracle clip `{other}`"),
    }
}

/// Per-head raw-plane diagnostic: where the biggest divergence is and how it
/// compares to the row's own scale, plus a per-row summary so a reviewer can
/// tell "one bad row" from "drift spread evenly across all of them".
struct PlaneDiag {
    max_diff: f32,
    max_row: usize,
    max_frame: usize,
    mean_diff: f32,
    oracle_at_max: f32,
    ours_at_max: f32,
    row_max_value: f32,
    per_row_max_diff: Vec<f32>,
}

fn diff_plane(ours: &[f32], oracle: &[f32], frames: usize) -> PlaneDiag {
    let rows = ours.len() / frames;
    let mut per_row_max_diff = vec![0.0f32; rows];
    let mut max_diff = 0.0f32;
    let mut max_row = 0;
    let mut max_frame = 0;
    let mut sum_diff = 0.0f64;

    for (r, row_max) in per_row_max_diff.iter_mut().enumerate() {
        for f in 0..frames {
            let i = r * frames + f;
            let d = (ours[i] - oracle[i]).abs();
            sum_diff += d as f64;
            if d > *row_max {
                *row_max = d;
            }
            if d > max_diff {
                max_diff = d;
                max_row = r;
                max_frame = f;
            }
        }
    }

    let row_max_value = ours[max_row * frames..(max_row + 1) * frames]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    PlaneDiag {
        max_diff,
        max_row,
        max_frame,
        mean_diff: (sum_diff / ours.len().max(1) as f64) as f32,
        oracle_at_max: oracle[max_row * frames + max_frame],
        ours_at_max: ours[max_row * frames + max_frame],
        row_max_value,
        per_row_max_diff,
    }
}

/// Print HF's mel/encoder stats next to ours, if the oracle carries them
/// (i.e. `whisper_word_oracle.py` has been run against this fixture).
/// Diagnostic only — this crate's mel/encoder implementation is not required
/// to match HF's bit-for-bit, so this never adds to `failures`.
fn report_frontend_stats(
    model: &str,
    clip: &str,
    label: &str,
    ours: &TensorStats,
    theirs: Option<&TensorStats>,
) {
    let Some(theirs) = theirs else {
        eprintln!(
            "{model}__{clip}: {label} stats: ours = {ours:?}; oracle carries none \
             (run whisper_word_oracle.py to add it)"
        );
        return;
    };
    eprintln!(
        "{model}__{clip}: {label} stats: ours = {ours:?}; oracle = {theirs:?}; \
         shapes match = {}, mean diff = {:.6}, std diff = {:.6}, min diff = {:.6}, \
         max diff = {:.6}",
        ours.shape == theirs.shape,
        (ours.mean - theirs.mean).abs(),
        (ours.std - theirs.std).abs(),
        (ours.min - theirs.min).abs(),
        (ours.max - theirs.max).abs(),
    );
}

/// Run one `(model, clip)` oracle case, or print a skip notice and return
/// early when any of its fixtures is absent.
fn run_case(model: &str, clip: &str) {
    let (client, device) = cpu_setup();

    let Some(dir) = model_dir(model) else {
        skip_notice(&format!("{model} checkpoint"), model_dir_var(model));
        return;
    };
    let Some(oracle) = oracle_fixture(model, clip) else {
        skip_notice(
            &format!("{model}__{clip} word-oracle fixture"),
            "WHISPER_WORD_ORACLE_DIR",
        );
        return;
    };
    let Some(samples) = clip_samples(clip, &device) else {
        let var = if clip == "en_speech" {
            "WHISPER_WORD_ORACLE_EN_WAV"
        } else {
            "WHISPER_MEL_FIXTURE"
        };
        skip_notice(&format!("{clip} clip"), var);
        return;
    };

    let bundle = WhisperBundle::<CpuRuntime>::from_dir(&dir, &device)
        .unwrap_or_else(|e| panic!("loading {model}: {e}"));

    let mel_opts = MelOptions::whisper(bundle.num_mel_bins, SAMPLE_RATE);
    let mel = compute_mel_spectrogram_with(&samples, SAMPLE_RATE, &mel_opts)
        .expect("computing the mel front end");
    let num_mel_frames = mel.len() / bundle.num_mel_bins;
    let mel_shape = [1, bundle.num_mel_bins, num_mel_frames];
    let mel_t = Tensor::<CpuRuntime>::from_slice(&mel, &mel_shape, &device).expect("mel tensor");
    let encoded = bundle
        .model
        .encode(&client, &mel_t)
        .expect("whisper encode");
    let encoded_data: Vec<f32> = encoded
        .try_to_vec()
        .expect("host copy of the encoder output");

    // Collected instead of asserted immediately: every diagnostic below must
    // be computed and printed before this function can fail, so a mismatch
    // in an early check (raw planes) never hides whether the alignment
    // matrix — the actual product surface — also passes or fails.
    let mut failures: Vec<String> = Vec::new();

    // Mel/encoder stats vs. HF's, purely diagnostic (see `report_frontend_stats`).
    report_frontend_stats(
        model,
        clip,
        "input_features",
        &TensorStats::from_data(&mel_shape, &mel),
        oracle.input_features_stats.as_ref(),
    );
    report_frontend_stats(
        model,
        clip,
        "encoder_last_hidden_state",
        &TensorStats::from_data(encoded.shape(), &encoded_data),
        oracle.encoder_last_hidden_state_stats.as_ref(),
    );

    let mut layers: Vec<usize> = oracle.alignment_heads.iter().map(|&(l, _)| l).collect();
    layers.sort_unstable();
    layers.dedup();

    let gen_opts = GenerateOptions {
        max_new_tokens: oracle.generated_ids.len(),
        ..bundle.generate_options()
    };
    let (tokens, weights) = bundle
        .model
        .generate_with_cross_attentions(&client, &encoded, &oracle.prompt_ids, &gen_opts, &layers)
        .expect("capturing generate");

    // Our own `generate` never returns EOS; the oracle's `generated_ids` is
    // HF's raw output, which does. Strip it, and confirm it really is EOS.
    let (oracle_eos, oracle_body) = oracle
        .generated_ids
        .split_last()
        .unwrap_or_else(|| panic!("{model}__{clip}: oracle generated_ids is empty"));
    if !gen_opts.eos_token_ids.contains(oracle_eos) {
        failures.push(format!(
            "{model}__{clip}: oracle generated_ids does not end in a configured eos id, ends in {oracle_eos}"
        ));
    }

    eprintln!("{model}__{clip}: produced  ({}) {tokens:?}", tokens.len());
    eprintln!(
        "{model}__{clip}: reference ({}) {oracle_body:?} (+ eos {oracle_eos})",
        oracle_body.len()
    );
    if tokens.as_slice() != oracle_body {
        failures.push(format!(
            "{model}__{clip}: generated ids differ from the oracle"
        ));
    }

    // Row count: HF captures a cross-attention row per prefill token, plus one
    // per decode step that fed a token back into the decoder. EOS is never
    // fed back (generation stops instead), so T_total is exactly
    // prefix_len + tokens.len() — no extra "final uncaptured token" fudge.
    let prefix_len = oracle.prompt_ids.len();
    let (_, first_layer_tensor) = weights
        .layers
        .first()
        .unwrap_or_else(|| panic!("{model}__{clip}: no cross-attention layer captured"));
    let t_total = first_layer_tensor.shape()[2];
    eprintln!(
        "{model}__{clip}: t_total = {t_total}, prefix_len = {prefix_len}, tokens = {}",
        tokens.len()
    );
    if t_total != prefix_len + tokens.len() {
        failures.push(format!(
            "{model}__{clip}: captured row count ({t_total}) must be prefix_len + generated \
             tokens ({}), eos excluded since it is never fed back",
            prefix_len + tokens.len()
        ));
    }
    let kept_rows = t_total - prefix_len;
    eprintln!(
        "{model}__{clip}: kept_rows = {kept_rows}, oracle matrix_after_filter rows = {}",
        oracle.matrix_after_filter.len()
    );
    if kept_rows != oracle.matrix_after_filter.len() {
        failures.push(format!(
            "{model}__{clip}: row count after dropping the prefix ({kept_rows}) must match the \
             oracle ({})",
            oracle.matrix_after_filter.len()
        ));
    }

    // Raw (unnormalized, unfiltered) per-head planes: same crop and same
    // prefix/final-token row drop as the matrix, but no normalize/filter/mean.
    let num_frames_arg = 2 * oracle.num_frames;
    let frames = num_frames_arg / 2;
    for (key, expected_rows) in &oracle.cross_attn_raw {
        let (layer, head) = parse_head_key(key);
        let (_, layer_tensor) = weights
            .layers
            .iter()
            .find(|(l, _)| *l == layer)
            .unwrap_or_else(|| panic!("{model}__{clip}: layer {layer} was not captured"));

        let plane: Vec<f32> = layer_tensor
            .narrow(0, 0, 1)
            .expect("narrow batch")
            .narrow(1, head, 1)
            .expect("narrow head")
            .squeeze(Some(1))
            .squeeze(Some(0))
            .narrow(0, prefix_len, kept_rows)
            .expect("narrow token rows")
            .narrow(1, 0, frames)
            .expect("narrow frame axis")
            // Four chained narrows leave a strided view: `to_vec`/`try_to_vec`
            // read raw contiguous bytes, so make it contiguous first.
            .contiguous()
            .expect("contiguous copy of the raw plane")
            .try_to_vec()
            .expect("host copy of the raw plane for the assertion");

        let expected_flat: Vec<f32> = expected_rows.iter().flatten().copied().collect();
        if plane.len() != expected_flat.len() {
            failures.push(format!(
                "{model}__{clip}: {key} plane size ({}) differs from the oracle ({})",
                plane.len(),
                expected_flat.len()
            ));
            continue;
        }

        let diag = diff_plane(&plane, &expected_flat, frames);
        eprintln!(
            "{model}__{clip}: {key} max abs diff = {} at (row={}, frame={}); mean abs diff = {}; \
             oracle={}, ours={}; row {}'s max value (ours) = {}",
            diag.max_diff,
            diag.max_row,
            diag.max_frame,
            diag.mean_diff,
            diag.oracle_at_max,
            diag.ours_at_max,
            diag.max_row,
            diag.row_max_value,
        );
        eprintln!(
            "{model}__{clip}: {key} per-row max abs diff (restricted to frames < {frames}) = {:?}",
            diag.per_row_max_diff
        );
        if diag.max_diff >= RAW_TOL {
            failures.push(format!(
                "{model}__{clip}: {key} raw plane differs from the oracle by {} at (row={}, \
                 frame={}), over {RAW_TOL}",
                diag.max_diff, diag.max_row, diag.max_frame
            ));
        }
    }

    // The alignment matrix itself — the actual product surface. `alignment_matrix`
    // drops the prefix rows internally (before normalizing), so its output is
    // already `[kept_rows, frames]` — no further slicing needed here.
    let matrix = alignment_matrix(
        &client,
        &weights,
        &oracle.alignment_heads,
        num_frames_arg,
        oracle.median_filter_width,
        prefix_len,
    )
    .expect("alignment_matrix");
    if matrix.shape() != [kept_rows, frames] {
        failures.push(format!(
            "{model}__{clip}: alignment matrix shape {:?} does not match the expected [{kept_rows}, {frames}]",
            matrix.shape()
        ));
    }

    let kept: Vec<f32> = matrix.try_to_vec().expect("host copy of the matrix");
    let expected_flat: Vec<f32> = oracle
        .matrix_after_filter
        .iter()
        .flatten()
        .copied()
        .collect();
    if kept.len() != expected_flat.len() {
        failures.push(format!(
            "{model}__{clip}: alignment matrix size ({}) differs from the oracle ({})",
            kept.len(),
            expected_flat.len()
        ));
    } else {
        let diag = diff_plane(&kept, &expected_flat, frames);
        eprintln!(
            "{model}__{clip}: alignment matrix max abs diff = {} at (row={}, frame={}); \
             mean abs diff = {}",
            diag.max_diff, diag.max_row, diag.max_frame, diag.mean_diff
        );
        if diag.max_diff >= MATRIX_TOL {
            failures.push(format!(
                "{model}__{clip}: alignment matrix differs from the oracle by {} at (row={}, \
                 frame={}), over {MATRIX_TOL}",
                diag.max_diff, diag.max_row, diag.max_frame
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{model}__{clip}: {} check(s) failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn whisper_tiny_en_speech_matches_oracle() {
    run_case("whisper-tiny", "en_speech");
}

#[test]
fn whisper_tiny_ms_codeswitch_matches_oracle() {
    run_case("whisper-tiny", "ms_codeswitch");
}

#[test]
fn whisper_base_en_speech_matches_oracle() {
    run_case("whisper-base", "en_speech");
}

#[test]
fn whisper_base_ms_codeswitch_matches_oracle() {
    run_case("whisper-base", "ms_codeswitch");
}

#[test]
fn whisper_small_en_speech_matches_oracle() {
    run_case("whisper-small", "en_speech");
}

#[test]
fn whisper_small_ms_codeswitch_matches_oracle() {
    run_case("whisper-small", "ms_codeswitch");
}
