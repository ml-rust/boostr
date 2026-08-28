//! End-to-end LoRA fine-tune of VoxCPM2 on real `(text, wav)` pairs.
//!
//! ```text
//! cargo run --release --features audio,f16 --example voxcpm_finetune -- \
//!     (--ckpt CKPT_DIR | --gguf MODEL.gguf [--config config.json]) \
//!     --audiovae audiovae.safetensors --manifest FILE.tsv \
//!     [--device cpu|cuda] [--targets q_proj,v_proj] [--rank 16] [--alpha 32] \
//!     [--lr 1e-4] [--epochs 3] [--seed 0] [--out adapters.safetensors] \
//!     [--lambda-stop 1.0] [--training-cfg-rate 0.1] [--eval-rows 4]
//! ```
//!
//! `CKPT_DIR` holds `config.json`, `model.safetensors` and `tokenizer.json`,
//! same layout `voxcpm_clone`'s `--ckpt` reads. `--audiovae` is the
//! separately converted `audiovae.safetensors` (`convert_audiovae.py`).
//!
//! `--gguf` is the single-file alternative, mutually exclusive with `--ckpt`,
//! same as `voxcpm_clone`. `--config` supplies the `config.json` the file has
//! no embedded copy of, and `tokenizer.json` is looked for beside the
//! `.gguf` and then beside `--config`.
//!
//! # Why the GGUF path is the one QLoRA wants
//!
//! A LoRA adapter trains on TOP of a frozen base — the base never needs to
//! be dense. `--ckpt` loads the transformer stack as dense F32 (~9.2 GB for
//! this model); `--gguf` keeps every matmul weight block-quantized (q6_k:
//! ~1.9 GB packed) and adds dense trainable adapters on top, which is QLoRA.
//! A quantized weight has no `Var` behind it, so a GGUF-loaded model
//! contributes NO base parameters to `trainable_parameters()` — the LoRA
//! adapters are automatically the entire trainable set, same as the
//! `--ckpt` path where the base `Var`s simply never enter `params`.
//!
//! `--manifest` is a TSV with a header row naming its columns; `wav` and
//! `text` are required and are looked up BY NAME, not by column position, so
//! a manifest can carry extra columns (speaker id, duration, ...) this
//! reader ignores. `wav` may be absolute or relative to the manifest's own
//! directory. `ref_wav`, same resolution rules as `wav`, is OPTIONAL — see
//! below.
//!
//! # Why `ref_wav` must name a DIFFERENT clip than `wav`
//!
//! [`VoxCpm2Model::prefill`]/[`prefill_capturing`] take a `ref_feat`
//! argument — the reference-audio conditioning prefix the model is allowed
//! to copy voice characteristics from — and `train_losses`'s
//! `target_patches` is the ground truth both `loss/diff` and `loss/stop`
//! are computed against. Feeding the SAME clip to
//! both is degenerate: the model can pass the reference straight through to
//! the output and score a low, still-falling loss without ever learning to
//! synthesize from text. So `ref_wav` names a *different* clip from the
//! *same speaker* as `wav`, per upstream's fine-tuning guide, and this file
//! never lets `wav`'s own patches serve as `ref_feat`.
//!
//! Upstream also specifies that only 30-50% of training rows should carry a
//! `ref_audio` at all, so the model keeps its zero-shot (no-reference)
//! ability alongside reference-based cloning. Whoever builds the manifest
//! should leave `ref_wav` blank (empty cell, or the column absent entirely)
//! on 50-70% of rows to match that.
//!
//! # Bounding training memory: `--max-patches`
//!
//! Measured peak RSS scales ~1.36 GB per SECOND of target audio (q6_k on
//! CPU: 3.5 s -> 6353 MB, 4.9 s -> 8049 MB, 12.5 s -> 18598 MB). A single
//! 12.5 s clip needs ~18.6 GB. Upstream's own fine-tuning guide hits the
//! same wall and handles it with `max_batch_tokens: 8192`, which FILTERS
//! long samples out of the run rather than shortening them.
//!
//! `--max-patches` (default [`DEFAULT_MAX_PATCHES`]) caps a clip's patch
//! count, but the target `wav` and `ref_wav` are NOT handled the same way,
//! because they play different roles:
//!
//! - The target `wav` is what `loss/diff` and `loss/stop` are computed
//!   against. Truncating it while keeping the full transcript would train
//!   the model to emit part of an utterance for the WHOLE text — a silent
//!   data-corruption bug, not a memory fix. So a row whose target exceeds
//!   the cap is DROPPED, never truncated.
//! - `ref_wav` is speaker conditioning ONLY — nothing computes loss against
//!   it. Truncating it to the cap is exactly what inference already does
//!   (`voxcpm_clone` conditions on a ~3 s reference routinely), so an
//!   over-cap reference is TRUNCATED to its leading
//!   `max_patches * patch_size * HOP_LENGTH` samples instead of dropping the
//!   row.
//!
//! The patch count is computed from the decoded 16 kHz sample count alone
//! (`frames = ceil(samples / HOP_LENGTH)`, `patches = ceil(frames /
//! patch_size)`) so a dropped row never reaches the AudioVAE encoder, let
//! alone the transformer — it costs nothing. Every dropped row is printed
//! with the offending path, its computed patch count, and the cap; every
//! truncated reference is printed separately, since a truncated reference is
//! NOT a dropped row. Manifest filtering ends with a summary line (rows
//! kept, rows skipped, references truncated, total duration retained) and a
//! second line splitting the kept rows into with-reference and
//! no-reference, and if EVERY row is skipped that is a hard error, not a
//! quietly empty run.
//!
//! ## Choosing `--max-patches`
//!
//! Measured on the real corpus (57 takes, min 3.5 s / 22 patches, median
//! 4.9 s / 31 patches, max 12.5 s / 79 patches):
//!
//! | cap       | target retention | est. peak |
//! | --------- | ----------------- | --------- |
//! | 25 (4 s)  | 12/57 (21%)        | ~7.0 GB   |
//! | 38 (6 s)  | 45/57 (78%)        | ~9.8 GB   |
//! | 50 (8 s)  | 50/57 (87%)        | ~12.5 GB  |
//! | 82 (13 s) | 57/57 (100%)       | ~19.3 GB  |
//!
//! [`DEFAULT_MAX_PATCHES`] is 38: keeps 78% of targets while staying under
//! ~10 GB peak. Raise `--max-patches` deliberately if the extra retention is
//! worth the extra memory.
//!
//! # `loss/diff` is not a progress signal — the eval batch is
//!
//! [`PatchGenerator::train_losses`] draws a FRESH flow-matching timestep
//! `t`, fresh noise, and a fresh CFG-dropout coin from the seed on every
//! training step. So the per-step `loss/diff`/`loss/stop` printed in the
//! training loop moves with WHICH `t`/noise got sampled that step, not with
//! how much the model has learned — a low value can be an easy `t`, a high
//! value a hard one, independent of training progress. Do not read a trend
//! into it step to step.
//!
//! `--eval-rows` (default [`DEFAULT_EVAL_ROWS`]) carves that many rows off
//! the END of the kept (post-filter) manifest, by index, before training
//! starts. For each held-out row this file decodes the audio, truncates the
//! reference exactly as training does, runs the same prefill path, and
//! draws exactly ONE `t` and ONE `noise` from the fixed [`EVAL_NOISE_SEED`]
//! — not `--seed`, not `step_counter`, so the eval metric stays comparable
//! across runs and across every step of a run.
//!
//! Only `t` and `noise` are cached. The prefill and target patches are
//! REBUILT at every eval pass, against the current weights. Caching them
//! would pin the conditioning to the weights at initialization, and since
//! the default `--targets q_proj,v_proj` adapt the very projections the
//! prefill runs through, `eval/diff` would then never see the LM learn.
//!
//! Once per epoch, after that epoch's training rows finish, every eval row
//! is scored with `train_losses_with_noise(..., drop_cond = false)` — always
//! the conditioned branch, since that is what inference actually runs — and
//! printed as `eval/diff`, `eval/stop`, `eval/total` (the mean over eval
//! rows), a series distinct from the per-row training prints. Pinning
//! `drop_cond = false` means `eval/diff` and the training `loss/diff` are
//! NOT perfectly apples-to-apples: training mixes in `--training-cfg-rate`
//! conditioning dropout, eval never does. The eval forward pass never calls
//! `backward` or the optimizer. `--eval-rows 0` disables eval outright: no
//! split, no eval batch, no eval logging.
//!
//! A row without a `ref_wav` trains ZERO-SHOT: `prefill_capturing` gets
//! `None`, and
//! [`SequenceLayout::build`](boostr::model::audio::voxcpm::model::sequence::SequenceLayout::build)
//! drops the reference prefix entirely, matching upstream's no-ref packer. A
//! missing `ref_wav` NEVER falls back to self-referencing `wav`.
//!
//! The run summary prints the with/without-reference split. That printed
//! split is the ONLY check on a manifest whose `ref_wav` column got renamed
//! or lost: such a run trains entirely zero-shot and otherwise looks healthy.
//!
//! # Why `prefill_capturing`, never plain `prefill`
//!
//! [`PatchGenerator::teacher_forced_conditioning`] (what
//! [`PatchGenerator::train_losses`] calls internally, ONCE, sharing it
//! between `loss/diff` and `loss/stop`) needs [`PrefillState::intermediates`] whenever
//! `prefill.position > 0` — the prefix embeddings it re-runs through a
//! batched forward pass live only there, not in the KV caches
//! `MiniCpm4Model::forward` cannot read. Every row here has a non-empty
//! prefix (at least the reference patches and `AUDIO_START_ID`), so
//! `prefill.position` is always `> 0`, and this file therefore ALWAYS calls
//! `prefill_capturing`, never the plain `prefill` `voxcpm_clone` uses for
//! generation (which never needs the batched teacher-forced path).
//!
//! # Why `load_lora_parameters` runs every step
//!
//! [`SimpleTrainer::step`] writes the optimizer's updated values into the
//! `HashMap<TensorId, Tensor<R>>` it is given — it does NOT touch the
//! `Var`s the model's `MaybeLoraLinear` adapters hold. Skipping
//! `VoxCpm2Model::load_lora_parameters` after a step means every subsequent
//! forward pass recomputes from the SAME pre-update weights: the loss would
//! never move. So it runs after every `trainer.step` that actually
//! finalizes (gradient accumulation defaults to 1 step here, so that is
//! every row).
//!
//! # Saving
//!
//! Adapters are written the same way every time: ONLY the adapter tensors
//! (`named_parameters()` entries ending `lora_a`/`lora_b`), named by their
//! full checkpoint-style path, via this crate's own
//! [`boostr::format::safetensors::save_safetensors`] writer — never a
//! hand-rolled one — carrying the same [`build_lora_metadata`] (rank, alpha,
//! targets) on every write, so every artifact this file produces loads in
//! `voxcpm_clone` the same way. That writer accepts CPU tensors only, so
//! each adapter tensor is round-tripped through `to_bytes`/`from_bytes` (the
//! same device-to-host pattern `trainer::async_checkpoint` uses), which
//! works whether the run trained on CPU or CUDA.
//!
//! Two artifact layouts come out of `--out PATH`:
//!
//! - A numbered file PER EPOCH, `PATH` with `.epochN` inserted before its
//!   extension (`lora.safetensors` -> `lora.epoch1.safetensors`,
//!   `lora.epoch2.safetensors`, ... — a `PATH` with no extension gets
//!   `lora.epoch1`; see [`epoch_checkpoint_path`]). Written after every
//!   epoch, unconditionally.
//! - `PATH` itself, unmodified — the PRIMARY artifact `voxcpm_clone` is
//!   expected to load. With eval enabled (`--eval-rows` > 0) this is the
//!   BEST epoch by `eval/total`, not necessarily the last: the trainer
//!   overwrites `PATH` only when an epoch's `eval/total` beats every prior
//!   epoch's, so a run that diverges in its final epoch still leaves the
//!   best checkpoint at `PATH`, with the worse-but-later epochs available
//!   only under their numbered names. With `--eval-rows 0` there is no
//!   `eval/total` to rank epochs by, so `PATH` instead gets the LAST epoch's
//!   adapters, same as this file did before per-epoch saving existed — the
//!   end-of-run log line says explicitly that no selection happened.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use boostr::format::safetensors::save_safetensors;
use boostr::model::audio::voxcpm::model::config::{AUDIO_START_ID, VoxCpm2Config};
use boostr::model::audio::voxcpm::model::{PatchGenerator, VoxCpm2Model};
use boostr::model::audio::voxcpm::{
    PrefillState, VoxCpmClient, load_tokenizer, normalize_whitespace, tokenize,
};
use boostr::model::audio::{decode_audio, extension_hint, to_mono_at_rate};
use boostr::nn::{LoraTargets, Module, build_lora_metadata};
use boostr::ops::FusedOptimizerOps;
use boostr::quant::traits::DequantOps;
use boostr::trainer::{SimpleTrainer, TrainingConfig};
use numr::autograd::backward_wrt;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, RandomOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
#[cfg(feature = "cuda")]
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::{Tensor, TensorId};
use splintr::AnyTokenizer;

/// Rate every manifest wav is resampled to before the AudioVAE encoder.
/// Fixed by the encoder, not a choice — see `voxcpm_clone.rs`'s identical
/// constant.
const REF_RATE: u32 = 16_000;

const DEFAULT_TARGETS: &str = "q_proj,v_proj";
const DEFAULT_RANK: usize = 16;
const DEFAULT_ALPHA: f32 = 32.0;
const DEFAULT_LR: f64 = 1e-4;
const DEFAULT_EPOCHS: usize = 3;
const DEFAULT_SEED: u64 = 0;
/// `lambda_stop` in upstream's `lambdas:` fine-tuning block. Upstream's own
/// default is `1.0`, matched here; upstream's FAQ names runaway generation
/// ("generation doesn't stop") as a top failure mode and recommends raising
/// this weight when it happens — see `--lambda-stop` in [`USAGE`].
const DEFAULT_LAMBDA_STOP: f64 = 1.0;
/// `lambda_diff` in upstream's `lambdas:` block. Fixed at upstream's own
/// default — unlike `lambda_stop`, upstream's FAQ names no failure mode
/// that calls for retuning it, so it is not exposed as a flag.
const LAMBDA_DIFF: f64 = 1.0;
/// `training_cfg_rate` — upstream's default, matched here. Upstream's FAQ
/// calls text-ignoring "the most common fine-tuning failure mode" and says
/// explicitly not to train with this at 0: `--training-cfg-rate` exists so
/// an operator can raise it, not so it gets turned off.
const DEFAULT_TRAINING_CFG_RATE: f64 = 0.1;
/// Default `--max-patches`: retains 78% of targets on the measured real
/// corpus while staying under ~10 GB peak RSS — see the module docs'
/// "Choosing `--max-patches`" table. Measured scaling is ~1.36 GB per second
/// of target audio (q6_k, CPU), so a 6.0 s cap keeps peak near `6.0 * 1.36
/// GB ≈ 8.2 GB` plus fixed model/runtime overhead — ~9.8 GB total. Converted
/// to patches at `patch_size = 4`
/// ([`VoxCpm2Config::default`](boostr::model::audio::voxcpm::model::config::VoxCpm2Config::default),
/// the checkpoint's usual value) and `HOP_LENGTH` (640): `6.0 s * 16_000
/// samples/s = 96_000 samples`; `ceil(96_000 / 640) = 150 frames`;
/// `ceil(150 / 4) = 38 patches`. A checkpoint with a different `patch_size`
/// shifts the seconds-per-patch ratio this default assumes, so pass
/// `--max-patches` explicitly for one.
const DEFAULT_MAX_PATCHES: usize = 38;
/// Default `--eval-rows`: how many of the kept manifest rows are held out
/// for the fixed eval batch — see the module docs' "Eval batch" section.
const DEFAULT_EVAL_ROWS: usize = 4;
/// Seed for the eval batch's ONE draw of `t` and noise per row. Fixed here,
/// not derived from `--seed` or `step_counter`, so the eval metric is
/// comparable across runs that only differ in `--seed` — the whole point of
/// the eval batch is a number that moves with LEARNING, not with sampling.
const EVAL_NOISE_SEED: u64 = 0xE7A1_5EED;

/// Where the transformer stack's weights come from.
///
/// `--ckpt` names a checkpoint DIRECTORY (`config.json`,
/// `model.safetensors`, `tokenizer.json`); `--gguf` names a single file that
/// carries the weights and nothing else. Mutually exclusive, and one of them
/// is required.
enum Weights {
    Checkpoint(PathBuf),
    Gguf(PathBuf),
}

/// Runtime to load the model and train on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Device {
    Cpu,
    Cuda,
}

fn parse_device(value: &str) -> Result<Device, String> {
    match value {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda),
        other => Err(format!(
            "--device: expected one of cpu, cuda, got {other:?}"
        )),
    }
}

struct Args {
    weights: Weights,
    /// `config.json` for the GGUF path. Ignored for `--ckpt`, which reads the
    /// one in the checkpoint directory.
    config: Option<PathBuf>,
    audiovae: PathBuf,
    manifest: PathBuf,
    device: Device,
    targets: String,
    rank: usize,
    alpha: f32,
    lr: f64,
    epochs: usize,
    seed: u64,
    out: Option<PathBuf>,
    lambda_stop: f64,
    /// Upstream's `training_cfg_rate`: the per-step probability of
    /// conditioning dropout during training. Upstream's FAQ calls text
    /// ignoring "the most common fine-tuning failure mode" and says
    /// explicitly DO NOT set this to 0 — leave it at the default unless a
    /// specific reason says otherwise.
    training_cfg_rate: f64,
    /// Upper bound on the target `wav`'s patch count — see the module docs'
    /// "Bounding training memory" section. A row whose target exceeds it is
    /// dropped. A `ref_wav` over the same cap is truncated instead, never
    /// dropped, since it is conditioning only.
    max_patches: usize,
    /// Rows carved off the END of the kept (post-filter) manifest for the
    /// fixed eval batch — see the module docs' "Eval batch" section. `0`
    /// disables eval entirely.
    eval_rows: usize,
    /// Run every transformer layer with activation checkpointing: drop the
    /// intermediates during the forward pass and recompute them during
    /// backward. Cuts the activation memory that dominates training peak
    /// VRAM, at ~33% extra compute. OFF by default, so a run without the
    /// flag behaves exactly as it did before the flag existed.
    activation_checkpointing: bool,
}

const USAGE: &str = "usage: voxcpm_finetune (--ckpt DIR | --gguf MODEL.gguf [--config config.json]) \
--audiovae audiovae.safetensors \
--manifest FILE.tsv (header-named TSV: wav, text, optional ref_wav) \
[--device cpu|cuda] [--targets q_proj,v_proj] [--rank 16] \
[--alpha 32] [--lr 1e-4] [--epochs 3] [--seed 0] [--out adapters.safetensors] \
[--lambda-stop 1.0] [--training-cfg-rate 0.1 (DO NOT set to 0 — upstream's FAQ \
names text-ignoring as the most common fine-tuning failure mode)] \
[--max-patches 38 (caps the target wav's patch count; over-cap targets are \
dropped, over-cap ref_wav clips are truncated to the cap instead — see the \
module docs)] \
[--eval-rows 4 (rows held out for the fixed eval batch; 0 disables eval)] \
[--checkpoint (activation checkpointing: recompute each layer's intermediates \
during backward instead of holding them, ~33% slower, much less VRAM)]";

/// Consume the value that follows `flag`, advancing `i` past it.
fn take_value(argv: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    *i += 1;
    argv.get(*i)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (mut ckpt, mut audiovae, mut manifest) = (None, None, None);
    let mut gguf: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut device = Device::Cpu;
    let mut targets = DEFAULT_TARGETS.to_string();
    let mut rank = DEFAULT_RANK;
    let mut alpha = DEFAULT_ALPHA;
    let mut lr = DEFAULT_LR;
    let mut epochs = DEFAULT_EPOCHS;
    let mut seed = DEFAULT_SEED;
    let mut out = None;
    let mut lambda_stop = DEFAULT_LAMBDA_STOP;
    let mut training_cfg_rate = DEFAULT_TRAINING_CFG_RATE;
    let mut max_patches = DEFAULT_MAX_PATCHES;
    let mut eval_rows = DEFAULT_EVAL_ROWS;
    let mut activation_checkpointing = false;

    let mut i = 0usize;
    while i < argv.len() {
        let flag = argv[i].as_str();
        match flag {
            "--ckpt" => ckpt = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--gguf" => gguf = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--config" => config = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--audiovae" => audiovae = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--manifest" => manifest = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--device" => device = parse_device(&take_value(&argv, &mut i, flag)?)?,
            "--targets" => targets = take_value(&argv, &mut i, flag)?,
            "--rank" => {
                rank = take_value(&argv, &mut i, flag)?
                    .parse()
                    .map_err(|e| format!("--rank: {e}"))?
            }
            "--alpha" => {
                alpha = take_value(&argv, &mut i, flag)?
                    .parse()
                    .map_err(|e| format!("--alpha: {e}"))?
            }
            "--lr" => {
                lr = take_value(&argv, &mut i, flag)?
                    .parse()
                    .map_err(|e| format!("--lr: {e}"))?
            }
            "--epochs" => {
                epochs = take_value(&argv, &mut i, flag)?
                    .parse()
                    .map_err(|e| format!("--epochs: {e}"))?
            }
            "--seed" => {
                seed = take_value(&argv, &mut i, flag)?
                    .parse()
                    .map_err(|e| format!("--seed: {e}"))?
            }
            "--out" => out = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--lambda-stop" => {
                lambda_stop = take_value(&argv, &mut i, flag)?
                    .parse()
                    .map_err(|e| format!("--lambda-stop: {e}"))?
            }
            "--training-cfg-rate" => {
                training_cfg_rate = take_value(&argv, &mut i, flag)?
                    .parse()
                    .map_err(|e| format!("--training-cfg-rate: {e}"))?
            }
            "--max-patches" => {
                max_patches = take_value(&argv, &mut i, flag)?
                    .parse()
                    .map_err(|e| format!("--max-patches: {e}"))?
            }
            "--eval-rows" => {
                eval_rows = take_value(&argv, &mut i, flag)?
                    .parse()
                    .map_err(|e| format!("--eval-rows: {e}"))?
            }
            "--checkpoint" => activation_checkpointing = true,
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown flag {other}\n{USAGE}")),
        }
        i += 1;
    }

    if rank == 0 {
        return Err("--rank must be at least 1".to_string());
    }
    if epochs == 0 {
        return Err("--epochs must be at least 1".to_string());
    }
    if targets.trim().is_empty() {
        return Err("--targets must name at least one projection".to_string());
    }
    if !(0.0..=1.0).contains(&training_cfg_rate) {
        return Err(format!(
            "--training-cfg-rate must be in [0.0, 1.0], got {training_cfg_rate}"
        ));
    }
    if max_patches == 0 {
        return Err("--max-patches must be at least 1".to_string());
    }

    // Exactly one weight source. Accepting both and silently preferring one
    // would load a different model than the operator asked for.
    let weights = match (ckpt, gguf) {
        (Some(_), Some(_)) => {
            return Err(format!("--ckpt and --gguf are mutually exclusive\n{USAGE}"));
        }
        (Some(dir), None) => Weights::Checkpoint(dir),
        (None, Some(path)) => Weights::Gguf(path),
        (None, None) => return Err(format!("--ckpt or --gguf is required\n{USAGE}")),
    };

    Ok(Args {
        weights,
        config,
        audiovae: audiovae.ok_or_else(|| format!("--audiovae is required\n{USAGE}"))?,
        manifest: manifest.ok_or_else(|| format!("--manifest is required\n{USAGE}"))?,
        device,
        targets,
        rank,
        alpha,
        lr,
        epochs,
        seed,
        out,
        lambda_stop,
        training_cfg_rate,
        max_patches,
        eval_rows,
        activation_checkpointing,
    })
}

/// Locate `tokenizer.json`.
///
/// A checkpoint directory holds it outright. A GGUF carries no tokenizer at
/// all, so it is looked for beside the `.gguf` first and beside `--config`
/// second — both of those normally sit in, or are copied from, the same
/// checkpoint directory. Neither: an error, rather than a tokenizer guess
/// that would silently produce the wrong token ids.
fn tokenizer_path(weights: &Weights, config: Option<&Path>) -> Result<PathBuf, String> {
    match weights {
        Weights::Checkpoint(dir) => Ok(dir.join("tokenizer.json")),
        Weights::Gguf(path) => {
            let beside = |p: &Path| {
                p.parent()
                    .map(|dir| dir.join("tokenizer.json"))
                    .filter(|candidate| candidate.is_file())
            };
            beside(path)
                .or_else(|| config.and_then(beside))
                .ok_or_else(|| {
                    format!(
                        "no tokenizer.json beside {} (a GGUF carries none); put it there \
                     or pass --config pointing into the checkpoint directory",
                        path.display()
                    )
                })
        }
    }
}

/// One `(wav, text, ref_wav)` row resolved from the manifest.
struct ManifestRow {
    wav: PathBuf,
    text: String,
    /// The reference-conditioning clip, a DIFFERENT clip from the same
    /// speaker as `wav` — never `wav` itself. `None` when the manifest row
    /// left the (optional) `ref_wav` column empty or absent.
    ref_wav: Option<PathBuf>,
}

/// Resolve a manifest-relative wav path: absolute paths pass through,
/// everything else is joined onto `manifest_dir`.
fn resolve_wav_path(manifest_dir: &Path, field: &str) -> PathBuf {
    let path = PathBuf::from(field);
    if path.is_absolute() {
        path
    } else {
        manifest_dir.join(path)
    }
}

/// Parse a header-named TSV manifest: `wav` and `text` columns are required
/// and an optional `ref_wav` column, all located by NAME so extra columns
/// (speaker id, duration, ...) are ignored rather than rejected. A row with
/// no `ref_wav` value (empty cell, short row, or the column absent from the
/// header entirely) gets `ManifestRow::ref_wav == None`. `wav`/`ref_wav`
/// paths are resolved relative to the manifest's own directory when not
/// already absolute.
fn load_manifest(path: &Path) -> Result<Vec<ManifestRow>, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("{}: failed to read manifest: {e}", path.display()))?;
    let mut lines = contents.lines();

    let header = lines.next().ok_or_else(|| {
        format!(
            "{}: manifest is empty, expected a header row",
            path.display()
        )
    })?;
    let columns: Vec<&str> = header.split('\t').map(str::trim).collect();
    let wav_idx = columns.iter().position(|c| *c == "wav").ok_or_else(|| {
        format!(
            "{}: manifest header missing required column \"wav\" (found: {columns:?})",
            path.display()
        )
    })?;
    let text_idx = columns.iter().position(|c| *c == "text").ok_or_else(|| {
        format!(
            "{}: manifest header missing required column \"text\" (found: {columns:?})",
            path.display()
        )
    })?;
    // Optional: absent entirely means every row trains without a reference.
    let ref_wav_idx = columns.iter().position(|c| *c == "ref_wav");
    let needed = wav_idx.max(text_idx) + 1;

    let manifest_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut rows = Vec::new();
    for (offset, line) in lines.enumerate() {
        let line_no = offset + 2; // 1 for the header, 1 for 1-indexing
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < needed {
            return Err(format!(
                "{}:{line_no}: expected at least {needed} tab-separated column(s), got {}",
                path.display(),
                fields.len()
            )
            .into());
        }
        let wav_path = resolve_wav_path(manifest_dir, fields[wav_idx].trim());
        // A short row (ref_wav column present in the header but this line
        // has fewer fields) is the same as an empty cell: no reference.
        let ref_wav = ref_wav_idx
            .and_then(|idx| fields.get(idx))
            .map(|field| field.trim())
            .filter(|field| !field.is_empty())
            .map(|field| resolve_wav_path(manifest_dir, field));
        rows.push(ManifestRow {
            wav: wav_path,
            text: fields[text_idx].trim().to_string(),
            ref_wav,
        });
    }
    if rows.is_empty() {
        return Err(format!("{}: no data rows after the header", path.display()).into());
    }
    Ok(rows)
}

/// Patch count a clip of `samples` 16 kHz samples folds to, WITHOUT running
/// the AudioVAE encoder. `VoxCpm2Config::ref_pad_multiple` right-pads to a
/// multiple of `patch_size * HOP_LENGTH` before the real encode, so the true
/// frame count is always a multiple of `patch_size`; `ceil(ceil(samples /
/// HOP_LENGTH) / patch_size)` equals `ceil(samples / (patch_size *
/// HOP_LENGTH))`, exactly that padded-then-folded patch count — this is not
/// an approximation. Reuses [`VoxCpm2Config::ref_pad_multiple`] rather than
/// re-deriving `patch_size * HOP_LENGTH` here.
fn estimate_patches(samples: usize, cfg: &VoxCpm2Config) -> usize {
    samples.div_ceil(cfg.ref_pad_multiple())
}

/// Truncate a `ref_wav`'s 16 kHz samples to its leading
/// `max_patches * ref_pad_multiple()` samples — the same cap
/// [`estimate_patches`] checks against — so the AudioVAE encoder never sees
/// the excess. Safe ONLY for the reference clip, never the target `wav`:
/// see the module docs' "Bounding training memory" section for why the two
/// are treated differently. A clip already at or under the cap is returned
/// unchanged.
fn truncate_reference(mut samples: Vec<f32>, cfg: &VoxCpm2Config, max_patches: usize) -> Vec<f32> {
    let cap_samples = max_patches * cfg.ref_pad_multiple();
    samples.truncate(cap_samples);
    samples
}

/// Filter `rows` to those whose target `wav` folds to at most `max_patches`
/// patches, without ever running the AudioVAE encoder — see the module
/// docs' "Bounding training memory" section. An over-cap `ref_wav` is NOT a
/// reason to drop the row — it is reported here as truncated (a separate
/// count from skipped rows) and the actual truncation happens where
/// `ref_wav` is loaded for training, via [`truncate_reference`]; this
/// function only measures and reports. Decodes each candidate clip once
/// here (cheap PCM decode, not the VAE) purely to read its sample count;
/// the training loop below decodes again per epoch, same as it always has.
/// Prints the kept rows' with-reference / no-reference split, the only
/// signal that a manifest lost its `ref_wav` column.
fn filter_rows_by_patch_cap<'a>(
    rows: &'a [ManifestRow],
    cfg: &VoxCpm2Config,
    max_patches: usize,
) -> Result<Vec<&'a ManifestRow>, Box<dyn std::error::Error>> {
    let mut kept = Vec::new();
    let mut skipped = 0usize;
    let mut truncated_refs = 0usize;
    let mut retained_seconds = 0.0f64;

    for row in rows {
        let wav = load_wav_16k(&row.wav).map_err(|e| format!("{}: {e}", row.wav.display()))?;
        let wav_patches = estimate_patches(wav.len(), cfg);
        if wav_patches > max_patches {
            eprintln!(
                "skip {}: {wav_patches} patches > --max-patches {max_patches}",
                row.wav.display()
            );
            skipped += 1;
            continue;
        }

        if let Some(ref_wav_path) = &row.ref_wav {
            let ref_wav = load_wav_16k(ref_wav_path)
                .map_err(|e| format!("{}: {e}", ref_wav_path.display()))?;
            let ref_patches = estimate_patches(ref_wav.len(), cfg);
            if ref_patches > max_patches {
                eprintln!(
                    "truncate {}: ref_wav {} {ref_patches} patches > --max-patches \
                     {max_patches}, using the leading {max_patches} (reference is speaker \
                     conditioning only, never the training target — see the module docs)",
                    row.wav.display(),
                    ref_wav_path.display()
                );
                truncated_refs += 1;
            }
        }

        retained_seconds += wav.len() as f64 / f64::from(REF_RATE);
        kept.push(row);
    }

    eprintln!(
        "manifest filter: {} row(s) kept, {skipped} skipped, {truncated_refs} reference(s) \
         truncated, {retained_seconds:.1}s retained (--max-patches {max_patches})",
        kept.len()
    );
    // MANDATORY, never drop this line. Removing the old hard error on a
    // missing `ref_wav` removed the only thing that caught a manifest whose
    // `ref_wav` column got renamed or lost. Without the printed split such a
    // run trains entirely zero-shot, looks healthy, and surfaces days later
    // as a model that never learned reference cloning.
    let with_ref = kept.iter().filter(|row| row.ref_wav.is_some()).count();
    eprintln!(
        "manifest: {with_ref} row(s) with reference, {} without (upstream recommends 30-50% \
         of rows WITH a reference, so most rows train reference-free; that is what keeps \
         zero-shot cloning alive)",
        kept.len() - with_ref
    );
    if kept.is_empty() {
        return Err(format!(
            "every one of {} manifest row(s) exceeds --max-patches {max_patches}; nothing to \
             train on",
            rows.len()
        )
        .into());
    }
    Ok(kept)
}

/// Read a manifest wav as mono 16 kHz PCM, matching `voxcpm_clone.rs`'s
/// `load_reference` exactly (`decode_audio` plus `to_mono_at_rate`).
fn load_wav_16k(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let hint = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(extension_hint);
    let data = decode_audio(&bytes, hint)?;
    Ok(to_mono_at_rate(&data, REF_RATE)?)
}

/// Copy any-runtime tensor data to host and rebuild it as a CPU tensor, the
/// same device-to-host pattern `trainer::async_checkpoint::TensorSnapshot`
/// uses. `save_safetensors` accepts CPU tensors only.
fn to_cpu_tensor<R: Runtime<DType = DType>>(
    tensor: &Tensor<R>,
) -> Result<Tensor<CpuRuntime>, Box<dyn std::error::Error>> {
    let bytes = tensor.to_bytes()?;
    let device = CpuDevice::default();
    Ok(Tensor::<CpuRuntime>::from_bytes(
        &bytes,
        tensor.shape(),
        tensor.dtype(),
        &device,
    )?)
}

/// Derive the per-epoch checkpoint path for epoch `epoch` from `--out`'s
/// `out`, inserting `.epochN` before the extension. Operates on `out`'s
/// FILE NAME only (`Path::file_stem`/`extension` already do this), so a
/// directory component containing dots (`a.b/lora.safetensors`) never
/// affects where the extension is split — a naive string split on `'.'`
/// would corrupt exactly that case. `out` with no extension
/// (`lora`) gets `lora.epoch1`, no trailing dot.
fn epoch_checkpoint_path(out: &Path, epoch: usize) -> PathBuf {
    let stem = out
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("adapters");
    let file_name = match out.extension().and_then(|s| s.to_str()) {
        Some(ext) => format!("{stem}.epoch{epoch}.{ext}"),
        None => format!("{stem}.epoch{epoch}"),
    };
    match out.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(file_name),
        _ => PathBuf::from(file_name),
    }
}

/// Collect every LoRA adapter tensor (`named_parameters()` entries ending
/// `lora_a`/`lora_b`) as CPU tensors, keyed by their full checkpoint-style
/// path.
fn collect_adapter_tensors<R: Runtime<DType = DType>>(
    model: &VoxCpm2Model<R>,
) -> Result<HashMap<String, Tensor<CpuRuntime>>, Box<dyn std::error::Error>> {
    let mut out = HashMap::new();
    for (name, var) in Module::named_parameters(model) {
        if name.ends_with("lora_a") || name.ends_with("lora_b") {
            out.insert(name, to_cpu_tensor(var.tensor())?);
        }
    }
    Ok(out)
}

/// Decode `row`'s target and (truncated) reference audio, encode both
/// through the AudioVAE, tokenize the text, and run `prefill_capturing` —
/// the exact per-row setup the training loop and the eval-batch builder both
/// need. Shared here so the sequence is defined once. A row without a
/// `ref_wav` builds the zero-shot form: `prefill_capturing` gets `None` and
/// the reference prefix is absent, so `S == text_token_ids.len()`.
fn build_prefill_and_target<R, C>(
    model: &VoxCpm2Model<R>,
    client: &C,
    tokenizer: &AnyTokenizer,
    row: &ManifestRow,
    max_patches: usize,
) -> Result<(PrefillState<R>, Tensor<R>), Box<dyn std::error::Error>>
where
    R: Runtime<DType = DType>,
    C: VoxCpmClient<R> + TypeConversionOps<R> + 'static,
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + TypeConversionOps<R>
        + DequantOps<R>,
{
    let wav = load_wav_16k(&row.wav).map_err(|e| format!("{}: {e}", row.wav.display()))?;
    // The training target: what the loss is computed against.
    let target_patches = model.encode_reference(client, &wav)?;

    // The reference-conditioning clip MUST be a different clip than `wav` —
    // see the module docs for why self-referencing is degenerate. A row with
    // no `ref_wav` trains zero-shot (`None`), never a silent fallback to
    // `target_patches`.
    let ref_patches = match &row.ref_wav {
        Some(ref_wav_path) => {
            let ref_wav = load_wav_16k(ref_wav_path)
                .map_err(|e| format!("{}: {e}", ref_wav_path.display()))?;
            // Truncate, never drop: `ref_wav` is speaker conditioning only,
            // not the loss target — see [`truncate_reference`] and the module
            // docs' "Bounding training memory" section.
            let ref_wav = truncate_reference(ref_wav, &model.config, max_patches);
            Some(model.encode_reference(client, &ref_wav)?)
        }
        None => None,
    };

    let normalized = normalize_whitespace(&row.text);
    let mut text_token_ids = tokenize(tokenizer, &normalized);
    // `prefill` requires the sequence to end here: AUDIO_START_ID is the
    // position the first (only, here) patch attends from.
    text_token_ids.push(AUDIO_START_ID);
    // S, exactly: the reference prefix contributes `t_ref + 2` rows, and
    // nothing at all when there is no reference.
    let max_length = match &ref_patches {
        Some(ref_patches) => ref_patches.shape()[0] + 2 + text_token_ids.len(),
        None => text_token_ids.len(),
    };

    // ALWAYS `prefill_capturing`: `cfm_loss`'s teacher-forced path needs
    // `PrefillState::intermediates`, and every row here has a non-empty
    // prefix — see the module docs.
    let prefill =
        model.prefill_capturing(client, ref_patches.as_ref(), &text_token_ids, max_length)?;
    Ok((prefill, target_patches))
}

/// One eval row: the manifest row plus the FIXED `t`/`noise` it is always
/// scored with. Only weight-INDEPENDENT state is cached here. `prefill` is
/// deliberately NOT cached: it runs the base and residual LMs, whose
/// `q_proj`/`v_proj` are the default LoRA targets, so a cached `prefill`
/// would pin the conditioning to the weights at initialization and
/// `eval/diff` would never see the LM learn. See the module docs' "Eval
/// batch" section.
struct EvalRow<'a, R: Runtime> {
    row: &'a ManifestRow,
    t: Tensor<R>,
    noise: Tensor<R>,
}

/// Build the fixed eval batch from the LAST `eval_rows` of `kept_rows`,
/// drawing each row's `t`/`noise` ONCE from [`EVAL_NOISE_SEED`] — never
/// re-derived per epoch, never mixed with `args.seed` or `step_counter`, so
/// the eval metric stays comparable across runs and across steps within a
/// run. See the module docs' "Eval batch" section.
fn build_eval_batch<'a, R, C>(
    model: &VoxCpm2Model<R>,
    client: &C,
    tokenizer: &AnyTokenizer,
    eval_source_rows: &[&'a ManifestRow],
    max_patches: usize,
) -> Result<Vec<EvalRow<'a, R>>, Box<dyn std::error::Error>>
where
    R: Runtime<DType = DType>,
    C: VoxCpmClient<R> + TypeConversionOps<R> + RandomOps<R> + 'static,
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + TypeConversionOps<R>
        + DequantOps<R>,
{
    let mut eval_batch = Vec::with_capacity(eval_source_rows.len());
    for (eval_index, row) in eval_source_rows.iter().enumerate() {
        // Built once here ONLY to read the target's shape and dtype, which
        // fix `t`/`noise`. Both are dropped immediately: every eval pass
        // rebuilds them against the CURRENT weights (see `EvalRow`).
        let (prefill, target_patches) =
            build_prefill_and_target(model, client, tokenizer, row, max_patches)?;
        let tcount = target_patches.shape()[0];
        let dtype = target_patches.dtype();
        let noise_shape = target_patches.shape().to_vec();
        drop(prefill);
        drop(target_patches);
        // Stride 2, matching `train_losses`'s own `seed`/`seed + 1` split
        // between the timestep and noise draws — each eval row gets its own
        // pair of streams off the same fixed base, never reused across rows.
        let row_seed = EVAL_NOISE_SEED.wrapping_add((eval_index as u64).wrapping_mul(2));
        let t = client.rand_seeded(&[tcount], dtype, row_seed)?;
        let noise = client.randn_seeded(&noise_shape, dtype, row_seed.wrapping_add(1))?;
        eval_batch.push(EvalRow { row, t, noise });
    }
    Ok(eval_batch)
}

/// Mean `(diff, stop, total)` over `eval_batch`, scored with `drop_cond =
/// false` (the conditioned branch — what inference actually runs; the
/// module docs explain why this makes eval not perfectly apples-to-apples
/// with train's CFG-dropout-mixed `loss/diff`). Forward only: never calls
/// `backward` or the optimizer. numr's autograd has no no-grad/detached-
/// forward context (checked: no `no_grad`/`NoGrad` construct exists, only
/// per-`Var` `detach()`/`requires_grad()`, and the model's own LoRA `Var`s
/// still require grad through this call), so each row is built, scored, and
/// dropped before the next row starts, keeping graph retention from stacking
/// across the batch.
///
/// `prefill`/`target_patches` are rebuilt HERE, every call, against the
/// current weights — only `t`/`noise` come cached from [`EvalRow`]. That is
/// what makes `eval/diff` a learning signal: the sampling is pinned, the
/// model is not.
fn score_eval_batch<R, C>(
    model: &VoxCpm2Model<R>,
    generator: &PatchGenerator<'_, R>,
    client: &C,
    tokenizer: &AnyTokenizer,
    eval_batch: &[EvalRow<'_, R>],
    max_patches: usize,
    lambda_stop: f64,
) -> Result<(f64, f64, f64), Box<dyn std::error::Error>>
where
    R: Runtime<DType = DType>,
    C: VoxCpmClient<R> + TypeConversionOps<R> + 'static,
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + TypeConversionOps<R>
        + DequantOps<R>,
{
    let mut diff_sum = 0.0f64;
    let mut stop_sum = 0.0f64;
    let mut total_sum = 0.0f64;
    for eval_row in eval_batch {
        let (prefill, target_patches) =
            build_prefill_and_target(model, client, tokenizer, eval_row.row, max_patches)?;
        let losses = generator.train_losses_with_noise(
            client,
            &prefill,
            &target_patches,
            &eval_row.t,
            &eval_row.noise,
            LAMBDA_DIFF,
            lambda_stop,
            false,
        )?;
        diff_sum += losses.diff.tensor().to_vec::<f32>()[0] as f64;
        stop_sum += losses.stop.tensor().to_vec::<f32>()[0] as f64;
        total_sum += losses.total.tensor().to_vec::<f32>()[0] as f64;
        // `losses` (and the autograd graph it pinned alive) drops here,
        // before the next row's forward pass starts.
    }
    let n = eval_batch.len() as f64;
    Ok((diff_sum / n, stop_sum / n, total_sum / n))
}

/// The model-and-training body: everything that runs on the chosen runtime
/// `R`. Loads the checkpoint, adapts it with LoRA, then trains one epoch at
/// a time over every manifest row, printing per-step and per-epoch loss.
fn run<R: Runtime<DType = DType>>(
    args: &Args,
    device: &R::Device,
    client: &(
         impl VoxCpmClient<R> + TypeConversionOps<R> + RandomOps<R> + FusedOptimizerOps<R> + 'static
     ),
    rows: &[ManifestRow],
    started: Instant,
) -> Result<(), Box<dyn std::error::Error>>
where
    R::Client: TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + TypeConversionOps<R>
        // A quantized projection's backward dequantizes the frozen weight to
        // carry the gradient through — the QLoRA path.
        + DequantOps<R>,
{
    // F32, not the checkpoint's native BF16: AdamW's running moments and the
    // CFM loss's backward pass are far more numerically stable in F32, and
    // this is a training loop, not `voxcpm_clone`'s inference path. On the
    // GGUF path `None` is the correct dtype request, not `Some(DType::F32)`:
    // `from_gguf` already dequantizes every DENSE tensor to F32 by default
    // when `dtype` is `None`, and every quantized matmul weight stays
    // packed for `quant_matmul`, which requires F32 activations and REJECTS
    // an explicit `Some(BF16)`/`Some(F16)` cast request outright (it would
    // mean dequantizing the very weights this path exists to keep packed).
    // Requesting `Some(DType::F32)` here would work too (it agrees with the
    // default), but `None` says "no cast" precisely, matching
    // `voxcpm_clone.rs`'s own GGUF arm.
    let mut model = match &args.weights {
        Weights::Checkpoint(dir) => {
            eprintln!("loading {} ...", dir.display());
            VoxCpm2Model::<R>::from_checkpoint(dir, &args.audiovae, device, Some(DType::F32))?
        }
        Weights::Gguf(path) => {
            eprintln!("loading {} (base stays quantized) ...", path.display());
            VoxCpm2Model::<R>::from_gguf(
                path,
                args.config.as_deref(),
                &args.audiovae,
                device,
                None,
            )?
        }
    };

    // Activation checkpointing, applied to every stack a training pass runs
    // (`feat_encoder`, `base_lm`, `residual_lm`, `feat_decoder`). Set BEFORE
    // any forward pass so the eval batch and the training loop agree.
    model.set_activation_checkpointing(args.activation_checkpointing);
    if args.activation_checkpointing {
        eprintln!(
            "activation checkpointing: ON (--checkpoint) — layer intermediates are \
             recomputed during backward, ~33% extra compute"
        );
    } else {
        eprintln!("activation checkpointing: off (pass --checkpoint to enable)");
    }

    // Filter BEFORE any LoRA/optimizer setup: a row over --max-patches must
    // never reach `encode_reference` (the AudioVAE encoder), which is the
    // whole point of the cap — see the module docs.
    let rows = filter_rows_by_patch_cap(rows, &model.config, args.max_patches)?;

    // Carve the eval set off the END of the kept rows, by index — no RNG, no
    // shuffle, so the split is deterministic and reproducing a run always
    // yields the same train/eval partition. `--eval-rows 0` disables eval
    // entirely: no split, no eval batch, no eval logging.
    if args.eval_rows >= rows.len() {
        return Err(format!(
            "--eval-rows {} >= {} kept manifest row(s); that would leave zero training rows",
            args.eval_rows,
            rows.len()
        )
        .into());
    }
    let split_at = rows.len() - args.eval_rows;
    let (train_rows, eval_source_rows) = rows.split_at(split_at);
    eprintln!(
        "eval split: {} training row(s), {} eval row(s) (--eval-rows {})",
        train_rows.len(),
        eval_source_rows.len(),
        args.eval_rows
    );
    let rows = train_rows;

    let tokenizer = load_tokenizer(tokenizer_path(&args.weights, args.config.as_deref())?)?;

    let target_names: Vec<String> = args
        .targets
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let lora_targets = LoraTargets::new(target_names.clone());
    let adapted = model.apply_lora(&lora_targets, args.rank, args.alpha, device)?;
    eprintln!(
        "LoRA: targets={target_names:?} rank={} alpha={} -> {adapted} projection(s) adapted",
        args.rank, args.alpha
    );

    let mut params: HashMap<TensorId, Tensor<R>> = Module::trainable_parameter_tensors(&model);
    eprintln!("trainable adapter tensors: {}", params.len());
    // The ONLY ids any optimizer step reads back. `backward_wrt` prunes the
    // traversal to these, which is the semantically correct request: a plain
    // `backward` also stores a full-size gradient under every id nothing can
    // read back.
    //
    // MEASURED: this prunes almost NOTHING here, and is not why the trainer is
    // capped at batch 1. Peak VRAM 11819 MiB with it vs 11808 without, runtime
    // unchanged, losses bit-identical. The reason is that ~120 LoRA adapters
    // sit throughout the network, so nearly every node is an ancestor of some
    // wanted id and survives pruning. The real cost is forward ACTIVATION
    // LIFETIME: training state measured 6266 MiB at a 24-patch cap and 8831
    // MiB at 31, so it scales with sequence length. Activation checkpointing
    // is the fix, and `--checkpoint` turns it on; do not expect this call to
    // deliver memory.
    //
    // Collected once — the adapter set never changes after `apply_lora`.
    let wanted: Vec<TensorId> = params.keys().copied().collect();

    let config = TrainingConfig::default().with_lr(args.lr);
    let mut trainer = SimpleTrainer::<R>::new(config)?;

    eprintln!(
        "manifest: {} row(s), {} epoch(s), lr={}, lambda_diff={LAMBDA_DIFF}, lambda_stop={}",
        rows.len(),
        args.epochs,
        args.lr,
        args.lambda_stop
    );

    // Built ONCE, here, before any training step — never rebuilt or
    // re-derived per epoch. `prefill`/`target_patches`/`t`/`noise` are all
    // fixed for the whole run: `eval/diff` moving is then a LEARNING signal,
    // not a resampling artifact. Empty when `--eval-rows 0`.
    let eval_batch = if args.eval_rows > 0 {
        eprintln!(
            "building eval batch ({} row(s)) ...",
            eval_source_rows.len()
        );
        build_eval_batch(
            &model,
            client,
            &tokenizer,
            eval_source_rows,
            args.max_patches,
        )?
    } else {
        Vec::new()
    };

    // Metadata is identical for every checkpoint this run writes (per-epoch
    // and the primary `--out`) — collected once, reused on every save so an
    // epoch checkpoint is never missing the rank/alpha/targets
    // `check_lora_metadata` hard-requires on load.
    let lora_metadata = build_lora_metadata(args.rank, args.alpha, &target_names);
    // Tracks the best epoch by `eval/total` so far. `None` until the first
    // eval pass runs; stays `None` for the whole run when `--eval-rows 0`,
    // which is the signal used below to fall back to "last epoch wins".
    let mut best_epoch: Option<(usize, f64)> = None;

    let mut step_counter: u64 = 0;
    for epoch in 1..=args.epochs {
        let mut epoch_diff_sum = 0.0f64;
        let mut epoch_stop_sum = 0.0f64;
        let mut epoch_steps = 0usize;

        for (row_index, row) in rows.iter().enumerate() {
            let (prefill, target_patches) =
                build_prefill_and_target(&model, client, &tokenizer, row, args.max_patches)?;

            let generator = model.patch_generator();
            // Stride 3, not 2: `train_losses` consumes THREE independent
            // streams per call — `seed` for the flow timestep, `seed + 1` for
            // the noise, `seed + 2` for the conditioning-dropout draw. A
            // stride of 2 would put this step's dropout draw on the same seed
            // value as the next step's timestep draw, correlating consecutive
            // steps. The stride must match the number of streams the callee
            // uses.
            let seed_for_step = args.seed.wrapping_add(step_counter.wrapping_mul(3));
            let losses = generator.train_losses(
                client,
                &prefill,
                &target_patches,
                seed_for_step,
                LAMBDA_DIFF,
                args.lambda_stop,
                args.training_cfg_rate,
            )?;
            let diff_val = losses.diff.tensor().to_vec::<f32>()[0] as f64;
            let stop_val = losses.stop.tensor().to_vec::<f32>()[0] as f64;
            let loss_val = losses.total.tensor().to_vec::<f32>()[0] as f64;

            let grads = backward_wrt(&losses.total, &wanted, client)?;
            if let Some(_metrics) = trainer.step(client, &mut params, grads, loss_val)? {
                // REQUIRED every finalized step: `trainer.step` updates
                // `params` only, not the model's own `Var`s — see the
                // module docs.
                model.load_lora_parameters(&params)?;
                epoch_diff_sum += diff_val;
                epoch_stop_sum += stop_val;
                epoch_steps += 1;
            }

            eprintln!(
                "epoch {epoch}/{} row {row_index}/{}: loss/diff {diff_val:.6} loss/stop \
                 {stop_val:.6} total {loss_val:.6}",
                args.epochs,
                rows.len()
            );
            step_counter += 1;
        }

        let (diff_mean, stop_mean) = if epoch_steps > 0 {
            (
                epoch_diff_sum / epoch_steps as f64,
                epoch_stop_sum / epoch_steps as f64,
            )
        } else {
            (f64::NAN, f64::NAN)
        };
        eprintln!(
            "epoch {epoch}/{} mean loss/diff: {diff_mean:.6} mean loss/stop: {stop_mean:.6}",
            args.epochs
        );

        let mut this_epoch_eval_total = None;
        if !eval_batch.is_empty() {
            let generator = model.patch_generator();
            let (eval_diff, eval_stop, eval_total) = score_eval_batch(
                &model,
                &generator,
                client,
                &tokenizer,
                &eval_batch,
                args.max_patches,
                args.lambda_stop,
            )?;
            eprintln!(
                "epoch {epoch}/{} eval/diff {eval_diff:.6} eval/stop {eval_stop:.6} eval/total \
                 {eval_total:.6} ({} eval row(s))",
                args.epochs,
                eval_batch.len()
            );
            this_epoch_eval_total = Some(eval_total);
        }

        // Per-epoch save: unconditional, since a diverging final epoch must
        // never be the only artifact on disk — see the module docs'
        // "Saving" section.
        if let Some(out) = &args.out {
            let adapters = collect_adapter_tensors(&model)?;
            let epoch_path = epoch_checkpoint_path(out, epoch);
            eprintln!(
                "saving epoch {epoch}/{} adapters ({} tensor(s)) to {} ...",
                args.epochs,
                adapters.len(),
                epoch_path.display()
            );
            save_safetensors(&epoch_path, &adapters, Some(&lora_metadata))?;

            // `PATH` (`--out`) tracks the best epoch by `eval/total` when
            // eval is enabled, and the LAST epoch when it is not (eval
            // disabled leaves `this_epoch_eval_total` `None` every epoch, so
            // this branch always overwrites `out`, matching the old
            // save-once-at-the-end behaviour).
            let is_best = match (this_epoch_eval_total, best_epoch) {
                (Some(total), Some((_, best_total))) => total < best_total,
                (Some(_), None) => true,
                (None, _) => true,
            };
            if is_best {
                save_safetensors(out, &adapters, Some(&lora_metadata))?;
                if let Some(total) = this_epoch_eval_total {
                    best_epoch = Some((epoch, total));
                }
            }
        }
    }

    if let Some(out) = &args.out {
        match best_epoch {
            Some((epoch, eval_total)) => eprintln!(
                "best epoch: {epoch}/{} (eval/total {eval_total:.6}) written to {}",
                args.epochs,
                out.display()
            ),
            None => eprintln!(
                "--eval-rows 0: no eval basis for selecting a best epoch; {} holds the final \
                 epoch ({}/{}) unmodified, not a selected best",
                out.display(),
                args.epochs,
                args.epochs
            ),
        }
    }

    eprintln!("total {:.1}s", started.elapsed().as_secs_f64());
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let started = Instant::now();

    let rows = load_manifest(&args.manifest)?;

    match args.device {
        Device::Cpu => {
            let device = CpuDevice::default();
            let client = CpuClient::new(device.clone());
            run::<CpuRuntime>(&args, &device, &client, &rows, started)?;
        }
        #[cfg(feature = "cuda")]
        Device::Cuda => {
            let device = CudaDevice::new(0);
            let client = CudaClient::new(device.clone())?;
            run::<CudaRuntime>(&args, &device, &client, &rows, started)?;
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda => {
            eprintln!(
                "--device cuda: this binary was built without CUDA support; rebuild with \
                 --features cuda"
            );
            std::process::exit(2);
        }
    }

    Ok(())
}
