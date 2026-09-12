//! End-to-end LoRA fine-tune of VoxCPM2 on real `(text, wav)` pairs.
//!
//! ```text
//! cargo run --release --features audio,f16 --example voxcpm_finetune -- \
//!     (--ckpt CKPT_DIR | --gguf MODEL.gguf | --tcf MODEL.tcf) \
//!     [--config config.json] \
//!     --audiovae audiovae.safetensors --manifest FILE.tsv \
//!     [--device cpu|cuda] [--targets q_proj,v_proj] [--rank 16] [--alpha 32] \
//!     [--lr 1e-4] [--epochs 3] [--seed 0] [--out adapters.safetensors] \
//!     [--lambda-stop 1.0] [--training-cfg-rate 0.1] [--eval-rows 4] \
//!     [--eval-only] [--dequant-weights]
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
//! `--tcf` is the third single-file form, mutually exclusive with both of the
//! above, loaded through
//! [`VoxCpm2Model::from_tcf`](boostr::model::audio::voxcpm::model::VoxCpm2Model::from_tcf)
//! on exactly the terms `voxcpm_clone`'s own `--tcf` arm uses — same loader,
//! same auxiliary inputs (`--config`, `--audiovae`, the tokenizer looked for
//! beside the model file and then beside `--config`). `--config` is REQUIRED
//! with `--tcf` rather than optional: the format carries no metadata map a
//! `config.json` could be embedded in. `--dtype` is not a flag here — the
//! training path always requests `None` (no cast), for the reason the `--gguf`
//! arm documents.
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
//! *same speaker* as `wav`, per the reference VoxCPM fine-tuning guide, and this file
//! never lets `wav`'s own patches serve as `ref_feat`.
//!
//! The reference VoxCPM fine-tuning guide also specifies that only 30-50% of training rows should carry a
//! `ref_audio` at all, so the model keeps its zero-shot (no-reference)
//! ability alongside reference-based cloning. Whoever builds the manifest
//! should leave `ref_wav` blank (empty cell, or the column absent entirely)
//! on 50-70% of rows to match that.
//!
//! # Bounding training memory: `--max-patches`
//!
//! Measured peak RSS scales ~1.36 GB per SECOND of target audio (q6_k on
//! CPU: 3.5 s -> 6353 MB, 4.9 s -> 8049 MB, 12.5 s -> 18598 MB). A single
//! 12.5 s clip needs ~18.6 GB. The reference VoxCPM fine-tuning guide hits the
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
//! # `--eval-only`: the metric as a standalone measurement
//!
//! `--eval-only` loads an artifact, scores the eval rows once, prints the
//! metrics and exits. It is the flag a format comparison uses: two artifacts
//! of the same base weights, differing only in how those weights are stored,
//! scored over one manifest with one seed. Teacher-forced CFM loss plus stop
//! loss gives thousands of paired scalar residuals per utterance, where a
//! transcription metric over the same corpus gives a handful of word
//! decisions.
//!
//! `--epochs 0` does NOT already do this — it is rejected outright
//! (`--epochs must be at least 1`), and even accepted it would still build
//! the optimizer and take the save path. So the mode is its own flag.
//!
//! `--eval-only` skips, in order: `apply_lora` (no adapters are allocated, so
//! the artifact is scored exactly as it sits on disk), the trainable-parameter
//! collection, `SimpleTrainer`, the epoch loop, `backward_wrt`,
//! `load_lora_parameters`, and every checkpoint write. `--out` is rejected
//! rather than ignored, since an eval-only run has nothing to save. `--lr`,
//! `--epochs`, `--rank`, `--alpha`, `--targets` and `--training-cfg-rate`
//! reach nothing in this mode.
//!
//! The eval-row selection differs in one deliberate way. Training must leave
//! rows to train on, so `--eval-rows N` carves N off the END of the kept rows
//! and `0` disables eval. Eval-only has no training half, so `--eval-rows 0`
//! means EVERY kept row is scored — the manifest is itself the held-out set.
//! A nonzero `--eval-rows N` still takes the last N kept rows, by index, so a
//! training run and an eval-only run over the same manifest with the same
//! `--eval-rows` score the same rows in the same order.
//!
//! # Comparing two artifacts: `--dequant-weights`
//!
//! A cross-format quality number is only meaningful when both artifacts run
//! the SAME activation contract (`boostr/src/tcf/FORMAT.md` Section 9.3).
//! The two single-file paths do match each other: a block TCF declares the
//! `GGML_REFERENCE` contract, which is the kernel family a GGUF of the same
//! ggml types runs, so a TCF and the GGUF it was built from score identically.
//! Neither matches `--ckpt`:
//!
//! - `--ckpt` loads dense F32 and the matmul runs f32 activations.
//! - `--gguf` and `--tcf` on CUDA at `m >= 2` take the feature-major MMQ
//!   path, which quantizes the activations (Q8_1-style, per-32 dynamic
//!   scale) before the tensor-core MMA.
//!
//! The packed side therefore absorbs activation-quantization error the dense
//! side never pays, and the difference reads as a weight-encoding difference
//! when it is nothing of the kind.
//!
//! `--dequant-weights` removes the confound. It materializes EVERY packed
//! weight to dense F32 at load, on the `--gguf` and `--tcf` paths alike
//! (`from_gguf_dense`/`from_tcf_dense`, both of which run the codec's own
//! `DequantOps::dequantize` — the same op the quantized-projection backward
//! already uses, and the same kernels `quant_matmul` decodes with). With it
//! set the forward pass is dense F32 end to end on both formats, so the only
//! difference left between two artifacts is the weight VALUES — which is the
//! weight-encoding damage the comparison is after.
//!
//! `--ckpt` already loads dense F32, so the flag changes nothing there and
//! says so at load.
//!
//! Everything else is untouched by the flag: the loss, the row selection, the
//! seed and the `t`/noise draw are identical in both modes, so a with-flag and
//! a without-flag run of one artifact differ ONLY in weight representation.
//!
//! It is a MEASUREMENT mode. A dense stack costs what an unquantized
//! checkpoint costs, which is the whole thing the packed path exists to
//! avoid — never fine-tune or serve a quantized artifact through it.
//!
//! The mode is printed at load and recorded in the JSON line's
//! `weights_dense` field, so two result files can never be compared across
//! modes by accident.
//!
//! # Machine-readable eval output
//!
//! Every eval pass — per epoch during training, once under `--eval-only` —
//! writes one JSON object on ONE line to STDOUT, alongside the unchanged
//! human-readable line on stderr. Every other line this file prints goes to
//! stderr, so stdout carries the metric lines and nothing else, and two runs
//! diff directly.
//!
//! The object carries `eval_diff`, `eval_stop`, `eval_total`, the row count
//! (`eval_rows`) and the seed (`eval_seed`) the pairing rests on, plus the
//! weight source, the manifest, the device, `max_patches` and the loss
//! weights. `epoch` is the 1-based epoch under training and `null` under
//! `--eval-only`. `mode` is `"train"` or `"eval-only"`. `weights_dense` is
//! `--dequant-weights`: two objects that disagree on it were scored under
//! different activation contracts and must not be compared.
//!
//! # What "deterministic" means here, exactly
//!
//! Two `--eval-only` runs of the same artifact, manifest and `--eval-rows` on
//! the same build and device print byte-identical metrics. Pinned:
//!
//! - `t` and the noise, drawn ONCE per eval row from [`EVAL_NOISE_SEED`] (a
//!   fixed constant, never `--seed`, never `step_counter`), at row-indexed
//!   stride 2.
//! - Row order and row membership: the manifest is read in file order, the
//!   `--max-patches` filter preserves that order, and the eval split takes a
//!   suffix by index. No shuffle, no RNG anywhere in the selection.
//! - The conditioning branch: `drop_cond = false` always, so no dropout coin
//!   is ever flipped in eval.
//! - The mean: `diff_sum / n` over rows visited in that same fixed order, so
//!   the floating-point summation order is fixed too.
//! - No weight updates run in `--eval-only`, so the scored weights are the
//!   artifact's own.
//!
//! What breaks it: a different `--max-patches` (changes which rows survive
//! and how far a reference is truncated), a different `--eval-rows`, a
//! reordered or edited manifest, a different `--lambda-stop` (`eval_stop`'s
//! weight in `eval_total`), a different `--device`, and any change to the
//! loss itself.
//!
//! NOT established as deterministic, and deliberately not claimed:
//!
//! - ACROSS devices. `--device cpu` and `--device cuda` run different
//!   kernels; nothing here checks that they agree bit for bit, and a
//!   comparison must hold the device fixed.
//! - ACROSS builds. Feature flags select different kernels, and any
//!   multi-threaded reduction inside a kernel is free to reorder its partial
//!   sums between runs. This file does not audit numr's kernels for
//!   thread-order-independent reductions, so run-to-run bit-identity is an
//!   empirical property of the backend, not a guarantee this file can make.
//!   Verify it by running `--eval-only` twice and diffing the stdout lines
//!   before trusting a small difference between two artifacts.
//! - ACROSS artifacts of different numeric encodings. That difference is the
//!   thing being MEASURED; only the sampling is pinned, so a nonzero delta
//!   between two formats is signal, not noise.
//!
//! A row without a `ref_wav` trains ZERO-SHOT: `prefill_capturing` gets
//! `None`, and
//! [`SequenceLayout::build`](boostr::model::audio::voxcpm::model::sequence::SequenceLayout::build)
//! drops the reference prefix entirely, matching the reference VoxCPM
//! implementation's no-ref packer. A
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
//!
//! # Where the eval machinery lives
//!
//! The manifest reader, the `--max-patches` filter, the per-row
//! prefill/target build and the eval loss itself live in the sibling
//! `eval_common` module, not here. `sensitivity.rs` measures a per-tensor
//! task delta against the SAME loss over the SAME rows with the SAME pinned
//! `t`/noise, and two copies of that loss would drift and make the two
//! binaries' numbers incomparable. Every determinism guarantee stated above
//! is a property of that module's code, unchanged by the move.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use boostr::format::safetensors::save_safetensors;
use boostr::model::audio::voxcpm::model::VoxCpm2Model;
use boostr::model::audio::voxcpm::{VoxCpmClient, load_tokenizer};
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

// Sibling module: the manifest reader, the `--max-patches` filter, the
// per-row prefill/target build, and the eval loss. Shared with
// `sensitivity.rs` so the loss and its pinned sampling have ONE definition —
// see `eval_common.rs`'s module docs.
mod eval_common;
use eval_common::{
    DEFAULT_EVAL_ROWS, DEFAULT_MAX_PATCHES, EVAL_NOISE_SEED, LAMBDA_DIFF, ManifestRow,
    build_eval_batch, build_prefill_and_target, filter_rows_by_patch_cap, load_manifest,
    score_eval_batch,
};

const DEFAULT_TARGETS: &str = "q_proj,v_proj";
const DEFAULT_RANK: usize = 16;
const DEFAULT_ALPHA: f32 = 32.0;
const DEFAULT_LR: f64 = 1e-4;
const DEFAULT_EPOCHS: usize = 3;
const DEFAULT_SEED: u64 = 0;
/// `lambda_stop` in the reference VoxCPM `lambdas:` fine-tuning block. The
/// reference implementation's own default is `1.0`, matched here; its FAQ names runaway generation
/// ("generation doesn't stop") as a top failure mode and recommends raising
/// this weight when it happens — see `--lambda-stop` in [`USAGE`].
const DEFAULT_LAMBDA_STOP: f64 = 1.0;
/// `training_cfg_rate` — the reference VoxCPM implementation's default,
/// matched here. Its FAQ
/// calls text-ignoring "the most common fine-tuning failure mode" and says
/// explicitly not to train with this at 0: `--training-cfg-rate` exists so
/// an operator can raise it, not so it gets turned off.
const DEFAULT_TRAINING_CFG_RATE: f64 = 0.1;

/// Where the transformer stack's weights come from.
///
/// `--ckpt` names a checkpoint DIRECTORY (`config.json`,
/// `model.safetensors`, `tokenizer.json`); `--gguf` and `--tcf` each name a
/// single file that carries the weights and nothing else. Mutually exclusive,
/// and exactly one of them is required.
enum Weights {
    Checkpoint(PathBuf),
    Gguf(PathBuf),
    Tcf(PathBuf),
}

/// The weight source's short name, for the machine-readable eval record.
/// Same vocabulary `voxcpm_clone`'s JSONL uses, so a comparison table can key
/// on one set of names across both binaries.
fn source_format(weights: &Weights) -> &'static str {
    match weights {
        Weights::Checkpoint(_) => "checkpoint",
        Weights::Gguf(_) => "gguf",
        Weights::Tcf(_) => "tcf",
    }
}

/// The weight file or directory, for the machine-readable eval record.
fn source_path(weights: &Weights) -> &Path {
    match weights {
        Weights::Checkpoint(path) | Weights::Gguf(path) | Weights::Tcf(path) => path,
    }
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
    /// `config.json` for the single-file paths. Ignored for `--ckpt`, which
    /// reads the one in the checkpoint directory. Optional for `--gguf`,
    /// required for `--tcf`.
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
    /// The reference VoxCPM implementation's `training_cfg_rate`: the
    /// per-step probability of
    /// conditioning dropout during training. Its FAQ calls text
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
    /// disables eval entirely while training, and means "every kept row" under
    /// [`Args::eval_only`].
    eval_rows: usize,
    /// Score the artifact once and exit: no LoRA, no optimizer, no weight
    /// update, no checkpoint — see the module docs' "`--eval-only`" section.
    eval_only: bool,
    /// Run every transformer layer with activation checkpointing: drop the
    /// intermediates during the forward pass and recompute them during
    /// backward. Cuts the activation memory that dominates training peak
    /// VRAM, at ~33% extra compute. OFF by default, so a run without the
    /// flag behaves exactly as it did before the flag existed.
    activation_checkpointing: bool,
    /// Materialize every packed weight to dense F32 at load, on the `--gguf`
    /// and `--tcf` paths alike, so the forward pass is dense F32 end to end
    /// and two artifacts differ only in weight VALUES — see the module docs'
    /// "Comparing two artifacts" section. OFF by default, so a run without
    /// the flag behaves exactly as it did before the flag existed.
    dequant_weights: bool,
}

const USAGE: &str = "usage: voxcpm_finetune (--ckpt DIR | --gguf MODEL.gguf | --tcf MODEL.tcf) \
[--config config.json (required with --tcf)] \
--audiovae audiovae.safetensors \
--manifest FILE.tsv (header-named TSV: wav, text, optional ref_wav) \
[--device cpu|cuda] [--targets q_proj,v_proj] [--rank 16] \
[--alpha 32] [--lr 1e-4] [--epochs 3] [--seed 0] [--out adapters.safetensors] \
[--lambda-stop 1.0] [--training-cfg-rate 0.1 (DO NOT set to 0 — the reference VoxCPM FAQ \
names text-ignoring as the most common fine-tuning failure mode)] \
[--max-patches 38 (caps the target wav's patch count; over-cap targets are \
dropped, over-cap ref_wav clips are truncated to the cap instead — see the \
module docs)] \
[--eval-rows 4 (rows held out for the fixed eval batch; 0 disables eval while \
training, and means every kept row under --eval-only)] \
[--eval-only (score the artifact once and exit: no LoRA, no optimizer, no \
weight update, no checkpoint; prints one JSON metric line on stdout)] \
[--checkpoint (activation checkpointing: recompute each layer's intermediates \
during backward instead of holding them, ~33% slower, much less VRAM)] \
[--dequant-weights (dequantize EVERY packed weight to dense F32 at load, for \
--gguf and --tcf alike, so the forward pass is dense F32 end to end. A \
cross-format quality comparison is only valid when both artifacts run the \
same activation contract: a packed artifact quantizes activations before the \
matmul and a --ckpt run does not, so without this flag a packed-vs-dense gap \
is not weight-encoding damage. Costs what an unquantized checkpoint costs; a \
measurement mode, not a way to serve or fine-tune a quantized artifact)]";

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
    let mut tcf: Option<PathBuf> = None;
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
    let mut eval_only = false;
    let mut activation_checkpointing = false;
    let mut dequant_weights = false;

    let mut i = 0usize;
    while i < argv.len() {
        let flag = argv[i].as_str();
        match flag {
            "--ckpt" => ckpt = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--gguf" => gguf = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--tcf" => tcf = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
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
            "--eval-only" => eval_only = true,
            "--checkpoint" => activation_checkpointing = true,
            "--dequant-weights" => dequant_weights = true,
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

    // Exactly one weight source. Accepting two and silently preferring one
    // would load a different model than the operator asked for.
    let weights = match (ckpt, gguf, tcf) {
        (Some(dir), None, None) => Weights::Checkpoint(dir),
        (None, Some(path), None) => Weights::Gguf(path),
        (None, None, Some(path)) => Weights::Tcf(path),
        (None, None, None) => {
            return Err(format!("--ckpt, --gguf or --tcf is required\n{USAGE}"));
        }
        _ => {
            return Err(format!(
                "--ckpt, --gguf and --tcf are mutually exclusive\n{USAGE}"
            ));
        }
    };
    // A TCF has no metadata map to embed a config.json in, so the path is the
    // only way the architecture can be known — the same check
    // `voxcpm_clone`'s `--tcf` arm makes, made here before the file is mapped
    // and verified.
    if matches!(weights, Weights::Tcf(_)) && config.is_none() {
        return Err(format!("--config is required with --tcf\n{USAGE}"));
    }

    // An eval-only run writes nothing, so accepting `--out` would promise a
    // file that never appears.
    if eval_only && out.is_some() {
        return Err(format!(
            "--out belongs to a training run; --eval-only writes no checkpoint\n{USAGE}"
        ));
    }

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
        eval_only,
        activation_checkpointing,
        dequant_weights,
    })
}

/// Locate `tokenizer.json`.
///
/// A checkpoint directory holds it outright. Neither a GGUF nor a TCF carries
/// a tokenizer at all, so it is looked for beside the model file first and
/// beside `--config` second — both of those normally sit in, or are copied
/// from, the same checkpoint directory. Neither: an error, rather than a
/// tokenizer guess that would silently produce the wrong token ids. Same
/// resolution order `voxcpm_clone` uses.
fn tokenizer_path(weights: &Weights, config: Option<&Path>) -> Result<PathBuf, String> {
    match weights {
        Weights::Checkpoint(dir) => Ok(dir.join("tokenizer.json")),
        Weights::Gguf(path) | Weights::Tcf(path) => {
            let beside = |p: &Path| {
                p.parent()
                    .map(|dir| dir.join("tokenizer.json"))
                    .filter(|candidate| candidate.is_file())
            };
            beside(path)
                .or_else(|| config.and_then(beside))
                .ok_or_else(|| {
                    format!(
                        "no tokenizer.json beside {} (a single-file model carries none); \
                     put it there or pass --config pointing into the checkpoint \
                     directory",
                        path.display()
                    )
                })
        }
    }
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

/// Write one eval pass as a single JSON object on ONE line of STDOUT.
///
/// Every other line this file prints goes to stderr, so stdout carries the
/// metric lines and nothing else and two runs diff directly. `epoch` is the
/// 1-based epoch during training and `null` under `--eval-only`.
///
/// The record carries the row count and [`EVAL_NOISE_SEED`] alongside the
/// three metrics, so the pairing between two artifacts' numbers is auditable
/// from the output alone rather than assumed: two lines are comparable only
/// when `eval_rows`, `eval_seed`, `manifest`, `max_patches`, `lambda_diff`,
/// `lambda_stop` and `device` all match. Serialization is `serde_json`,
/// already a boostr dependency, and its float formatting is the shortest
/// round-trip form, so identical `f64` values print identically.
fn print_eval_record(
    args: &Args,
    epoch: Option<usize>,
    eval_rows: usize,
    eval_diff: f64,
    eval_stop: f64,
    eval_total: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let record = serde_json::json!({
        "record": "eval",
        "mode": if args.eval_only { "eval-only" } else { "train" },
        "epoch": epoch,
        "source_format": source_format(&args.weights),
        "model_path": source_path(&args.weights).display().to_string(),
        "config": args.config.as_ref().map(|p| p.display().to_string()),
        "audiovae": args.audiovae.display().to_string(),
        "manifest": args.manifest.display().to_string(),
        "device": match args.device {
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
        },
        // Which weight representation the forward pass ran. Two result
        // files that disagree here were scored under different activation
        // contracts and are NOT comparable — see the module docs'
        // "Comparing two artifacts" section.
        "weights_dense": args.dequant_weights,
        "eval_rows": eval_rows,
        "eval_seed": EVAL_NOISE_SEED,
        "max_patches": args.max_patches,
        "lambda_diff": LAMBDA_DIFF,
        "lambda_stop": args.lambda_stop,
        "eval_diff": eval_diff,
        "eval_stop": eval_stop,
        "eval_total": eval_total,
    });
    println!("{}", serde_json::to_string(&record)?);
    Ok(())
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
    //
    // `--dequant-weights` switches BOTH single-file arms to the `*_dense`
    // loaders instead, which materialize every packed weight to dense F32 at
    // load — the weight-encoding-only mode, see the module docs' "Comparing
    // two artifacts" section. `--ckpt` is dense F32 already, so the flag
    // changes nothing there and says so.
    //
    // MANDATORY, never drop this line: the mode has to be visible in the log
    // of every run, because two runs that differ only in it are NOT
    // comparable and nothing else in the output distinguishes them.
    if args.dequant_weights {
        eprintln!(
            "weights: DENSE F32 (--dequant-weights) — every packed weight is dequantized at              load, so both formats run the same dense F32 activation contract and differ only              in weight values; costs what an unquantized checkpoint costs"
        );
    } else {
        eprintln!(
            "weights: as stored (pass --dequant-weights to dequantize every packed weight to              dense F32 before scoring)"
        );
    }
    let mut model = match &args.weights {
        Weights::Checkpoint(dir) => {
            eprintln!("loading {} ...", dir.display());
            if args.dequant_weights {
                eprintln!(
                    "--dequant-weights: --ckpt already loads dense F32, nothing to dequantize"
                );
            }
            VoxCpm2Model::<R>::from_checkpoint(dir, &args.audiovae, device, Some(DType::F32))?
        }
        Weights::Gguf(path) if args.dequant_weights => {
            eprintln!("loading {} (dequantized to dense F32) ...", path.display());
            VoxCpm2Model::<R>::from_gguf_dense(
                path,
                args.config.as_deref(),
                &args.audiovae,
                device,
                client,
            )?
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
        Weights::Tcf(path) if args.dequant_weights => {
            eprintln!("loading {} (dequantized to dense F32) ...", path.display());
            // `parse_args` already rejected a missing `--config`; this arm
            // repeats the check rather than unwrapping on that invariant.
            let config = args
                .config
                .as_deref()
                .ok_or("--config is required with --tcf")?;
            VoxCpm2Model::<R>::from_tcf_dense(path, config, &args.audiovae, device, client)?
        }
        // Same loader `voxcpm_clone`'s `--tcf` arm calls, on the same
        // auxiliary inputs, with the `None` dtype the `--gguf` arm above
        // explains: a natively encoded projection stays PACKED, dense
        // tensors arrive F32, and `Some(BF16)`/`Some(F16)` would be rejected
        // by the loader anyway.
        Weights::Tcf(path) => {
            eprintln!("loading {} (base stays packed) ...", path.display());
            let config = args
                .config
                .as_deref()
                .ok_or("--config is required with --tcf")?;
            VoxCpm2Model::<R>::from_tcf(path, config, &args.audiovae, device, None)?
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
    // entirely while training: no split, no eval batch, no eval logging.
    //
    // `--eval-only` has no training half to protect, so the two guards that
    // exist for training's sake are lifted there: `--eval-rows 0` means EVERY
    // kept row is scored (the manifest IS the held-out set), and a count that
    // would leave zero training rows is fine. A nonzero `--eval-rows N` still
    // takes the same last N rows either way, so a training run and an
    // eval-only run over one manifest score the same rows in the same order.
    let (rows, eval_source_rows): (&[&ManifestRow], &[&ManifestRow]) = if args.eval_only {
        if args.eval_rows == 0 || args.eval_rows >= rows.len() {
            (&[], rows.as_slice())
        } else {
            rows.split_at(rows.len() - args.eval_rows)
        }
    } else {
        if args.eval_rows >= rows.len() {
            return Err(format!(
                "--eval-rows {} >= {} kept manifest row(s); that would leave zero training rows",
                args.eval_rows,
                rows.len()
            )
            .into());
        }
        rows.split_at(rows.len() - args.eval_rows)
    };
    if args.eval_only {
        eprintln!(
            "eval-only: {} eval row(s) (--eval-rows {}{})",
            eval_source_rows.len(),
            args.eval_rows,
            if args.eval_rows == 0 {
                " = every kept row"
            } else {
                ""
            }
        );
    } else {
        eprintln!(
            "eval split: {} training row(s), {} eval row(s) (--eval-rows {})",
            rows.len(),
            eval_source_rows.len(),
            args.eval_rows
        );
    }

    let tokenizer = load_tokenizer(tokenizer_path(&args.weights, args.config.as_deref())?)?;

    // `--eval-only` returns HERE, before `apply_lora` — the artifact is
    // scored exactly as it sits on disk, with no adapters allocated, no
    // optimizer built, no backward pass and no checkpoint written. See the
    // module docs' "`--eval-only`" section.
    if args.eval_only {
        if eval_source_rows.is_empty() {
            return Err("--eval-only: no eval rows survived the --max-patches filter".into());
        }
        eprintln!(
            "building eval batch ({} row(s)) ...",
            eval_source_rows.len()
        );
        let eval_batch = build_eval_batch(
            &model,
            client,
            &tokenizer,
            eval_source_rows,
            args.max_patches,
        )?;
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
            "eval/diff {eval_diff:.6} eval/stop {eval_stop:.6} eval/total {eval_total:.6} \
             ({} eval row(s))",
            eval_batch.len()
        );
        print_eval_record(
            args,
            None,
            eval_batch.len(),
            eval_diff,
            eval_stop,
            eval_total,
        )?;
        eprintln!("total {:.1}s", started.elapsed().as_secs_f64());
        return Ok(());
    }

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
            print_eval_record(
                args,
                Some(epoch),
                eval_batch.len(),
                eval_diff,
                eval_stop,
                eval_total,
            )?;
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
