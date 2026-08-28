//! End-to-end LoRA fine-tune of VoxCPM2 on real `(text, wav)` pairs.
//!
//! ```text
//! cargo run --release --features audio,f16 --example voxcpm_finetune -- \
//!     (--ckpt CKPT_DIR | --gguf MODEL.gguf [--config config.json]) \
//!     --audiovae audiovae.safetensors --manifest FILE.tsv \
//!     [--device cpu|cuda] [--targets q_proj,v_proj] [--rank 16] [--alpha 32] \
//!     [--lr 1e-4] [--epochs 3] [--seed 0] [--out adapters.safetensors] \
//!     [--lambda-stop 1.0] [--training-cfg-rate 0.1]
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
//! [`SequenceLayout::build`](boostr::model::audio::voxcpm::model::sequence::SequenceLayout::build)
//! rejects `t_ref == 0`, so `prefill`/`prefill_capturing` have NO supported
//! no-reference form today — every row this binary actually trains on must
//! carry a `ref_wav` until that gap is closed upstream. A row without one
//! fails loudly, naming the row, rather than silently falling back to
//! self-referencing `wav`.
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
//! `--out` writes ONLY the adapter tensors (`named_parameters()` entries
//! ending `lora_a`/`lora_b`), named by their full checkpoint-style path, via
//! this crate's own [`boostr::format::safetensors::save_safetensors`]
//! writer — never a hand-rolled one. That writer accepts CPU tensors only,
//! so each adapter tensor is round-tripped through `to_bytes`/`from_bytes`
//! (the same device-to-host pattern `trainer::async_checkpoint` uses),
//! which works whether the run trained on CPU or CUDA.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use boostr::format::safetensors::save_safetensors;
use boostr::model::audio::voxcpm::model::VoxCpm2Model;
use boostr::model::audio::voxcpm::model::config::AUDIO_START_ID;
use boostr::model::audio::voxcpm::{VoxCpmClient, load_tokenizer, normalize_whitespace, tokenize};
use boostr::model::audio::{decode_audio, extension_hint, to_mono_at_rate};
use boostr::nn::{LoraTargets, Module};
use boostr::ops::FusedOptimizerOps;
use boostr::quant::traits::DequantOps;
use boostr::trainer::{SimpleTrainer, TrainingConfig};
use numr::autograd::backward;
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
}

const USAGE: &str = "usage: voxcpm_finetune (--ckpt DIR | --gguf MODEL.gguf [--config config.json]) \
--audiovae audiovae.safetensors \
--manifest FILE.tsv (header-named TSV: wav, text, optional ref_wav) \
[--device cpu|cuda] [--targets q_proj,v_proj] [--rank 16] \
[--alpha 32] [--lr 1e-4] [--epochs 3] [--seed 0] [--out adapters.safetensors] \
[--lambda-stop 1.0] [--training-cfg-rate 0.1 (DO NOT set to 0 — upstream's FAQ \
names text-ignoring as the most common fine-tuning failure mode)]";

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

/// The model-and-training body: everything that runs on the chosen runtime
/// `R`. Loads the checkpoint, adapts it with LoRA, then trains one epoch at
/// a time over every manifest row, printing per-step and per-epoch loss.
fn run<R: Runtime<DType = DType>>(
    args: &Args,
    device: &R::Device,
    client: &(impl VoxCpmClient<R> + TypeConversionOps<R> + RandomOps<R> + FusedOptimizerOps<R>),
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

    let config = TrainingConfig::default().with_lr(args.lr);
    let mut trainer = SimpleTrainer::<R>::new(config)?;

    eprintln!(
        "manifest: {} row(s), {} epoch(s), lr={}, lambda_diff={LAMBDA_DIFF}, lambda_stop={}",
        rows.len(),
        args.epochs,
        args.lr,
        args.lambda_stop
    );

    let mut step_counter: u64 = 0;
    for epoch in 1..=args.epochs {
        let mut epoch_diff_sum = 0.0f64;
        let mut epoch_stop_sum = 0.0f64;
        let mut epoch_steps = 0usize;

        for (row_index, row) in rows.iter().enumerate() {
            let wav = load_wav_16k(&row.wav).map_err(|e| format!("{}: {e}", row.wav.display()))?;
            // The training target: what the loss is computed against.
            let target_patches = model.encode_reference(client, &wav)?;

            // The reference-conditioning clip MUST be a different clip than
            // `wav` — see the module docs for why self-referencing is
            // degenerate. `prefill`/`prefill_capturing` have no supported
            // no-reference form (`SequenceLayout::build` rejects `t_ref ==
            // 0`), so a row missing `ref_wav` is a hard error naming the
            // row, never a silent fallback to `target_patches`.
            let ref_wav_path = row.ref_wav.as_ref().ok_or_else(|| {
                format!(
                    "{} (row {row_index}): no ref_wav column value, and \
                     VoxCpm2Model::prefill has no supported no-reference form; add a \
                     ref_wav naming a DIFFERENT clip from the same speaker",
                    row.wav.display()
                )
            })?;
            let ref_wav = load_wav_16k(ref_wav_path)
                .map_err(|e| format!("{}: {e}", ref_wav_path.display()))?;
            let ref_patches = model.encode_reference(client, &ref_wav)?;
            let t_ref = ref_patches.shape()[0];

            let normalized = normalize_whitespace(&row.text);
            let mut text_token_ids = tokenize(&tokenizer, &normalized);
            // `prefill` requires the sequence to end here: AUDIO_START_ID is
            // the position the first (only, here) patch attends from.
            text_token_ids.push(AUDIO_START_ID);
            let max_length = t_ref + 2 + text_token_ids.len();

            // ALWAYS `prefill_capturing`: `cfm_loss`'s teacher-forced path
            // needs `PrefillState::intermediates`, and every row here has a
            // non-empty prefix — see the module docs.
            let prefill =
                model.prefill_capturing(client, &ref_patches, &text_token_ids, max_length)?;

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

            let grads = backward(&losses.total, client)?;
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
    }

    if let Some(out) = &args.out {
        let adapters = collect_adapter_tensors(&model)?;
        eprintln!(
            "saving {} adapter tensor(s) to {} ...",
            adapters.len(),
            out.display()
        );
        save_safetensors(out, &adapters, None)?;
        eprintln!("wrote {}", out.display());
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
