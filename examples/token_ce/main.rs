//! Deterministic token cross-entropy for a causal LM, over a fixed set of
//! text windows, teacher forced.
//!
//! ```text
//! cargo run --release --example token_ce -- \
//!     (--ckpt CKPT_DIR | --gguf MODEL.gguf | --tcf MODEL.tcf) [--config config.json] \
//!     --text FILE.txt [--tokenizer tokenizer.json] [--device cpu|cuda] \
//!     [--seq-len 512] [--windows 64] [--stride N] [--dequant-weights]
//! ```
//!
//! # What this answers
//!
//! What a weight ENCODING costs a causal language model, in the metric that
//! model is trained on.
//!
//! A quality claim about a 4-bit format measured on one model with one metric
//! is a claim about that model. Reproducing it needs a second architecture and
//! a second metric, and this example is the second metric: mean token
//! cross-entropy, in natural log, over a fixed corpus slice. Two runs that
//! differ only in `--gguf` versus `--tcf` produce two numbers whose difference
//! is the encoding's cost.
//!
//! # Comparing two artifacts
//!
//! Pass `--dequant-weights` on BOTH runs.
//!
//! `CONFORMANCE.md` Section 7.1 requires the ACTIVATION CONTRACT to be matched
//! across two artifacts being compared, and it is not matched by default: a
//! TCF declares an exact F32 contract, so its matmul runs F32 activations,
//! while a GGUF declares none and its feature-major MMQ path quantizes the
//! activations to int8 before the tensor-core MMA. The GGUF side then absorbs
//! activation-quantization error the TCF side never pays, and the gap reads as
//! a weight-format difference that it is not.
//!
//! `--dequant-weights` removes that confound: every packed weight becomes a
//! dense F32 tensor at load, both artifacts run the same dense F32 matmul, and
//! the only thing left that differs is the weight VALUES. It costs what an
//! unquantized checkpoint costs in memory — that is the price of the
//! controlled comparison, and it is a MEASUREMENT mode, never a serving mode.
//!
//! The mode is reported in the output as `weights_dense`. Two records that
//! disagree on it were scored under different activation contracts and must
//! not be compared.
//!
//! # The metric
//!
//! Mean cross-entropy per token, in nats:
//!
//! - The corpus is tokenized ONCE, whole, in file order, with no special
//!   tokens added.
//! - Window `i` is the `--seq-len + 1` tokens starting at `i * --stride`.
//!   `--stride` defaults to `--seq-len`, which walks the corpus from the start
//!   without overlap.
//! - Each window feeds its first `--seq-len` ids to the model in one forward
//!   pass and scores the prediction of ids `1..=--seq-len`, so every window
//!   contributes exactly `--seq-len` scored tokens.
//! - The reported `mean_cross_entropy_nats` is the token-weighted mean over
//!   every window, and `scored_tokens` is what it averaged over.
//!
//! `perplexity` is `exp(mean_cross_entropy_nats)` and is emitted for reading
//! convenience only. The MEAN is the comparable number: perplexity is a
//! monotone re-scaling of it that exaggerates differences at the top of the
//! range and compresses them at the bottom.
//!
//! # What "deterministic" means here, exactly
//!
//! Two runs of the same command line over the same artifact, on the same build
//! and device, print byte-identical metrics. Pinned:
//!
//! - Window membership and order: window `i` starts at `i * stride`, taken from
//!   the start of the corpus. No sampling, no shuffling, no RNG anywhere in the
//!   selection.
//! - The corpus slice: `--seq-len` and `--windows` are both flags and both
//!   appear in the output, so a record carries the definition of its own
//!   sample.
//! - Tokenization: one `tokenizer.json`, `encode_raw`, no BOS and no other
//!   special token inserted.
//! - The forward pass: one prefill per window, batch size 1, causal mask, no
//!   KV cache, no decode loop, no sampling. There is no dropout module
//!   anywhere in this model's forward path and nothing in it draws a random
//!   number.
//! - The mean: `sum(mean_w * seq_len) / scored_tokens` accumulated in `f64`
//!   over windows in that same fixed order, so the floating-point summation
//!   order is fixed too.
//!
//! What breaks it: a different `--seq-len`, `--windows` or `--stride` (all
//! change which tokens are scored), an edited `--text`, a different
//! tokenizer, and a different `--dequant-weights`.
//!
//! NOT established as deterministic, and deliberately not claimed — the same
//! reservation `voxcpm_finetune` makes:
//!
//! - ACROSS devices. `--device cpu` and `--device cuda` run different kernels;
//!   nothing here checks that they agree bit for bit, and a comparison must
//!   hold the device fixed.
//! - ACROSS builds. Feature flags select different kernels, and a
//!   multi-threaded reduction inside a kernel is free to reorder its partial
//!   sums between runs. This file does not audit numr's kernels for
//!   thread-order-independent reductions, so run-to-run bit-identity is an
//!   empirical property of the backend, not a guarantee this file can make.
//!   Run the same artifact twice and diff the stdout lines before trusting a
//!   small difference between two artifacts.
//! - ACROSS artifacts of different numeric encodings. That difference is the
//!   thing being MEASURED; only the sampling is pinned, so a nonzero delta
//!   between two formats is signal, not noise.
//!
//! # Cost
//!
//! One prefill of `--seq-len` tokens per window, `--windows` times, and one
//! model load. Nothing is cached between windows and no window is revisited,
//! so the compute is linear in `--windows` and the peak memory is one model
//! plus one window's activations and one `[1, seq_len, vocab]` logit tensor.
//! `--dequant-weights` raises the model's resident size to what its
//! unquantized form would cost.
//!
//! # Output
//!
//! ONE JSON object on stdout, and nothing else on stdout, so two runs diff
//! directly. Progress goes to stderr.

use std::path::PathBuf;
use std::time::Instant;

use boostr::model::llama::Llama;
use boostr::model::traits::{Model, ModelClient};
use boostr::nn::VarBuilder;
use boostr::quant::ImportanceMatrix;
use boostr::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
#[cfg(feature = "cuda")]
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use tcf_core::NativeEncoding;

// Which artifact the weights come from, and how one open artifact becomes a
// `VarMap`. Lives in `examples/shared/` because the `imatrix` example loads
// the same three artifact forms the same way; the three formats' resolution
// rules are one concern and the measurement below is another.
#[path = "../shared/source.rs"]
mod source;
use source::{Weights, config_path, load_config, load_varmap, source_format, source_path};

// The corpus, the window selection, and the scored loss. Shared with the
// `imatrix` example so both binaries select the SAME windows from the same
// `--text`: an importance matrix collected over one corpus slice and an
// evaluation measuring its effect over a different one are not comparable.
#[path = "../shared/windows.rs"]
mod windows;
use windows::{score_windows, select_windows, tokenize_corpus};

// The tensor loop shared by every dense-weight probe below: cast-to-F32,
// candidate selection, importance-entry gating, accounting, write-back.
mod probe;

// The AWQ-style per-input-channel smoothing PROBE: `--smooth-encoding` and
// friends transform the loaded `VarMap`'s dense weights before the model is
// built, so the rest of this file's scoring path runs unchanged. See the
// module's own docs for what "probe" means here — TCF stores no scale plane,
// so this scores the dense weight the format WOULD reconstruct, exactly.
mod smooth;
use smooth::{
    SmoothObjective, SmoothSource, apply_smoothing, parse_native_encoding, parse_smooth_objective,
    parse_smooth_source,
};

// The codebook PROBE: `--codebook` transforms the loaded `VarMap`'s dense
// weights through a self-contained 4-bit block quantizer, at identical
// geometry to the encodings already measured, isolating whether a
// non-uniform reconstruction codebook reduces task damage. Mutually
// exclusive with `--smooth-encoding` — see `parse_args`.
mod codebook;
use codebook::{CodebookObjective, apply_codebook, parse_codebook, parse_codebook_objective};

/// Tokens per scored window. Long enough that most positions are predicted
/// with real context, short enough that one `[1, seq_len, vocab]` logit tensor
/// stays modest on a 100k-wide vocabulary.
const DEFAULT_SEQ_LEN: usize = 512;
/// Windows scored per run. This is the cost dial: the forward work is linear
/// in it, and so is the token count the mean averages over.
const DEFAULT_WINDOWS: usize = 64;

/// Runtime to load the model and score on.
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
    /// Architecture config. Optional for `--ckpt`, which falls back to the
    /// `config.json` in the checkpoint directory; required for `--gguf` and
    /// `--tcf`, which carry none.
    config: Option<PathBuf>,
    /// `tokenizer.json`. Optional everywhere: a checkpoint directory holds
    /// one, and for a single-file artifact it is looked for beside the file
    /// and beside `--config`.
    tokenizer: Option<PathBuf>,
    text: PathBuf,
    device: Device,
    seq_len: usize,
    windows: usize,
    /// Distance between two window starts, in tokens. Defaults to `seq_len`,
    /// which walks the corpus from the start without overlap.
    stride: Option<usize>,
    /// Materialize every packed weight to dense F32 at load, so two artifacts
    /// run the same activation contract and differ only in weight VALUES —
    /// see the module docs' "Comparing two artifacts" section. OFF by
    /// default.
    dequant_weights: bool,
    /// The AWQ-style smoothing probe. `Some` turns it on; see `smooth.rs`.
    /// Mutually exclusive with `codebook`.
    smoothing: Option<SmoothingArgs>,
    /// The codebook probe. `Some` turns it on; see `codebook.rs`. Mutually
    /// exclusive with `smoothing`.
    codebook: Option<CodebookArgs>,
}

/// `--smooth-*` flags, gathered once presence is confirmed by
/// `--smooth-encoding` being set.
struct SmoothingArgs {
    encoding: NativeEncoding,
    alpha: f32,
    imatrix: PathBuf,
    objective: SmoothObjective,
    source: SmoothSource,
}

/// `--codebook*` flags, gathered once presence is confirmed by `--codebook`
/// being set.
struct CodebookArgs {
    codebook: boostr::quant::Codebook,
    imatrix: PathBuf,
    objective: CodebookObjective,
}

const USAGE: &str = "usage: token_ce (--ckpt DIR | --gguf MODEL.gguf | --tcf MODEL.tcf) \
[--config config.json (required with --gguf and --tcf, which carry no architecture config)] \
--text FILE.txt (the corpus the scored windows are taken from, in file order) \
[--tokenizer tokenizer.json (default: the checkpoint directory's, or the one beside a \
single-file artifact)] \
[--device cpu|cuda] \
[--seq-len 512 (tokens per scored window)] \
[--windows 64 (windows scored; the cost dial)] \
[--stride N (tokens between two window starts; default --seq-len, i.e. no overlap)] \
[--dequant-weights (dequantize EVERY packed weight to dense F32 at load, so both \
artifacts of a comparison run the same dense F32 activation contract and differ only in \
weight values; costs what an unquantized checkpoint costs, and is required for a valid \
cross-format comparison)] \
[--smooth-encoding ENCODING (turns on the AWQ-style per-input-channel smoothing PROBE; a \
TCF NativeEncoding name, e.g. Q4AS32DT64; requires --ckpt and --smooth-imatrix)] \
[--smooth-alpha 0.5 (in [0, 1]; trades activation RMS against weight magnitude in the \
smoothing scale; 0 is the exact no-op control)] \
[--smooth-imatrix FILE.bstrimtx (required with --smooth-encoding: the activation \
importance the smoothing scale is derived from)] \
[--smooth-objective uniform|imatrix (default imatrix: the error objective the probe's \
quantize call scores against, letting a run separate the smoothing effect from the \
objective effect)] \
[--smooth-source activation|weight (default activation: activation is today's behaviour, \
requiring --smooth-imatrix's RMS activation AND the weight; weight is calibration-free, \
derived from the weight's own column magnitudes alone — --smooth-imatrix is still required \
and still selects which tensors are transformed, for a like-for-like tensor set between the \
two sources)] \
[--codebook uniform|nf4 (turns on the codebook PROBE: quantize/dequantize every candidate \
weight through a self-contained 4-bit block quantizer at fixed geometry, comparing an evenly \
spaced 16-level grid against NF4's non-uniform one; requires --ckpt and --smooth-imatrix; \
mutually exclusive with --smooth-encoding)] \
[--codebook-objective uniform|imatrix (default imatrix: the per-element weight the codebook's \
group scale search scores against, exactly like --smooth-objective)]";

/// Consume the value that follows `flag`, advancing `i` past it.
fn take_value(argv: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    *i += 1;
    argv.get(*i)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut ckpt: Option<PathBuf> = None;
    let mut gguf: Option<PathBuf> = None;
    let mut tcf: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut text: Option<PathBuf> = None;
    let mut device = Device::Cpu;
    let mut seq_len = DEFAULT_SEQ_LEN;
    let mut windows = DEFAULT_WINDOWS;
    let mut stride: Option<usize> = None;
    let mut dequant_weights = false;
    let mut smooth_encoding: Option<NativeEncoding> = None;
    let mut smooth_alpha = 0.5f32;
    let mut smooth_imatrix: Option<PathBuf> = None;
    let mut smooth_objective = SmoothObjective::Imatrix;
    let mut smooth_source = SmoothSource::Activation;
    let mut codebook_choice: Option<boostr::quant::Codebook> = None;
    let mut codebook_objective = CodebookObjective::Imatrix;

    let mut i = 0usize;
    while i < argv.len() {
        let flag = argv[i].as_str();
        match flag {
            "--ckpt" => ckpt = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--gguf" => gguf = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--tcf" => tcf = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--config" => config = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--tokenizer" => tokenizer = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--text" => text = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
            "--device" => device = parse_device(&take_value(&argv, &mut i, flag)?)?,
            "--seq-len" => {
                seq_len = take_value(&argv, &mut i, flag)?
                    .parse()
                    .map_err(|e| format!("--seq-len: {e}"))?
            }
            "--windows" => {
                windows = take_value(&argv, &mut i, flag)?
                    .parse()
                    .map_err(|e| format!("--windows: {e}"))?
            }
            "--stride" => {
                stride = Some(
                    take_value(&argv, &mut i, flag)?
                        .parse()
                        .map_err(|e| format!("--stride: {e}"))?,
                )
            }
            "--dequant-weights" => dequant_weights = true,
            "--smooth-encoding" => {
                smooth_encoding = Some(parse_native_encoding(&take_value(&argv, &mut i, flag)?)?)
            }
            "--smooth-alpha" => {
                smooth_alpha = take_value(&argv, &mut i, flag)?
                    .parse()
                    .map_err(|e| format!("--smooth-alpha: {e}"))?
            }
            "--smooth-imatrix" => {
                smooth_imatrix = Some(PathBuf::from(take_value(&argv, &mut i, flag)?))
            }
            "--smooth-objective" => {
                smooth_objective = parse_smooth_objective(&take_value(&argv, &mut i, flag)?)?
            }
            "--smooth-source" => {
                smooth_source = parse_smooth_source(&take_value(&argv, &mut i, flag)?)?
            }
            "--codebook" => {
                codebook_choice = Some(parse_codebook(&take_value(&argv, &mut i, flag)?)?)
            }
            "--codebook-objective" => {
                codebook_objective = parse_codebook_objective(&take_value(&argv, &mut i, flag)?)?
            }
            "-h" | "--help" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown flag {other}\n{USAGE}")),
        }
        i += 1;
    }

    if seq_len == 0 {
        return Err("--seq-len must be at least 1".to_string());
    }
    if windows == 0 {
        return Err("--windows must be at least 1".to_string());
    }
    if stride == Some(0) {
        return Err(
            "--stride must be at least 1: stride 0 scores one window repeatedly".to_string(),
        );
    }

    // Exactly one weight source. Accepting two and silently preferring one
    // would score a different artifact than the operator asked for.
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

    // Exactly one probe is OFF unless one of `--smooth-encoding` or
    // `--codebook` names it, and the two are mutually exclusive: each
    // rewrites the same dense `VarMap` in place, so running both would
    // score neither transform in isolation.
    let (smoothing, codebook) = match (smooth_encoding, codebook_choice) {
        (Some(_), Some(_)) => {
            return Err(format!(
                "--smooth-encoding and --codebook are mutually exclusive\n{USAGE}"
            ));
        }
        (None, None) => {
            if smooth_imatrix.is_some() {
                return Err(format!(
                    "--smooth-imatrix has no effect without --smooth-encoding or \
                     --codebook\n{USAGE}"
                ));
            }
            (None, None)
        }
        (Some(encoding), None) => {
            // A packed weight has no dense `Var` to transform — same reason
            // `examples/voxcpm/sensitivity.rs` refuses `--gguf`/`--tcf`.
            if !matches!(weights, Weights::Checkpoint(_)) {
                return Err(format!(
                    "--smooth-encoding requires --ckpt: a GGUF or TCF weight is already \
                     packed, has no dense Var to transform, and the probe would silently \
                     no-op on it\n{USAGE}"
                ));
            }
            let imatrix = smooth_imatrix.ok_or_else(|| {
                format!(
                    "--smooth-encoding requires --smooth-imatrix: without it there is no \
                     activation importance to derive the smoothing scale from\n{USAGE}"
                )
            })?;
            if !(0.0..=1.0).contains(&smooth_alpha) {
                return Err(format!(
                    "--smooth-alpha must be in [0, 1], got {smooth_alpha}\n{USAGE}"
                ));
            }
            let smoothing = SmoothingArgs {
                encoding,
                alpha: smooth_alpha,
                imatrix,
                objective: smooth_objective,
                source: smooth_source,
            };
            (Some(smoothing), None)
        }
        (None, Some(chosen)) => {
            // Same reasoning as `--smooth-encoding` above: no dense `Var` to
            // transform behind `--gguf`/`--tcf`.
            if !matches!(weights, Weights::Checkpoint(_)) {
                return Err(format!(
                    "--codebook requires --ckpt: a GGUF or TCF weight is already packed, has \
                     no dense Var to transform, and the probe would silently no-op on it\n{USAGE}"
                ));
            }
            let imatrix = smooth_imatrix.ok_or_else(|| {
                format!(
                    "--codebook requires --smooth-imatrix: without it there is no \
                     importance-entry-gated candidate set to transform\n{USAGE}"
                )
            })?;
            let codebook = CodebookArgs {
                codebook: chosen,
                imatrix,
                objective: codebook_objective,
            };
            (None, Some(codebook))
        }
    };

    Ok(Args {
        weights,
        config,
        tokenizer,
        text: text.ok_or_else(|| format!("--text is required\n{USAGE}"))?,
        device,
        seq_len,
        windows,
        stride,
        dequant_weights,
        smoothing,
        codebook,
    })
}

/// The model-and-scoring body: everything that runs on the chosen runtime `R`.
fn run<R, C>(args: &Args, device: &R::Device, client: &C) -> Result<(), Box<dyn std::error::Error>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R> + DequantOps<R> + TypeConversionOps<R>,
    R::Client: ModelClient<R> + DequantOps<R> + TypeConversionOps<R>,
{
    let started = Instant::now();
    let stride = args.stride.unwrap_or(args.seq_len);

    // MANDATORY, never drop this line: the mode has to be visible in the log
    // of every run, because two runs that differ only in it are NOT
    // comparable and nothing else on stderr distinguishes them.
    if args.dequant_weights {
        eprintln!(
            "weights: DENSE F32 (--dequant-weights) — every packed weight is dequantized at \
             load, so both formats run the same dense F32 activation contract and differ only \
             in weight values; costs what an unquantized checkpoint costs"
        );
    } else {
        eprintln!(
            "weights: as stored — pass --dequant-weights before comparing two formats, or the \
             measurement includes their activation contracts as well as their weights"
        );
    }
    if args.dequant_weights && matches!(args.weights, Weights::Checkpoint(_)) {
        eprintln!("--dequant-weights: --ckpt stores no packed weight, nothing to dequantize");
    }

    let config_file = config_path(&args.weights, args.config.as_deref())?;
    let config = load_config(&config_file)?;
    if config.attention.is_none() {
        return Err(format!(
            "{}: this config declares no attention block, so it is not a causal transformer \
             this example can score",
            config_file.display()
        )
        .into());
    }

    let tokenizer_path = source::tokenizer_path(
        &args.weights,
        args.tokenizer.as_deref(),
        args.config.as_deref(),
    )?;
    eprintln!("tokenizer: {}", tokenizer_path.display());
    let tokenizer = splintr::from_json_path(&tokenizer_path)
        .map_err(|e| format!("{}: {e}", tokenizer_path.display()))?;

    let text = std::fs::read_to_string(&args.text)
        .map_err(|e| format!("{}: failed to read corpus: {e}", args.text.display()))?;
    let tokens = tokenize_corpus(&tokenizer, &text);
    eprintln!(
        "corpus: {} token(s) from {}",
        tokens.len(),
        args.text.display()
    );
    let selected = select_windows(&tokens, args.seq_len, args.windows, stride)?;

    eprintln!("loading {} ...", source_path(&args.weights).display());
    let mut var_map = load_varmap::<R, C>(&args.weights, args.dequant_weights, device, client)?;

    if let Some(smoothing) = &args.smoothing {
        eprintln!(
            "smoothing: loading importance matrix {} ...",
            smoothing.imatrix.display()
        );
        let imatrix = ImportanceMatrix::read_from_path(&smoothing.imatrix)?;
        let summary = apply_smoothing::<R>(
            &mut var_map,
            &imatrix,
            smoothing.encoding,
            smoothing.alpha,
            smoothing.objective,
            smoothing.source,
        )?;
        // MANDATORY, never drop this line: a probe that silently skipped
        // most of the model would look like a null result. `apply_smoothing`
        // already refuses to return at all when `transformed == 0`, so
        // reaching this line means the totals below reconcile:
        // `examined == candidates + non_candidates` and
        // `candidates == transformed + skipped_no_entry`.
        eprintln!(
            "smoothing: encoding={:?} alpha={} source={:?} objective={:?} examined={} \
             cast_to_f32={} candidates={} non_candidates={} transformed={} skipped_no_entry={}",
            smoothing.encoding,
            smoothing.alpha,
            smoothing.source,
            smoothing.objective,
            summary.examined,
            summary.cast_to_f32,
            summary.candidates,
            summary.non_candidates,
            summary.transformed,
            summary.skipped_no_entry
        );
    }

    if let Some(codebook) = &args.codebook {
        eprintln!(
            "codebook: loading importance matrix {} ...",
            codebook.imatrix.display()
        );
        let imatrix = ImportanceMatrix::read_from_path(&codebook.imatrix)?;
        let summary = apply_codebook::<R>(
            &mut var_map,
            &imatrix,
            codebook.codebook,
            codebook.objective,
        )?;
        // MANDATORY, never drop this line: same reasoning as the smoothing
        // probe's summary line above — `apply_codebook` already refuses to
        // return at all when `transformed == 0`.
        eprintln!(
            "codebook: codebook={:?} objective={:?} examined={} cast_to_f32={} candidates={} \
             non_candidates={} transformed={} skipped_no_entry={}",
            codebook.codebook,
            codebook.objective,
            summary.examined,
            summary.cast_to_f32,
            summary.candidates,
            summary.non_candidates,
            summary.transformed,
            summary.skipped_no_entry
        );
    }

    let mut vb = VarBuilder::new(&mut var_map, device);
    let model = Llama::<R>::from_varbuilder(&mut vb, &config)?;
    eprintln!(
        "model: {} layer(s), vocab {}, loaded in {:?}",
        config.num_layers,
        config.vocab_size,
        started.elapsed()
    );

    let (mean, scored) = score_windows::<R, C>(&model, client, device, &selected)?;

    let record = serde_json::json!({
        "record": "token_ce",
        "source_format": source_format(&args.weights),
        "model_path": source_path(&args.weights).display().to_string(),
        "config": config_file.display().to_string(),
        "tokenizer": tokenizer_path.display().to_string(),
        "text": args.text.display().to_string(),
        "device": match args.device {
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
        },
        // Which weight representation the forward pass ran. Two records that
        // disagree here were scored under different activation contracts and
        // are NOT comparable — see the module docs' "Comparing two artifacts"
        // section.
        "weights_dense": args.dequant_weights,
        "seq_len": args.seq_len,
        "windows": selected.len(),
        "stride": stride,
        "corpus_tokens": tokens.len(),
        "scored_tokens": scored,
        "vocab_size": config.vocab_size,
        "num_layers": config.num_layers,
        "mean_cross_entropy_nats": mean,
        "perplexity": mean.exp(),
    });
    // `serde_json`'s float formatting is the shortest round-trip form, so two
    // identical `f64` values print identically and two records diff cleanly.
    println!("{}", serde_json::to_string(&record)?);
    eprintln!(
        "mean cross-entropy {mean:.6} nats over {scored} token(s); total {:?}",
        started.elapsed()
    );
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

    match args.device {
        Device::Cpu => {
            let device = CpuDevice::default();
            let client = CpuClient::new(device.clone());
            run::<CpuRuntime, CpuClient>(&args, &device, &client)?;
        }
        #[cfg(feature = "cuda")]
        Device::Cuda => {
            let device = CudaDevice::new(0);
            let client = CudaClient::new(device.clone())?;
            run::<CudaRuntime, CudaClient>(&args, &device, &client)?;
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda => {
            return Err(
                "--device cuda: this binary was built without CUDA support; rebuild with \
                 --features cuda"
                    .into(),
            );
        }
    }
    Ok(())
}
