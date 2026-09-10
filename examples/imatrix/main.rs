//! Collect an IMPORTANCE MATRIX for a causal LM over a fixed set of text
//! windows, and write it to one file a quantizer can read.
//!
//! ```text
//! cargo run --release --example imatrix -- \
//!     (--ckpt CKPT_DIR | --gguf MODEL.gguf | --tcf MODEL.tcf) [--config config.json] \
//!     --text FILE.txt --out IMPORTANCE.bstrimtx [--tokenizer tokenizer.json] \
//!     [--device cpu|cuda] [--seq-len 512] [--windows 64] [--stride N] \
//!     [--dequant-weights]
//! ```
//!
//! # What this measures
//!
//! For a linear weight `W` of shape `[out_features, in_features]`, column `j`
//! multiplies input feature `x_j`. How much that column MATTERS to the output
//! is how large `x_j` typically is, so the statistic is `sum(x_j^2)` over
//! every token the calibration corpus put through that layer. A quantizer
//! spends its error budget on the columns with the largest values.
//!
//! One vector per weight, `in_features` long. The file also carries the row
//! count each vector was summed over and the run's total token count, so a
//! consumer normalizes it however it likes and can tell a 10-token collection
//! from a 100k-token one.
//!
//! # Where the numbers come from
//!
//! `boostr::quant::imatrix` taps `MaybeQuantLinear::forward`'s DENSE arm.
//! Nothing about the model changes: an ordinary run reads one relaxed atomic
//! bool and takes a branch it never takes. Every weight loaded through
//! `VarBuilder::take_maybe_quant_linear` is bound to the checkpoint key it was
//! read from at LOAD time, so an entry's name is the exact string the
//! quantizer sees. Nothing is inferred from the model's structure.
//!
//! # What is NOT tapped, and why it is absent rather than zero
//!
//! - A weight still PACKED at load — a GGUF K-quant, a TCF native encoding —
//!   runs a quantized kernel and has no dense weight to guide. Pass
//!   `--dequant-weights` to materialize such an artifact to dense F32 first;
//!   without it a packed artifact measures nothing at all.
//! - Stacked MoE expert weights, which are one tensor multiplied by a grouped
//!   GEMM rather than a `Linear` per expert.
//! - Embedding lookups, which are a gather, not a matmul.
//!
//! Anything not measured has NO entry in the file. It is never written as a
//! zero vector: zero importance and "never measured" mean opposite things to
//! a quantizer, and only one of them is safe to act on. The unmeasured names
//! this run knows about are listed on stderr and counted in the stdout record.
//!
//! # Window selection
//!
//! Identical to `token_ce`'s, because it is the same code — `examples/shared/
//! windows.rs`, which both binaries compile. The corpus is tokenized once,
//! whole, in file order, with no special token added; window `i` is the
//! `--seq-len + 1` tokens starting at `i * --stride`; `--stride` defaults to
//! `--seq-len`, which walks the corpus from the start without overlap. There
//! is no sampling and no RNG anywhere in the selection. This pass feeds each
//! window's first `--seq-len` ids to the model — exactly the positions
//! `token_ce`'s loss depends on — so an importance matrix and an evaluation
//! of its effect are computed over the same text.
//!
//! # Determinism
//!
//! Pinned by this file and the code it shares with `token_ce`:
//!
//! - Window membership and order: window `i` starts at `i * stride`, taken
//!   from the start of the corpus.
//! - Tokenization: one `tokenizer.json`, `encode_raw`, no BOS.
//! - The forward pass: one prefill per window, batch size 1, causal mask, no
//!   KV cache, no decode loop, no sampling.
//! - Accumulation ORDER: windows are accumulated in that same fixed order,
//!   and each window's contribution is added into a device tensor as it is
//!   produced. The sum is therefore NOT order-independent — floating-point
//!   addition is not associative — but the order is fixed, so it does not
//!   vary between runs of the same command line.
//! - Entry order in the file: sorted by tensor name, so the byte layout does
//!   not depend on which tensor the collector happened to see first.
//!
//! Two runs of the same command line over the same artifact, on the same
//! build and device, produce the same file. NOT claimed, deliberately, and
//! for the same reasons `token_ce` states: bit-identity ACROSS devices,
//! ACROSS builds, or across differing `--dequant-weights`. A multi-threaded
//! reduction inside a backend kernel may reorder its partial sums, so
//! run-to-run bit-identity is an empirical property of the backend. Run twice
//! and compare the files before trusting a small difference.
//!
//! # Cost
//!
//! One prefill of `--seq-len` tokens per window, `--windows` times, plus one
//! model load — the same shape as `token_ce`, without the loss. Memory adds
//! one `[in_features]` `f32` device tensor per measured weight, held for the
//! whole run; for a dense transformer that is roughly one hidden-size vector
//! per projection, which is negligible beside the weights themselves. The
//! ONLY device-to-host transfer is those vectors, once, at the end.
//!
//! # Output
//!
//! The importance file at `--out`, and ONE JSON summary object on stdout and
//! nothing else on stdout. Progress goes to stderr.

use std::time::Instant;

use boostr::model::llama::Llama;
use boostr::model::traits::{Model, ModelClient};
use boostr::nn::VarBuilder;
use boostr::quant::imatrix;
use boostr::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
#[cfg(feature = "cuda")]
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

// Artifact resolution and `VarMap` loading, shared with `token_ce`.
#[path = "../shared/source.rs"]
mod source;
use source::{Weights, config_path, load_config, load_varmap, source_format, source_path};

// The corpus and the window selection, shared with `token_ce` so both
// binaries select the same tokens.
#[path = "../shared/windows.rs"]
mod windows;
use windows::{select_windows, tokenize_corpus};

// The calibration pass itself.
mod collect;
use collect::collect_windows;

// Flags, defaults and usage text.
mod cli;
use cli::{Args, Device, parse_args};

/// The calibration body: everything that runs on the chosen runtime `R`.
fn run<R, C>(args: &Args, device: &R::Device, client: &C) -> Result<(), Box<dyn std::error::Error>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R> + DequantOps<R> + TypeConversionOps<R>,
    R::Client: ModelClient<R> + DequantOps<R> + TypeConversionOps<R>,
{
    let started = Instant::now();
    let stride = args.stride.unwrap_or(args.seq_len);

    let packed_source = !matches!(args.weights, Weights::Checkpoint(_));
    if packed_source && !args.dequant_weights {
        eprintln!(
            "note: this artifact can store matmul weights PACKED. A packed weight runs a \
             quantized kernel, has no dense weight to guide, and is not measured. Pass \
             --dequant-weights to materialize every weight to dense F32 at load"
        );
    }

    let config_file = config_path(&args.weights, args.config.as_deref())?;
    let config = load_config(&config_file)?;
    if config.attention.is_none() {
        return Err(format!(
            "{}: this config declares no attention block, so it is not a causal transformer \
             this example can calibrate",
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

    // ARM BEFORE LOADING. The checkpoint names are bound while the weights
    // are read, by the `VarBuilder` that reads them — that is the only place
    // a name exists without being inferred. Arming after the model was built
    // would collect statistics no name could be attached to, and
    // `imatrix::finish` would refuse the run rather than guess.
    imatrix::arm()?;
    let collected = match load_and_collect::<R, C>(args, device, client, &config, &selected) {
        Ok(collected) => collected,
        Err(e) => {
            // Leave no armed collector behind for a later run in this process.
            imatrix::disarm();
            return Err(e);
        }
    };

    if collected.matrix.is_empty() {
        return Err(format!(
            "no weight was measured. {}",
            if packed_source && !args.dequant_weights {
                "This artifact's matmul weights are stored packed; pass --dequant-weights"
            } else {
                "The model exercised no dense linear layer"
            }
        )
        .into());
    }
    for name in &collected.unexercised {
        eprintln!("not exercised, absent from the file: {name}");
    }

    collected.matrix.write_to_path(&args.out)?;

    let record = serde_json::json!({
        "record": "imatrix",
        "source_format": source_format(&args.weights),
        "model_path": source_path(&args.weights).display().to_string(),
        "config": config_file.display().to_string(),
        "tokenizer": tokenizer_path.display().to_string(),
        "text": args.text.display().to_string(),
        "out": args.out.display().to_string(),
        "device": match args.device {
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
        },
        "weights_dense": args.dequant_weights,
        "seq_len": args.seq_len,
        "windows": selected.len(),
        "stride": stride,
        "corpus_tokens": tokens.len(),
        "accumulated_tokens": collected.tokens,
        "measured_tensors": collected.matrix.len(),
        "unexercised_tensors": collected.unexercised.len(),
        "num_layers": config.num_layers,
    });
    println!("{}", serde_json::to_string(&record)?);
    eprintln!(
        "wrote {} ({} tensor(s), {} token(s)); total {:?}",
        args.out.display(),
        collected.matrix.len(),
        collected.tokens,
        started.elapsed()
    );
    Ok(())
}

/// Load the model with the collector already armed, then run every window.
///
/// Separate from [`run`] so the caller has exactly one place to disarm the
/// collector if anything between arming and finishing fails.
fn load_and_collect<R, C>(
    args: &Args,
    device: &R::Device,
    client: &C,
    config: &boostr::model::ModelConfig,
    selected: &[&[i64]],
) -> Result<collect::Collected, Box<dyn std::error::Error>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R> + DequantOps<R> + TypeConversionOps<R>,
    R::Client: ModelClient<R> + DequantOps<R> + TypeConversionOps<R>,
{
    eprintln!("loading {} ...", source_path(&args.weights).display());
    let mut var_map = load_varmap::<R, C>(&args.weights, args.dequant_weights, device, client)?;
    let mut vb = VarBuilder::new(&mut var_map, device);
    let model = Llama::<R>::from_varbuilder(&mut vb, config)?;
    eprintln!(
        "model: {} layer(s), vocab {}, {} dense weight(s) bound to a checkpoint name",
        config.num_layers,
        config.vocab_size,
        imatrix::registered_names().len()
    );
    collect_windows::<R, C>(&model, client, device, selected)
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
