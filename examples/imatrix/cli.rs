//! Command line for the `imatrix` example.
//!
//! Every flag `token_ce` shares is spelled and defaulted identically, so one
//! command line ports between the two binaries unchanged and both cover the
//! same corpus slice. `--out` is the one addition.

use std::path::PathBuf;

use crate::source::Weights;

/// Tokens per window. Same default as `token_ce`, so the two binaries cover
/// the same corpus slice when neither flag is passed.
const DEFAULT_SEQ_LEN: usize = 512;
/// Windows per run. The cost dial, and the token count the sums average over.
const DEFAULT_WINDOWS: usize = 64;

/// Runtime to load the model and run the calibration on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Device {
    Cpu,
    Cuda,
}

pub fn parse_device(value: &str) -> Result<Device, String> {
    match value {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda),
        other => Err(format!(
            "--device: expected one of cpu, cuda, got {other:?}"
        )),
    }
}

/// Everything the command line decided.
pub struct Args {
    pub weights: Weights,
    pub config: Option<PathBuf>,
    pub tokenizer: Option<PathBuf>,
    pub text: PathBuf,
    pub out: PathBuf,
    pub device: Device,
    pub seq_len: usize,
    pub windows: usize,
    pub stride: Option<usize>,
    pub dequant_weights: bool,
}

pub const USAGE: &str = "usage: imatrix (--ckpt DIR | --gguf MODEL.gguf | --tcf MODEL.tcf) \
[--config config.json (required with --gguf and --tcf, which carry no architecture config)] \
--text FILE.txt (the corpus the calibration windows are taken from, in file order) \
--out IMPORTANCE.bstrimtx (the importance file to write) \
[--tokenizer tokenizer.json (default: the checkpoint directory's, or the one beside a \
single-file artifact)] \
[--device cpu|cuda] \
[--seq-len 512 (tokens per window)] \
[--windows 64 (windows run; the cost dial)] \
[--stride N (tokens between two window starts; default --seq-len, i.e. no overlap)] \
[--dequant-weights (dequantize EVERY packed weight to dense F32 at load; REQUIRED to \
collect from an artifact whose matmul weights are stored packed, which otherwise runs a \
quantized kernel and measures nothing)]";

/// Consume the value that follows `flag`, advancing `i` past it.
fn take_value(argv: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    *i += 1;
    argv.get(*i)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

pub fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut ckpt: Option<PathBuf> = None;
    let mut gguf: Option<PathBuf> = None;
    let mut tcf: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut text: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut device = Device::Cpu;
    let mut seq_len = DEFAULT_SEQ_LEN;
    let mut windows = DEFAULT_WINDOWS;
    let mut stride: Option<usize> = None;
    let mut dequant_weights = false;

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
            "--out" => out = Some(PathBuf::from(take_value(&argv, &mut i, flag)?)),
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
            "--stride must be at least 1: stride 0 accumulates one window repeatedly".to_string(),
        );
    }

    // Exactly one weight source, same rule `token_ce` applies: accepting two
    // and silently preferring one would calibrate a different artifact than
    // the operator asked for.
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

    Ok(Args {
        weights,
        config,
        tokenizer,
        text: text.ok_or_else(|| format!("--text is required\n{USAGE}"))?,
        out: out.ok_or_else(|| format!("--out is required\n{USAGE}"))?,
        device,
        seq_len,
        windows,
        stride,
        dequant_weights,
    })
}
