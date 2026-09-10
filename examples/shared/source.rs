//! Where the weights come from, and how one open artifact becomes a `VarMap`.
//!
//! The three accepted forms — a checkpoint directory, a GGUF file, a TCF file
//! — reach the model through ONE path:
//! [`WeightSource`](boostr::format::weight_source::WeightSource), the
//! named-tensor read contract every checkpoint format implements, and
//! `VarMap::from_weight_source`, which drains a source into the map
//! `VarBuilder` reads. Nothing here knows a block layout, a plane offset or a
//! GGML type; a format that keeps a matmul weight packed says so through the
//! trait and the model multiplies it with `quant_matmul` unchanged.
//!
//! `--dequant-weights` inserts exactly one thing into that path:
//! [`DenseWeightSource`](boostr::format::weight_source::DenseWeightSource),
//! the decorator that turns every packed weight into a dense F32 tensor on the
//! way in. It is the same decorator the VoxCPM2 dense loaders use, not a
//! second copy of it, so the two measurements are made under one definition of
//! "dense".
//!
//! Shared by the `token_ce` and `imatrix` examples, in `examples/shared/`, so
//! both binaries open the same artifact forms by the same rules. Each
//! compiles this module separately and uses a subset of it, so an item unused
//! by one of them is not dead code.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use boostr::format::gguf::{Gguf, gguf_to_hf_name};
use boostr::format::weight_source::{DenseWeightSource, TcfSource};
use boostr::format::{SafeTensorsLoader, TcfLoader};
use boostr::model::{ModelConfig, load_config_auto};
use boostr::nn::VarMap;
use boostr::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Where the model's weights come from.
///
/// `--ckpt` names a checkpoint DIRECTORY (`config.json`, one or more
/// `.safetensors` files, `tokenizer.json`); `--gguf` and `--tcf` each name a
/// single file that carries weights and nothing else. Mutually exclusive, and
/// exactly one is required.
pub enum Weights {
    Checkpoint(PathBuf),
    Gguf(PathBuf),
    Tcf(PathBuf),
}

/// The weight source's short name, for the machine-readable record. Same
/// vocabulary `voxcpm_finetune`'s JSON uses, so one comparison table can key
/// on a single set of names across both binaries.
pub fn source_format(weights: &Weights) -> &'static str {
    match weights {
        Weights::Checkpoint(_) => "checkpoint",
        Weights::Gguf(_) => "gguf",
        Weights::Tcf(_) => "tcf",
    }
}

/// The weight file or directory, for the machine-readable record.
pub fn source_path(weights: &Weights) -> &Path {
    match weights {
        Weights::Checkpoint(path) | Weights::Gguf(path) | Weights::Tcf(path) => path,
    }
}

/// Locate the architecture config.
///
/// A checkpoint directory holds its own `config.json`. Neither a GGUF nor a
/// TCF is read for architecture here, so `--config` is required with both:
/// guessing an architecture would silently score a different model than the
/// artifact encodes.
pub fn config_path(weights: &Weights, config: Option<&Path>) -> Result<PathBuf, String> {
    match (weights, config) {
        (_, Some(path)) => Ok(path.to_path_buf()),
        (Weights::Checkpoint(dir), None) => Ok(dir.join("config.json")),
        (Weights::Gguf(path), None) | (Weights::Tcf(path), None) => Err(format!(
            "--config is required with {}: a single-file artifact carries no config.json",
            path.display()
        )),
    }
}

/// Locate `tokenizer.json`.
///
/// A checkpoint directory holds it outright. Neither a GGUF nor a TCF carries
/// one, so it is looked for beside the model file first and beside `--config`
/// second — both normally sit in, or were copied from, the same checkpoint
/// directory. Neither: an error, rather than a tokenizer guess that would
/// silently produce different token ids and therefore different windows. Same
/// resolution order `voxcpm_finetune` uses.
pub fn tokenizer_path(
    weights: &Weights,
    tokenizer: Option<&Path>,
    config: Option<&Path>,
) -> Result<PathBuf, String> {
    if let Some(path) = tokenizer {
        return Ok(path.to_path_buf());
    }
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
                         put it there, pass --tokenizer, or pass --config pointing into the \
                         checkpoint directory",
                        path.display()
                    )
                })
        }
    }
}

/// Parse the architecture config named by [`config_path`].
pub fn load_config(path: &Path) -> Result<ModelConfig, Box<dyn std::error::Error>> {
    Ok(load_config_auto(path)?)
}

/// Drain the artifact into a `VarMap` keyed by the names `VarBuilder` asks
/// for.
///
/// `dense` is `--dequant-weights`: every packed weight is materialized to
/// dense F32 at load, so two artifacts run the same activation contract and
/// differ only in weight VALUES. Off, each format is loaded as it is stored —
/// a GGUF K-quant and a TCF native encoding both stay packed for
/// `quant_matmul`.
///
/// A checkpoint stores no packed weight, so `dense` changes nothing on that
/// path and the caller says so on stderr rather than implying a conversion
/// happened.
pub fn load_varmap<R, C>(
    weights: &Weights,
    dense: bool,
    device: &R::Device,
    client: &C,
) -> Result<VarMap<R>, Box<dyn std::error::Error>>
where
    R: Runtime<DType = DType>,
    R::Client: numr::ops::ShapeOps<R>,
    C: DequantOps<R>,
{
    let map = match weights {
        // Handles a single `model.safetensors` and a sharded directory alike.
        // A safetensors tensor is always dense, so the decorator is a
        // pass-through here and is applied anyway rather than branching: one
        // code path, one contract.
        Weights::Checkpoint(dir) => {
            let mut loader = SafeTensorsLoader::open(dir)?;
            let names = loader.tensor_names();
            if dense {
                VarMap::<R>::from_weight_source(
                    &mut DenseWeightSource::new(&mut loader, client),
                    &names,
                    |name| name.to_string(),
                    device,
                )?
            } else {
                VarMap::<R>::from_weight_source(
                    &mut loader,
                    &names,
                    |name| name.to_string(),
                    device,
                )?
            }
        }
        // GGUF tensor names are rewritten to their HuggingFace spellings,
        // exactly as `VarMap::from_gguf` does, because that is the vocabulary
        // the model's `VarBuilder` prefixes are written in.
        Weights::Gguf(path) => {
            let mut gguf = Gguf::open(path)?;
            let names: Vec<String> = gguf.tensor_names().map(|s| s.to_string()).collect();
            if dense {
                VarMap::<R>::from_weight_source(
                    &mut DenseWeightSource::new(&mut gguf, client),
                    &names,
                    gguf_to_hf_name,
                    device,
                )?
            } else {
                VarMap::<R>::from_weight_source(&mut gguf, &names, gguf_to_hf_name, device)?
            }
        }
        // `compressr convert --format tcf` writes the checkpoint's ORIGINAL
        // HuggingFace tensor names verbatim, so no rewrite applies. Opening
        // the file validates its header, directory and every record digest;
        // each tensor load then verifies its own payload (SPECIFICATION.md
        // Section 15) before any value is used.
        Weights::Tcf(path) => {
            let loader = TcfLoader::open(path)?;
            let names: Vec<String> = loader.tensor_names().map(|s| s.to_string()).collect();
            let mut source = TcfSource::new(&loader)?;
            if dense {
                VarMap::<R>::from_weight_source(
                    &mut DenseWeightSource::new(&mut source, client),
                    &names,
                    |name| name.to_string(),
                    device,
                )?
            } else {
                VarMap::<R>::from_weight_source(
                    &mut source,
                    &names,
                    |name| name.to_string(),
                    device,
                )?
            }
        }
    };
    Ok(map)
}
