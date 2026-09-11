//! Codebook probe: does moving the 16 reconstruction levels off a uniform
//! grid reduce task damage, holding geometry and byte cost fixed?
//!
//! Same probe discipline as `smooth.rs`: transform a loaded `--ckpt`'s dense
//! weights in F32, write them back, score — no file is written, and TCF's
//! own codec is not touched. Unlike smoothing, this path applies NO
//! per-channel scale: `boostr::quant::codebook_round_trip` is a complete
//! block quantizer on its own (symmetric, group 32, one scale per group), so
//! the transform is exactly quantize-then-dequantize, no unscale step.
//!
//! `--codebook-objective` selects the per-element weight the group search
//! scores against, exactly as `--smooth-objective` does for the smoothing
//! path: `imatrix` (default) expands the same per-column importance
//! `--smooth-imatrix` supplies; `uniform` scores every element equally.
//! Either way the importance matrix still gates which tensors are
//! transformed — see `super::probe::run_probe`.

use boostr::nn::VarMap;
use boostr::quant::{Codebook, ImportanceMatrix, codebook_round_trip};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use tcf_core::column_weights;

use super::probe::{ProbeSummary, run_probe};

/// Which per-element weight the codebook group search scores against.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CodebookObjective {
    /// Every element weighted equally: `tcf-core`'s plain unweighted error.
    Uniform,
    /// The importance entry's per-column mean square, expanded to every
    /// element — what a real imatrix-guided conversion scores against.
    Imatrix,
}

pub fn parse_codebook_objective(value: &str) -> Result<CodebookObjective, String> {
    match value {
        "uniform" => Ok(CodebookObjective::Uniform),
        "imatrix" => Ok(CodebookObjective::Imatrix),
        other => Err(format!(
            "--codebook-objective: expected uniform or imatrix, got {other:?}"
        )),
    }
}

pub fn parse_codebook(value: &str) -> Result<Codebook, String> {
    match value {
        "uniform" => Ok(Codebook::Uniform),
        "nf4" => Ok(Codebook::Nf4),
        other => Err(format!(
            "--codebook: expected uniform or nf4, got {other:?}"
        )),
    }
}

/// Runs [`super::probe::run_probe`] with the codebook transform: quantize
/// then dequantize every candidate weight against `codebook`, under
/// `objective`. No per-channel scale is applied — see the module docs.
///
/// Returns an error if it transforms zero tensors: same rule
/// [`super::smooth::apply_smoothing`] follows, and for the same reason.
pub fn apply_codebook<R>(
    var_map: &mut VarMap<R>,
    imatrix: &ImportanceMatrix,
    codebook: Codebook,
    objective: CodebookObjective,
) -> Result<ProbeSummary, Box<dyn std::error::Error>>
where
    R: Runtime<DType = DType>,
    R::Client: TypeConversionOps<R>,
{
    run_probe(
        var_map,
        imatrix,
        |name, original, in_features, mean_square| {
            let weights: Vec<f32> = match objective {
                CodebookObjective::Uniform => vec![1.0f32; original.len()],
                CodebookObjective::Imatrix => {
                    column_weights(mean_square, in_features, 0, original.len())
                        .map_err(|e| format!("{name}: expanding importance weights: {e}"))?
                }
            };
            Ok(codebook_round_trip(
                original,
                in_features,
                codebook,
                &weights,
            ))
        },
    )
}
