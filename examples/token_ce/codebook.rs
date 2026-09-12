//! Codebook probe: does moving the 16 reconstruction levels off a uniform
//! grid reduce task damage, holding geometry and byte cost fixed? And,
//! independently, does adding a per-group minimum (affine) instead of a
//! pure scale (symmetric) reduce it? A third, separate geometry probes
//! whether a two-level super-scale's STORAGE FORMAT (bf16 pre-divided,
//! f16 undivided, or an unrounded f32 ceiling) explains TCF `Q6S16D_T64`'s
//! gap against GGUF `q6_k` — and whether that finding generalizes to
//! `Q4AS32D_T64`'s asymmetric geometry, which stores TWO such supers (a
//! scale and a minimum). `--codebook` selects one of all eleven
//! combinations — see [`parse_codebook`].
//!
//! Same probe discipline as `smooth.rs`: transform a loaded `--ckpt`'s dense
//! weights in F32, write them back, score — no file is written, and TCF's
//! own codec is not touched. Unlike smoothing, this path applies NO
//! per-channel scale: `boostr::quant::codebook_round_trip`,
//! `affine_codebook_round_trip`, `two_level_codebook_round_trip` and
//! `two_level_asymmetric_round_trip` are complete block quantizers on their
//! own (group 32 with one scale per group, plus one minimum for the affine
//! arm; group 16 with one sub-scale per group and one super-scale per 256
//! for the symmetric two-level arm; group 32 with one sub-scale AND one
//! sub-minimum per group, one super-scale AND one super-minimum per 256,
//! for the asymmetric two-level arm), so the transform is exactly
//! quantize-then-dequantize, no unscale step.
//!
//! `--codebook-objective` selects the per-element weight the group search
//! scores against, exactly as `--smooth-objective` does for the smoothing
//! path: `imatrix` (default) expands the same per-column importance
//! `--smooth-imatrix` supplies; `uniform` scores every element equally.
//! Either way the importance matrix still gates which tensors are
//! transformed — see `super::probe::run_probe`.

use boostr::nn::VarMap;
use boostr::quant::{
    AffineCodebook, Codebook, ImportanceMatrix, SuperPrecision, affine_codebook_round_trip,
    codebook_round_trip, two_level_asymmetric_round_trip, two_level_codebook_round_trip,
};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;

use super::probe::{ProbeSummary, run_probe};

/// The codebook this probe quantizes each candidate weight against: a
/// SYMMETRIC 16-level grid (`d * level`), an AFFINE one (`m + d * level`), a
/// TWO-LEVEL symmetric super-scale probe (6-bit codes, one super-scale per
/// 256 elements, storage format given by [`SuperPrecision`]), or a
/// TWO-LEVEL ASYMMETRIC super-scale probe (4-bit unsigned codes, one
/// super-scale AND one super-minimum per 256 elements, same
/// [`SuperPrecision`] applied to both). One `--codebook` flag selects among
/// all eleven underlying codebooks — see [`parse_codebook`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CodebookChoice {
    Symmetric(Codebook),
    Affine(AffineCodebook),
    TwoLevel(SuperPrecision),
    TwoLevelAsymmetric(SuperPrecision),
}

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

pub fn parse_codebook(value: &str) -> Result<CodebookChoice, String> {
    match value {
        "uniform" => Ok(CodebookChoice::Symmetric(Codebook::Uniform)),
        "nf4" => Ok(CodebookChoice::Symmetric(Codebook::Nf4)),
        "uniform-affine" => Ok(CodebookChoice::Affine(AffineCodebook::Uniform)),
        "nf4-affine" => Ok(CodebookChoice::Affine(AffineCodebook::Nf4Shifted)),
        "q6-bf16" => Ok(CodebookChoice::TwoLevel(SuperPrecision::Bf16)),
        "q6-f16" => Ok(CodebookChoice::TwoLevel(SuperPrecision::F16)),
        "q6-f32" => Ok(CodebookChoice::TwoLevel(SuperPrecision::F32)),
        "q6-bf16-reserved" => Ok(CodebookChoice::TwoLevel(SuperPrecision::Bf16Reserved)),
        "q4a-bf16" => Ok(CodebookChoice::TwoLevelAsymmetric(SuperPrecision::Bf16)),
        "q4a-f16" => Ok(CodebookChoice::TwoLevelAsymmetric(SuperPrecision::F16)),
        "q4a-f32" => Ok(CodebookChoice::TwoLevelAsymmetric(SuperPrecision::F32)),
        other => Err(format!(
            "--codebook: expected uniform, nf4, uniform-affine, nf4-affine, q6-bf16, q6-f16, \
             q6-f32, q6-bf16-reserved, q4a-bf16, q4a-f16 or q4a-f32, got {other:?}"
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
    codebook: CodebookChoice,
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
                CodebookObjective::Imatrix => column_weights(mean_square, original.len()),
            };
            Ok(match codebook {
                CodebookChoice::Symmetric(codebook) => {
                    codebook_round_trip(original, in_features, codebook, &weights)
                }
                CodebookChoice::Affine(codebook) => {
                    affine_codebook_round_trip(original, in_features, codebook, &weights)
                }
                CodebookChoice::TwoLevel(precision) => {
                    two_level_codebook_round_trip(original, in_features, precision, &weights)
                }
                CodebookChoice::TwoLevelAsymmetric(precision) => {
                    two_level_asymmetric_round_trip(original, in_features, precision, &weights)
                        .map_err(|e| format!("{name}: {e}"))?
                }
            })
        },
    )
}

/// One weight per element from a per-column importance vector: row-major, so
/// every row reads the same `mean_square` entry for its column.
fn column_weights(mean_square: &[f32], count: usize) -> Vec<f32> {
    mean_square.iter().copied().cycle().take(count).collect()
}
