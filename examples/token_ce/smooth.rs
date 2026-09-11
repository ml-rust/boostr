//! AWQ-style per-input-channel weight smoothing, applied to a loaded
//! `--ckpt`'s dense weights before scoring.
//!
//! # Why this is a PROBE, not a format change
//!
//! TCF has no plane to store a per-channel scale vector in, so nothing is
//! written to a file here. Instead this computes the dense weight the format
//! WOULD reconstruct if it stored one, and scores that directly — numerically
//! exact, not an approximation, because every byte the real codec would touch
//! is touched the same way:
//!
//! 1. `W * s`, broadcasting `s` along the INPUT (last) dimension.
//! 2. Quantize `W * s` through `tcf-core`, then dequantize it.
//! 3. Divide the result by `s`, elementwise along the same axis.
//!
//! Step 3 is what keeps this weight-only: no activation is touched and no
//! contract changes, only the values `token_ce`'s existing scoring path reads
//! afterward.
//!
//! # Objective
//!
//! The quantize call in step 2 can score against the SAME imatrix used to
//! derive `s` (`--smooth-objective imatrix`, the default) or against the
//! codec's plain unweighted error (`--smooth-objective uniform`), so a run
//! can separate the smoothing effect from the objective effect.
//!
//! # Source: activation-derived, or calibration-free
//!
//! `--smooth-source activation` (the default) derives `s` from the imatrix
//! entry's RMS activation AND the weight, via
//! `boostr::quant::smoothing_scale`. `--smooth-source weight` derives `s`
//! from the weight's own column magnitudes alone, via
//! `boostr::quant::weight_only_smoothing_scale`, needing no calibration data.
//!
//! Either way, `--smooth-imatrix` still selects the transformed tensor SET —
//! only a tensor with an entry is touched — so a `weight` run and an
//! `activation` run of the same command line transform exactly the same
//! tensors and are directly comparable. The weight-only source never reads
//! the entry's activation statistic for the scale itself.
//!
//! # Every tensor is cast to F32 first, not just the candidates
//!
//! A checkpoint's stored dtype is whatever `torch_dtype` says — `bfloat16`
//! for most released Llama checkpoints — and `load_varmap` casts nothing, so
//! the varmap arrives holding that dtype. This mirrors what a
//! `--tcf --dequant-weights` run produces instead: TCF's dense loader
//! materializes EVERY tensor as F32, never a mix, so an all-F32 varmap is
//! what the transform this probe models actually runs against. So every
//! `Standard` tensor in the varmap is cast to F32 up front, whether or not it
//! ends up a smoothing candidate — a varmap left in mixed BF16/F32 would also
//! risk a dtype mismatch in the forward pass that follows. A tensor already
//! F32 is left alone; the cast count reports only tensors actually
//! converted.
//!
//! # What is left untouched
//!
//! A tensor with no importance entry is left EXACTLY as loaded (beyond the
//! F32 cast above) — never quantized, never smoothed — and counted, so a run
//! that silently touched a fraction of the model does not read as a null
//! result. A tensor that is not a rank-2 contiguous weight, or is itself
//! block-quantized (should not occur behind `--ckpt`, but checked rather than
//! assumed), is not a smoothing candidate at all and is likewise left alone
//! (embeddings, norms, biases).
//!
//! # A null result is an error, not a measurement
//!
//! [`apply_smoothing`] returns a hard error when it transforms zero tensors.
//! A probe that can silently do nothing and still hand the caller a
//! cross-entropy number indistinguishable from the untransformed baseline is
//! worse than one that refuses to run: the error names how many varmap
//! entries were examined, how many were candidates, and how many of those had
//! no importance entry, so the operator can tell a wrong `--ckpt`/
//! `--smooth-imatrix` pairing from a genuinely-measured null effect.
//!
//! # Memory
//!
//! One tensor in flight: its snapshot and its scaled copy are read, the
//! round trip runs, the reconstructed weight is written back, and every
//! buffer is dropped before the next tensor starts. The F32 cast allocates
//! one extra tensor per non-F32 source, freed once `VarMap::insert` replaces
//! the varmap's old entry.
//!
//! # Shared loop
//!
//! The cast, candidate selection, importance-entry gating, accounting, and
//! write-back are [`super::probe::run_probe`], shared with
//! `codebook.rs`'s codebook probe. This file supplies only the `transform`
//! closure: scale by `s`, round-trip through `tcf-core`, unscale by `s`.

use boostr::nn::VarMap;
use boostr::quant::{ImportanceMatrix, smoothing_scale, weight_only_smoothing_scale};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use tcf_core::{
    NativeEncoding, QuantizeParams, SearchEffort, WeightSource, column_weights, dequantize_into,
    quantize_with,
};

use super::probe::{ProbeSummary, run_probe};

/// Which error objective the quantize call in the round trip scores against.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SmoothObjective {
    /// `tcf-core`'s plain unweighted reconstruction error.
    Uniform,
    /// The same per-column importance the smoothing scale itself was derived
    /// from, expanded to every element — what a real imatrix-guided
    /// conversion scores against.
    Imatrix,
}

pub fn parse_smooth_objective(value: &str) -> Result<SmoothObjective, String> {
    match value {
        "uniform" => Ok(SmoothObjective::Uniform),
        "imatrix" => Ok(SmoothObjective::Imatrix),
        other => Err(format!(
            "--smooth-objective: expected uniform or imatrix, got {other:?}"
        )),
    }
}

/// Which per-input-channel scale the round trip smooths by.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SmoothSource {
    /// `boostr::quant::smoothing_scale`: needs the imatrix entry's RMS
    /// activation, in addition to the weight.
    Activation,
    /// `boostr::quant::weight_only_smoothing_scale`: needs only the weight's
    /// own column magnitudes, no calibration data. The importance entry is
    /// still consulted to select the tensor SET (see the module docs), never
    /// to compute the scale.
    Weight,
}

pub fn parse_smooth_source(value: &str) -> Result<SmoothSource, String> {
    match value {
        "activation" => Ok(SmoothSource::Activation),
        "weight" => Ok(SmoothSource::Weight),
        other => Err(format!(
            "--smooth-source: expected activation or weight, got {other:?}"
        )),
    }
}

/// Parse a TCF `NativeEncoding` identifier, spelled exactly as the enum's own
/// variant names (`Q4AS32DT64`, not `Q4AS32D_T64`).
pub fn parse_native_encoding(value: &str) -> Result<NativeEncoding, String> {
    const NAMES: &[(&str, NativeEncoding)] = &[
        ("Q4S32T64", NativeEncoding::Q4S32T64),
        ("Q4AS32T64", NativeEncoding::Q4AS32T64),
        ("Q4AS32DT64", NativeEncoding::Q4AS32DT64),
        ("Q4AS64T64", NativeEncoding::Q4AS64T64),
        ("Q6S32T64", NativeEncoding::Q6S32T64),
        ("Q6S16DT64", NativeEncoding::Q6S16DT64),
        ("Q8S32T64", NativeEncoding::Q8S32T64),
    ];
    NAMES
        .iter()
        .find(|(name, _)| *name == value)
        .map(|(_, encoding)| *encoding)
        .ok_or_else(|| {
            let known: Vec<&str> = NAMES.iter().map(|(name, _)| *name).collect();
            format!(
                "--smooth-encoding: expected one of {}, got {value:?}",
                known.join(", ")
            )
        })
}

/// Runs [`super::probe::run_probe`] with the AWQ-style smoothing transform:
/// derive a per-input-channel scale `s` (via `source`), compute
/// `W * s`, round-trip it through `tcf-core` under `encoding` and
/// `objective`, then divide by `s` — see the module docs' numbered steps.
///
/// Returns an error if it transforms zero tensors: see the module docs'
/// "A null result is an error, not a measurement" section.
pub fn apply_smoothing<R>(
    var_map: &mut VarMap<R>,
    imatrix: &ImportanceMatrix,
    encoding: NativeEncoding,
    alpha: f32,
    objective: SmoothObjective,
    source: SmoothSource,
) -> Result<ProbeSummary, Box<dyn std::error::Error>>
where
    R: Runtime<DType = DType>,
    R::Client: TypeConversionOps<R>,
{
    let layout = encoding.layout();

    run_probe(
        var_map,
        imatrix,
        |name, original, in_features, mean_square| {
            let out_features = original.len() / in_features;

            // `mean_square` is always usable here: `run_probe` already skipped
            // any entry without one. The weight-only source never reads it for
            // the scale itself — see the module docs' "Source" section.
            let s = match source {
                SmoothSource::Activation => {
                    smoothing_scale(mean_square, original, in_features, alpha)
                }
                SmoothSource::Weight => weight_only_smoothing_scale(original, in_features, alpha),
            };

            // Step 1: W * s, broadcast along the input (last) dimension.
            let mut scaled: Vec<f32> = Vec::with_capacity(original.len());
            for row in 0..out_features {
                let base = row * in_features;
                for j in 0..in_features {
                    scaled.push(original[base + j] * s[j]);
                }
            }

            // Step 2: quantize -> dequantize through tcf-core, under the
            // selected error objective.
            let dims: Vec<u64> = vec![out_features as u64, in_features as u64];
            let rank = dims.len() as u32;
            let expanded_weights: Option<Vec<f32>> = match objective {
                SmoothObjective::Uniform => None,
                SmoothObjective::Imatrix => Some(
                    column_weights(mean_square, in_features, 0, scaled.len())
                        .map_err(|e| format!("{name}: expanding importance weights: {e}"))?,
                ),
            };
            let weights = match &expanded_weights {
                Some(w) => WeightSource::Explicit(w),
                None => WeightSource::Uniform,
            };
            let params = QuantizeParams {
                weights,
                effort: SearchEffort::Standard,
            };
            let tiles = quantize_with(&scaled, &dims, rank, layout, params)
                .map_err(|e| format!("{name}: quantizing the smoothed weight: {e}"))?;
            drop(scaled);
            drop(expanded_weights);

            let mut reconstructed: Vec<f32> = Vec::new();
            dequantize_into(&tiles, layout, &mut reconstructed)
                .map_err(|e| format!("{name}: dequantizing the smoothed weight: {e}"))?;
            drop(tiles);

            // Step 3: divide by s, elementwise along the same axis — the
            // reconstructed weight the probe scores.
            for row in 0..out_features {
                let base = row * in_features;
                for j in 0..in_features {
                    reconstructed[base + j] /= s[j];
                }
            }

            Ok(reconstructed)
        },
    )
}
