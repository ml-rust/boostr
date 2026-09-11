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

use boostr::nn::VarMap;
use boostr::quant::{ImportanceMatrix, smoothing_scale, weight_only_smoothing_scale};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use tcf_core::{
    NativeEncoding, QuantizeParams, SearchEffort, WeightSource, column_weights, dequantize_into,
    quantize_with,
};

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

/// Full accounting of one smoothing pass, so the totals visibly reconcile:
/// `examined == candidates + non_candidates` and
/// `candidates == transformed + skipped_no_entry`.
pub struct SmoothingSummary {
    /// Varmap entries looked at, of any dtype or shape.
    pub examined: usize,
    /// `Standard` tensors cast from a non-F32 dtype to F32. A tensor already
    /// F32 is not counted here even though it was examined.
    pub cast_to_f32: usize,
    /// Rank-2, contiguous, non-quantized tensors — the only shape this pass
    /// can smooth.
    pub candidates: usize,
    /// Candidates with no importance entry, left exactly as loaded.
    pub skipped_no_entry: usize,
    /// Entries that are not smoothing candidates at all: block-quantized, or
    /// not rank-2/contiguous.
    pub non_candidates: usize,
    /// Candidates actually put through the round trip.
    pub transformed: usize,
}

/// Overwrite `tensor`'s device buffer with `values`. Same entry point
/// `examples/voxcpm/sensitivity.rs` uses to rewrite a tensor in place.
fn write_values<R: Runtime<DType = DType>>(
    tensor: &Tensor<R>,
    values: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes: &[u8] = bytemuck::cast_slice(values);
    R::copy_to_device(bytes, tensor.ptr(), tensor.device())?;
    Ok(())
}

/// Cast every non-quantized, non-F32 tensor in `var_map` to F32, then apply
/// the smoothing round trip to every rank-2 contiguous weight that has an
/// importance entry, in place. `source` selects which scale is computed;
/// either way, the importance matrix decides only which tensors are touched
/// — see the module docs' "Source" section.
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
) -> Result<SmoothingSummary, Box<dyn std::error::Error>>
where
    R: Runtime<DType = DType>,
    R::Client: TypeConversionOps<R>,
{
    let layout = encoding.layout();
    let names: Vec<String> = var_map.names().map(str::to_string).collect();
    let examined = names.len();

    let mut cast_to_f32 = 0usize;
    let mut candidates = 0usize;
    let mut non_candidates = 0usize;
    let mut skipped_no_entry = 0usize;
    let mut transformed = 0usize;

    for name in &names {
        // A block-quantized weight cannot occur behind `--ckpt`, but this is
        // checked rather than assumed: it is not `Standard`, so it has no
        // dense buffer to cast or rewrite.
        if var_map.get(name)?.is_quantized() {
            non_candidates += 1;
            continue;
        }

        // Cast BEFORE the shape check: every tensor is cast regardless of
        // candidacy, mirroring the all-F32 varmap a `--tcf --dequant-weights`
        // run produces (see the module docs). Each `get_tensor` call's borrow
        // ends with the value it returns, so the later `insert` never
        // conflicts with a live reference into the same map.
        if var_map.get_tensor(name)?.dtype() != DType::F32 {
            let casted = var_map.get_tensor(name)?.to_dtype(DType::F32)?;
            var_map.insert(name.clone(), casted);
            cast_to_f32 += 1;
        }

        let tensor = var_map.get_tensor(name)?;
        let shape: Vec<usize> = tensor.shape().to_vec();
        if shape.len() != 2 || !tensor.is_contiguous() {
            // Not a candidate at all: an embedding, a norm, a bias, or a
            // tensor this pass cannot rewrite in place. Left untouched
            // (beyond the F32 cast above) and not counted as
            // skipped-for-no-entry, which is a different situation (see the
            // module docs).
            non_candidates += 1;
            continue;
        }
        candidates += 1;
        let out_features = shape[0];
        let in_features = shape[1];

        let entry = match imatrix.get(name) {
            Some(entry) => entry,
            None => {
                skipped_no_entry += 1;
                continue;
            }
        };
        if entry.in_features() != in_features {
            return Err(format!(
                "{name}: importance file holds {} column(s), this weight has {in_features}",
                entry.in_features()
            )
            .into());
        }
        // `rows == 0` means the entry carries no usable mean — measured
        // nothing, in effect. Handled the same as no entry at all: this
        // tensor is left exactly as loaded.
        let mean_square = match entry.mean_square() {
            Some(values) => values,
            None => {
                skipped_no_entry += 1;
                continue;
            }
        };

        // One tensor in flight: the snapshot, then the scaled copy, then the
        // round trip, then the reconstruction — each buffer dropped as soon
        // as the next step no longer needs it.
        let original: Vec<f32> = tensor.try_to_vec::<f32>()?;

        // `mean_square` is fetched above regardless of `source`: it is what
        // makes an entry "usable" (see the `rows == 0` comment above), and
        // `--smooth-objective imatrix` needs it too. The weight-only source
        // below never reads it for the scale itself — see the module docs.
        let s = match source {
            SmoothSource::Activation => {
                smoothing_scale(&mean_square, &original, in_features, alpha)
            }
            SmoothSource::Weight => weight_only_smoothing_scale(&original, in_features, alpha),
        };

        // Step 1: W * s, broadcast along the input (last) dimension.
        let mut scaled: Vec<f32> = Vec::with_capacity(original.len());
        for row in 0..out_features {
            let base = row * in_features;
            for j in 0..in_features {
                scaled.push(original[base + j] * s[j]);
            }
        }
        drop(original);

        // Step 2: quantize -> dequantize through tcf-core, under the
        // selected error objective.
        let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        let rank = dims.len() as u32;
        let expanded_weights: Option<Vec<f32>> = match objective {
            SmoothObjective::Uniform => None,
            SmoothObjective::Imatrix => Some(
                column_weights(&mean_square, in_features, 0, scaled.len())
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

        write_values(tensor, &reconstructed)?;
        drop(reconstructed);
        transformed += 1;
    }

    if transformed == 0 {
        return Err(format!(
            "smoothing transformed 0 tensors — refusing to score an untransformed baseline as \
             if it were a measurement. Examined {examined} varmap entr{}, found {candidates} \
             rank-2 contiguous candidate(s) ({non_candidates} non-candidate(s) left alone), of \
             which {skipped_no_entry} had no usable importance entry. Cast {cast_to_f32} \
             tensor(s) to F32 first. Check that --smooth-imatrix was collected against this \
             exact checkpoint's tensor names.",
            if examined == 1 { "y" } else { "ies" }
        )
        .into());
    }

    Ok(SmoothingSummary {
        examined,
        cast_to_f32,
        candidates,
        skipped_no_entry,
        non_candidates,
        transformed,
    })
}
