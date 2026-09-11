//! The tensor loop shared by every dense-weight probe under `token_ce`:
//! cast-to-F32, candidate selection, importance-entry gating, accounting,
//! and write-back. `smooth.rs`'s AWQ-style scale probe and `codebook.rs`'s
//! codebook probe both call [`run_probe`], differing only in what each does
//! to one candidate tensor's values — see each caller's `transform`
//! closure.

use boostr::nn::VarMap;
use boostr::quant::ImportanceMatrix;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Full accounting of one probe pass, so the totals visibly reconcile:
/// `examined == candidates + non_candidates` and
/// `candidates == transformed + skipped_no_entry`.
pub struct ProbeSummary {
    /// Varmap entries looked at, of any dtype or shape.
    pub examined: usize,
    /// `Standard` tensors cast from a non-F32 dtype to F32. A tensor already
    /// F32 is not counted here even though it was examined.
    pub cast_to_f32: usize,
    /// Rank-2, contiguous, non-quantized tensors — the only shape this pass
    /// can transform.
    pub candidates: usize,
    /// Candidates with no importance entry, left exactly as loaded.
    pub skipped_no_entry: usize,
    /// Entries that are not transform candidates at all: block-quantized,
    /// or not rank-2/contiguous.
    pub non_candidates: usize,
    /// Candidates actually put through `transform`.
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

/// Cast every non-quantized, non-F32 tensor in `var_map` to F32, then run
/// `transform` over every rank-2 contiguous weight that has an importance
/// entry, writing its result back in place.
///
/// `transform(name, original, in_features, mean_square)` receives the
/// tensor's current F32 values, its row width, and the importance entry's
/// per-column mean square (already checked non-empty and column-count
/// matched); it returns the reconstructed values to write back. What it
/// does with `mean_square` — derive a scale from it, expand it into
/// per-element weights, ignore it entirely — is the caller's policy, not
/// this loop's.
///
/// A tensor with no importance entry, or one whose entry carries no usable
/// mean, is left EXACTLY as loaded (beyond the F32 cast) and counted as
/// `skipped_no_entry` — never passed to `transform`. A tensor that is not a
/// rank-2 contiguous weight, or is itself block-quantized (should not occur
/// behind `--ckpt`, but checked rather than assumed), is not a candidate at
/// all (embeddings, norms, biases) and is likewise left alone.
///
/// Returns an error when it transforms zero tensors: a probe that can
/// silently do nothing and still hand the caller a cross-entropy number
/// indistinguishable from the untransformed baseline is worse than one that
/// refuses to run. The error names how many varmap entries were examined,
/// how many were candidates, and how many of those had no importance entry,
/// so the operator can tell a wrong `--ckpt`/importance-file pairing from a
/// genuinely-measured null effect.
///
/// One tensor in flight: its snapshot is read, `transform` produces the
/// reconstruction, the reconstruction is written back, and every buffer is
/// dropped before the next tensor starts.
pub fn run_probe<R>(
    var_map: &mut VarMap<R>,
    imatrix: &ImportanceMatrix,
    mut transform: impl FnMut(
        &str,
        &[f32],
        usize,
        &[f32],
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>>,
) -> Result<ProbeSummary, Box<dyn std::error::Error>>
where
    R: Runtime<DType = DType>,
    R::Client: TypeConversionOps<R>,
{
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
        // run produces. Each `get_tensor` call's borrow ends with the value
        // it returns, so the later `insert` never conflicts with a live
        // reference into the same map.
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
            // skipped-for-no-entry, which is a different situation.
            non_candidates += 1;
            continue;
        }
        candidates += 1;
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

        let original: Vec<f32> = tensor.try_to_vec::<f32>()?;
        let reconstructed = transform(name, &original, in_features, &mean_square)?;
        drop(original);

        write_values(tensor, &reconstructed)?;
        drop(reconstructed);
        transformed += 1;
    }

    if transformed == 0 {
        return Err(format!(
            "probe transformed 0 tensors — refusing to score an untransformed baseline as if it \
             were a measurement. Examined {examined} varmap entr{}, found {candidates} rank-2 \
             contiguous candidate(s) ({non_candidates} non-candidate(s) left alone), of which \
             {skipped_no_entry} had no usable importance entry. Cast {cast_to_f32} tensor(s) to \
             F32 first. Check that the importance file was collected against this exact \
             checkpoint's tensor names.",
            if examined == 1 { "y" } else { "ies" }
        )
        .into());
    }

    Ok(ProbeSummary {
        examined,
        cast_to_f32,
        candidates,
        skipped_no_entry,
        non_candidates,
        transformed,
    })
}
