//! Sample a token from logits — generic implementation.

use super::output::write_token_id;
use super::stochastic::{filtered_probs, sample_from_probs};
use super::validate::{SamplingLogitsShape, validate_sampling_logits, validate_scratch_vocab};
use crate::error::Result;
use crate::ops::traits::inference::sampling_probs::FilterRule;
use crate::ops::traits::inference::sampling_scratch::SamplingScratch;
use numr::dtype::DType;
use numr::ops::{CompareOps, RandomOps, ScalarOps, TensorOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Sample a token from logits — generic implementation.
///
/// Filters by the canonical rule in [`super::probs`], then draws in id order.
/// Randomness generated via `RandomOps::rand_into` (on-device for GPU backends).
/// CUDA backend provides a fused kernel that keeps everything in a single launch.
///
/// Allocates a one-off [`SamplingScratch`] and delegates to
/// [`sample_token_into_impl`]. A caller that samples every step should
/// allocate its own scratch once and call [`sample_token_into_impl`]
/// directly instead.
///
/// # Errors
///
/// Returns an error if the [`FilterRule`] of the arguments fails its check.
pub fn sample_token_impl<R: Runtime<DType = DType>>(
    client: &R::Client,
    logits: &Tensor<R>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
) -> Result<u32>
where
    R::Client: RandomOps<R>
        + numr::ops::AdvancedRandomOps<R>
        + TensorOps<R>
        + CompareOps<R>
        + ScalarOps<R>,
{
    let vocab_size = logits.shape().first().copied().unwrap_or(1);
    let scratch = SamplingScratch::new(vocab_size, logits.device())?;
    sample_token_into_impl(client, logits, temperature, top_k, top_p, min_p, &scratch)?;
    let token_id: Vec<i64> = scratch.output.to_vec();
    Ok(token_id[0] as u32)
}

/// Like [`sample_token_impl`], but writes the sampled token into
/// `scratch.output` instead of allocating.
///
/// `scratch.probs` is unused on this path: the filter returns a fresh
/// probability tensor, and the draw reads it on the host. `scratch.output`
/// is fully overwritten before this function returns, so it carries nothing
/// from a previous call.
pub fn sample_token_into_impl<R: Runtime<DType = DType>>(
    client: &R::Client,
    logits: &Tensor<R>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    scratch: &SamplingScratch<R>,
) -> Result<()>
where
    R::Client: RandomOps<R>
        + numr::ops::AdvancedRandomOps<R>
        + TensorOps<R>
        + CompareOps<R>
        + ScalarOps<R>,
{
    validate_sampling_logits("sample_token", logits, SamplingLogitsShape::Vocab)?;
    validate_scratch_vocab("sample_token", scratch, logits.shape()[0])?;
    let rule = FilterRule {
        temperature,
        top_k,
        top_p,
        min_p,
    };
    let probs = filtered_probs::<R>(client, logits, &rule)?;

    // Draw into scratch.random instead of allocating a fresh tensor, then
    // read back the scalar it was just fully overwritten with.
    scratch.draw_random(client, None)?;
    let random_val: f32 = scratch
        .random
        .item::<f32>()
        .map_err(crate::error::Error::Numr)?;

    let token_id = sample_from_probs(&probs, random_val) as i64;
    write_token_id("sample_token", token_id, &scratch.output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    /// A `[vocab_size]` view with stride 2: every other element of a
    /// `[vocab_size, 2]` buffer.
    fn non_contiguous_vocab(device: &CpuDevice) -> Tensor<CpuRuntime> {
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let base = Tensor::<CpuRuntime>::from_slice(&data, &[4, 2], device).expect("base");
        let view = base.narrow(1, 0, 1).expect("narrow").squeeze(Some(1));
        assert!(!view.is_contiguous());
        view
    }

    #[test]
    fn rejects_non_contiguous_logits() {
        let (client, device) = cpu_setup();
        let logits = non_contiguous_vocab(&device);

        let err = sample_token_impl::<CpuRuntime>(&client, &logits, 1.0, 0, 1.0, 0.0)
            .expect_err("non-contiguous logits must be rejected");
        assert!(format!("{err}").contains("sample_token"));
    }

    #[test]
    fn unchanged_on_contiguous_logits() {
        let (client, device) = cpu_setup();
        let logits = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 5.0, 1.0, -1.0], &[4], &device)
            .expect("logits");

        // top_k = 1 makes the outcome deterministic: only the argmax survives.
        let token = sample_token_impl::<CpuRuntime>(&client, &logits, 1.0, 1, 1.0, 0.0)
            .expect("contiguous logits must be accepted");
        assert_eq!(token, 1);
    }

    /// A scratch reused across calls produces the same tokens as a fresh
    /// allocation per call, given the same inputs and the same seed —
    /// nothing leaks between calls through the shared buffers.
    #[test]
    fn a_reused_scratch_matches_fresh_allocation_per_call() {
        let (client, device) = cpu_setup();
        let logits = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 5.0, 1.0, -1.0], &[4], &device)
            .expect("logits");
        let scratch = SamplingScratch::<CpuRuntime>::new(4, &device).expect("scratch");

        for _ in 0..3 {
            let fresh = sample_token_impl::<CpuRuntime>(&client, &logits, 1.0, 1, 1.0, 0.0)
                .expect("fresh alloc path");
            sample_token_into_impl::<CpuRuntime>(&client, &logits, 1.0, 1, 1.0, 0.0, &scratch)
                .expect("scratch path");
            let reused: Vec<i64> = scratch.output.to_vec();
            assert_eq!(reused[0] as u32, fresh);
        }
    }
}
