//! Sample a token from logits — generic implementation.

use super::output::write_token_id;
use super::stochastic::{filter_distribution, sample_from_distribution};
use super::validate::{SamplingLogitsShape, validate_sampling_logits, validate_scratch_vocab};
use crate::error::Result;
use crate::ops::traits::inference::sampling_scratch::SamplingScratch;
use numr::dtype::DType;
use numr::ops::RandomOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Sample a token from logits — generic implementation.
///
/// Performs temperature → softmax → top-k → top-p → min-p → multinomial.
/// Randomness generated via `RandomOps::rand` (on-device for GPU backends).
/// CUDA backend provides a fused kernel that keeps everything in a single launch.
///
/// Allocates a one-off [`SamplingScratch`] and delegates to
/// [`sample_token_into_impl`]. A caller that samples every step should
/// allocate its own scratch once and call [`sample_token_into_impl`]
/// directly instead.
pub fn sample_token_impl<R: Runtime<DType = DType>>(
    client: &R::Client,
    logits: &Tensor<R>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
) -> Result<u32>
where
    R::Client: RandomOps<R>,
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
/// `scratch.probs` is unused on this path: the generic implementation
/// filters the distribution on the host, in `logits_vec`, rather than in a
/// device buffer. `scratch.output` is fully overwritten before this
/// function returns, so it carries nothing from a previous call.
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
    R::Client: RandomOps<R>,
{
    validate_sampling_logits("sample_token", logits, SamplingLogitsShape::Vocab)?;
    validate_scratch_vocab("sample_token", scratch, logits.shape()[0])?;
    let mut logits_vec: Vec<f32> = logits.to_vec();

    let indexed = filter_distribution(&mut logits_vec, temperature, top_k, top_p, min_p);

    // Generate random value on-device via RandomOps, read back scalar
    let rand_tensor = client
        .rand(&[1], numr::dtype::DType::F32)
        .map_err(crate::error::Error::Numr)?;
    let random_val: f32 = rand_tensor
        .item::<f32>()
        .map_err(crate::error::Error::Numr)?;

    let token_id = sample_from_distribution(&indexed, random_val) as i64;
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
