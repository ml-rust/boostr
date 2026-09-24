//! Sample a token from logits — generic implementation.

use super::stochastic::{filter_distribution, sample_from_distribution};
use super::validate::{SamplingLogitsShape, validate_sampling_logits};
use crate::error::Result;
use numr::ops::RandomOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Sample a token from logits — generic implementation.
///
/// Performs temperature → softmax → top-k → top-p → min-p → multinomial.
/// Randomness generated via `RandomOps::rand` (on-device for GPU backends).
/// CUDA backend provides a fused kernel that keeps everything in a single launch.
pub fn sample_token_impl<R: Runtime>(
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
    validate_sampling_logits("sample_token", logits, SamplingLogitsShape::Vocab)?;
    let mut logits_vec: Vec<f32> = logits.to_vec();

    let indexed = filter_distribution(&mut logits_vec, temperature, top_k, top_p, min_p);

    // Generate random value on-device via RandomOps, read back scalar
    let rand_tensor = client
        .rand(&[1], numr::dtype::DType::F32)
        .map_err(crate::error::Error::Numr)?;
    let random_val: f32 = rand_tensor
        .item::<f32>()
        .map_err(crate::error::Error::Numr)?;

    Ok(sample_from_distribution(&indexed, random_val) as u32)
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
}
