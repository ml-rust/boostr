//! Apply sampling penalties to logits — generic implementation.

use super::validate::{SamplingLogitsShape, validate_contiguous_arg, validate_sampling_logits};
use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Apply sampling penalties to logits — generic implementation.
///
/// Pulls logits and penalty data to CPU, applies penalties, writes back.
/// CUDA backend overrides this with a fused kernel.
pub fn apply_sampling_penalties_impl<R: Runtime>(
    _client: &R::Client,
    logits: &Tensor<R>,
    token_ids: &Tensor<R>,
    token_counts: &Tensor<R>,
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
) -> Result<()> {
    validate_sampling_logits(
        "apply_sampling_penalties",
        logits,
        SamplingLogitsShape::Vocab,
    )?;
    validate_contiguous_arg("apply_sampling_penalties", "token_ids", token_ids)?;
    validate_contiguous_arg("apply_sampling_penalties", "token_counts", token_counts)?;
    let mut logits_vec: Vec<f32> = logits.to_vec();
    let ids_vec: Vec<i64> = token_ids.to_vec();
    let counts_vec: Vec<i32> = token_counts.to_vec();

    for (&token_id, &count) in ids_vec.iter().zip(counts_vec.iter()) {
        let i = token_id as usize;
        if i >= logits_vec.len() {
            continue;
        }

        // Repetition penalty (llama.cpp style)
        if repeat_penalty != 1.0 {
            if logits_vec[i] > 0.0 {
                logits_vec[i] /= repeat_penalty;
            } else {
                logits_vec[i] *= repeat_penalty;
            }
        }

        // Frequency penalty: proportional to count
        if frequency_penalty != 0.0 {
            logits_vec[i] -= frequency_penalty * count as f32;
        }

        // Presence penalty
        if presence_penalty != 0.0 {
            logits_vec[i] -= presence_penalty;
        }
    }

    // Write modified logits back to device
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(logits_vec.as_ptr() as *const u8, logits_vec.len() * 4)
    };
    R::copy_to_device(bytes, logits.ptr(), logits.device()).map_err(|e| {
        crate::error::Error::Numr(numr::error::Error::Internal(format!(
            "Failed to write back penalized logits: {}",
            e
        )))
    })?;

    Ok(())
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
        let token_ids = Tensor::<CpuRuntime>::from_slice(&[0i64], &[1], &device).expect("ids");
        let token_counts = Tensor::<CpuRuntime>::from_slice(&[1i32], &[1], &device).expect("cnt");

        let err = apply_sampling_penalties_impl::<CpuRuntime>(
            &client,
            &logits,
            &token_ids,
            &token_counts,
            1.1,
            0.0,
            0.0,
        )
        .expect_err("non-contiguous logits must be rejected");
        assert!(format!("{err}").contains("apply_sampling_penalties"));
    }

    /// A `[len]` i64 view with stride 2: every other element of a `[len, 2]`
    /// buffer of token ids.
    fn non_contiguous_ids(device: &CpuDevice) -> Tensor<CpuRuntime> {
        let data: Vec<i64> = vec![0, 9, 1, 9];
        let base = Tensor::<CpuRuntime>::from_slice(&data, &[2, 2], device).expect("base");
        let view = base.narrow(1, 0, 1).expect("narrow").squeeze(Some(1));
        assert!(!view.is_contiguous());
        view
    }

    /// A `[len]` i32 view with stride 2: every other element of a `[len, 2]`
    /// buffer of token counts.
    fn non_contiguous_counts(device: &CpuDevice) -> Tensor<CpuRuntime> {
        let data: Vec<i32> = vec![1, 9, 2, 9];
        let base = Tensor::<CpuRuntime>::from_slice(&data, &[2, 2], device).expect("base");
        let view = base.narrow(1, 0, 1).expect("narrow").squeeze(Some(1));
        assert!(!view.is_contiguous());
        view
    }

    #[test]
    fn rejects_non_contiguous_token_ids() {
        let (client, device) = cpu_setup();
        let logits =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, -2.0], &[2], &device).expect("logits");
        let token_ids = non_contiguous_ids(&device);
        let token_counts =
            Tensor::<CpuRuntime>::from_slice(&[1i32, 1], &[2], &device).expect("cnt");

        let err = apply_sampling_penalties_impl::<CpuRuntime>(
            &client,
            &logits,
            &token_ids,
            &token_counts,
            1.1,
            0.0,
            0.0,
        )
        .expect_err("non-contiguous token_ids must be rejected");
        assert!(format!("{err}").contains("token_ids"));
    }

    #[test]
    fn rejects_non_contiguous_token_counts() {
        let (client, device) = cpu_setup();
        let logits =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, -2.0], &[2], &device).expect("logits");
        let token_ids = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device).expect("ids");
        let token_counts = non_contiguous_counts(&device);

        let err = apply_sampling_penalties_impl::<CpuRuntime>(
            &client,
            &logits,
            &token_ids,
            &token_counts,
            1.1,
            0.0,
            0.0,
        )
        .expect_err("non-contiguous token_counts must be rejected");
        assert!(format!("{err}").contains("token_counts"));
    }

    #[test]
    fn unchanged_on_contiguous_logits() {
        let (client, device) = cpu_setup();
        let logits = Tensor::<CpuRuntime>::from_slice(&[1.0f32, -2.0, 3.0, 0.5], &[4], &device)
            .expect("logits");
        let token_ids = Tensor::<CpuRuntime>::from_slice(&[1i64], &[1], &device).expect("ids");
        let token_counts = Tensor::<CpuRuntime>::from_slice(&[2i32], &[1], &device).expect("cnt");

        apply_sampling_penalties_impl::<CpuRuntime>(
            &client,
            &logits,
            &token_ids,
            &token_counts,
            1.5,
            0.1,
            0.2,
        )
        .expect("contiguous logits must be accepted");

        let result: Vec<f32> = logits.to_vec();
        // Token 1's logit was -2.0: repetition penalty multiplies (negative
        // branch), then frequency and presence penalties subtract.
        let expected = -2.0 * 1.5 - 0.1 * 2.0 - 0.2;
        assert!((result[1] - expected).abs() < 1e-6);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[2], 3.0);
        assert_eq!(result[3], 0.5);
    }
}
