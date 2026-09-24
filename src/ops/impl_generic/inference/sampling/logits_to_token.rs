//! Fused logits-to-token — generic implementation.

use super::output::write_token_id;
use super::stochastic::{filter_distribution, sample_from_distribution};
use super::validate::{
    SamplingLogitsShape, validate_contiguous_arg, validate_sampling_logits, validate_scratch_vocab,
};
use crate::error::Result;
use crate::ops::traits::inference::sampling_scratch::SamplingScratch;
use numr::dtype::DType;
use numr::ops::RandomOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Fused logits-to-token — generic implementation.
///
/// Narrows to last seq position, casts to F32, applies penalties, then either
/// argmax (temperature == 0) or full stochastic sampling. Returns `[1]` I64 tensor.
///
/// Allocates a one-off [`SamplingScratch`] and delegates to
/// [`logits_to_token_into_impl`]. A caller that samples every step should
/// allocate its own scratch once and call [`logits_to_token_into_impl`]
/// directly instead.
#[allow(clippy::too_many_arguments)]
pub fn logits_to_token_impl<R: Runtime<DType = numr::dtype::DType>>(
    client: &R::Client,
    logits: &Tensor<R>,
    token_ids: &Tensor<R>,
    token_counts: &Tensor<R>,
    num_unique: usize,
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    seed: Option<u64>,
) -> Result<Tensor<R>>
where
    R::Client: numr::ops::RandomOps<R> + numr::ops::TypeConversionOps<R>,
{
    let vocab_size = logits.shape().last().copied().unwrap_or(1);
    let scratch = SamplingScratch::new(vocab_size, logits.device())?;
    logits_to_token_into_impl(
        client,
        logits,
        token_ids,
        token_counts,
        num_unique,
        repeat_penalty,
        frequency_penalty,
        presence_penalty,
        temperature,
        top_k,
        top_p,
        min_p,
        seed,
        &scratch,
    )?;
    Ok(scratch.output.clone())
}

/// Like [`logits_to_token_impl`], but writes the sampled token into
/// `scratch.output` instead of allocating.
///
/// `scratch.probs` is unused on this path: the generic implementation
/// filters the distribution on the host, in a `Vec<f32>`, rather than in a
/// device buffer. `scratch.output` is fully overwritten before this
/// function returns, so it carries nothing from a previous call.
#[allow(clippy::too_many_arguments)]
pub fn logits_to_token_into_impl<R: Runtime<DType = numr::dtype::DType>>(
    client: &R::Client,
    logits: &Tensor<R>,
    token_ids: &Tensor<R>,
    token_counts: &Tensor<R>,
    num_unique: usize,
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    seed: Option<u64>,
    scratch: &SamplingScratch<R>,
) -> Result<()>
where
    R::Client: numr::ops::RandomOps<R> + numr::ops::TypeConversionOps<R>,
{
    validate_sampling_logits(
        "logits_to_token",
        logits,
        SamplingLogitsShape::BatchSeqVocab,
    )?;

    // 1. Read logits at last seq position (cast to F32 if needed)
    let logits = if logits.dtype() != DType::F32 {
        use numr::ops::TypeConversionOps;
        client
            .cast(logits, DType::F32)
            .map_err(crate::error::Error::Numr)?
    } else {
        logits.clone()
    };
    let shape = logits.shape();
    let seq_len = shape[1];
    let vocab_size = shape[2];
    if seq_len == 0 || vocab_size == 0 {
        return Err(crate::error::Error::InvalidArgument {
            arg: "logits",
            reason: format!("seq_len and vocab_size must be > 0, got shape {:?}", shape),
        });
    }
    validate_scratch_vocab("logits_to_token", scratch, vocab_size)?;
    let all_logits: Vec<f32> = logits.to_vec();
    let offset = (seq_len - 1) * vocab_size;
    let mut last_logits: Vec<f32> = all_logits[offset..offset + vocab_size].to_vec();

    // 2. Apply penalties
    if num_unique > 0 {
        validate_contiguous_arg("logits_to_token", "token_ids", token_ids)?;
        validate_contiguous_arg("logits_to_token", "token_counts", token_counts)?;
        let ids_vec: Vec<i64> = token_ids.to_vec();
        let counts_vec: Vec<i32> = token_counts.to_vec();
        let penalty_count = num_unique.min(ids_vec.len()).min(counts_vec.len());

        for idx in 0..penalty_count {
            let token_id = ids_vec[idx] as usize;
            if token_id >= vocab_size {
                continue;
            }
            let count = counts_vec[idx];

            if repeat_penalty != 1.0 {
                if last_logits[token_id] > 0.0 {
                    last_logits[token_id] /= repeat_penalty;
                } else {
                    last_logits[token_id] *= repeat_penalty;
                }
            }
            if frequency_penalty != 0.0 {
                last_logits[token_id] -= frequency_penalty * count as f32;
            }
            if presence_penalty != 0.0 {
                last_logits[token_id] -= presence_penalty;
            }
        }
    }

    // 3. Greedy or stochastic
    let token_id = if temperature == 0.0 {
        // Argmax
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in last_logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        best_idx as i64
    } else {
        let indexed = filter_distribution(&mut last_logits, temperature, top_k, top_p, min_p);

        // Random value (seeded for reproducibility if seed is provided)
        let rand_tensor = if let Some(s) = seed {
            client
                .rand_seeded(&[1], numr::dtype::DType::F32, s)
                .map_err(crate::error::Error::Numr)?
        } else {
            client
                .rand(&[1], numr::dtype::DType::F32)
                .map_err(crate::error::Error::Numr)?
        };
        let random_val: f32 = rand_tensor
            .item::<f32>()
            .map_err(crate::error::Error::Numr)?;

        sample_from_distribution(&indexed, random_val) as i64
    };

    write_token_id("logits_to_token", token_id, &scratch.output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    /// A `[1, seq_len, vocab_size]` view whose vocab axis is narrowed out of
    /// a wider buffer, so it keeps the parent's stride instead of a
    /// contiguous one.
    fn non_contiguous_batch_seq_vocab(device: &CpuDevice) -> Tensor<CpuRuntime> {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let base = Tensor::<CpuRuntime>::from_slice(&data, &[1, 2, 8], device).expect("base");
        let view = base.narrow(2, 0, 4).expect("narrow");
        assert!(!view.is_contiguous());
        view
    }

    #[test]
    fn rejects_non_contiguous_logits() {
        let (client, device) = cpu_setup();
        let logits = non_contiguous_batch_seq_vocab(&device);
        let token_ids = Tensor::<CpuRuntime>::from_slice(&[0i64], &[1], &device).expect("ids");
        let token_counts = Tensor::<CpuRuntime>::from_slice(&[1i32], &[1], &device).expect("cnt");

        let err = logits_to_token_impl::<CpuRuntime>(
            &client,
            &logits,
            &token_ids,
            &token_counts,
            0,
            1.0,
            0.0,
            0.0,
            0.0,
            0,
            1.0,
            0.0,
            None,
        )
        .expect_err("non-contiguous logits must be rejected");
        assert!(format!("{err}").contains("logits_to_token"));
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
        let data: Vec<f32> = vec![0.0, 1.0, 0.0, 2.0];
        let logits = Tensor::<CpuRuntime>::from_slice(&data, &[1, 1, 4], &device).expect("logits");
        let token_ids = non_contiguous_ids(&device);
        let token_counts =
            Tensor::<CpuRuntime>::from_slice(&[1i32, 1], &[2], &device).expect("cnt");

        let err = logits_to_token_impl::<CpuRuntime>(
            &client,
            &logits,
            &token_ids,
            &token_counts,
            1,
            1.0,
            0.0,
            0.0,
            0.0,
            0,
            1.0,
            0.0,
            None,
        )
        .expect_err("non-contiguous token_ids must be rejected");
        assert!(format!("{err}").contains("token_ids"));
    }

    #[test]
    fn rejects_non_contiguous_token_counts() {
        let (client, device) = cpu_setup();
        let data: Vec<f32> = vec![0.0, 1.0, 0.0, 2.0];
        let logits = Tensor::<CpuRuntime>::from_slice(&data, &[1, 1, 4], &device).expect("logits");
        let token_ids = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device).expect("ids");
        let token_counts = non_contiguous_counts(&device);

        let err = logits_to_token_impl::<CpuRuntime>(
            &client,
            &logits,
            &token_ids,
            &token_counts,
            1,
            1.0,
            0.0,
            0.0,
            0.0,
            0,
            1.0,
            0.0,
            None,
        )
        .expect_err("non-contiguous token_counts must be rejected");
        assert!(format!("{err}").contains("token_counts"));
    }

    #[test]
    fn unchanged_on_contiguous_logits() {
        let (client, device) = cpu_setup();
        let data: Vec<f32> = vec![0.0, 1.0, 0.0, 2.0, 9.0, -1.0, 0.5, 3.0];
        let logits = Tensor::<CpuRuntime>::from_slice(&data, &[1, 2, 4], &device).expect("logits");
        let token_ids = Tensor::<CpuRuntime>::from_slice(&[0i64], &[1], &device).expect("ids");
        let token_counts = Tensor::<CpuRuntime>::from_slice(&[0i32], &[1], &device).expect("cnt");

        let token = logits_to_token_impl::<CpuRuntime>(
            &client,
            &logits,
            &token_ids,
            &token_counts,
            0,
            1.0,
            0.0,
            0.0,
            0.0, // temperature 0.0 => greedy argmax
            0,
            1.0,
            0.0,
            None,
        )
        .expect("contiguous logits must be accepted");
        // Last seq position is [9.0, -1.0, 0.5, 3.0]; argmax is index 0.
        let token_id: Vec<i64> = token.to_vec();
        assert_eq!(token_id, vec![0]);
    }

    /// A scratch reused across calls produces the same tokens as a fresh
    /// allocation per call, given the same inputs and the same seed —
    /// nothing leaks between calls through the shared output buffer.
    #[test]
    fn a_reused_scratch_matches_fresh_allocation_per_call() {
        let (client, device) = cpu_setup();
        let data: Vec<f32> = vec![0.0, 1.0, 0.0, 2.0, 9.0, -1.0, 0.5, 3.0];
        let logits = Tensor::<CpuRuntime>::from_slice(&data, &[1, 2, 4], &device).expect("logits");
        let token_ids = Tensor::<CpuRuntime>::from_slice(&[0i64], &[1], &device).expect("ids");
        let token_counts = Tensor::<CpuRuntime>::from_slice(&[1i32], &[1], &device).expect("cnt");
        let scratch = SamplingScratch::<CpuRuntime>::new(4, &device).expect("scratch");

        for seed in [1u64, 2u64, 3u64] {
            let fresh = logits_to_token_impl::<CpuRuntime>(
                &client,
                &logits,
                &token_ids,
                &token_counts,
                1,
                1.1,
                0.1,
                0.1,
                0.7,
                0,
                1.0,
                0.0,
                Some(seed),
            )
            .expect("fresh alloc path");
            logits_to_token_into_impl::<CpuRuntime>(
                &client,
                &logits,
                &token_ids,
                &token_counts,
                1,
                1.1,
                0.1,
                0.1,
                0.7,
                0,
                1.0,
                0.0,
                Some(seed),
                &scratch,
            )
            .expect("scratch path");
            let fresh_id: Vec<i64> = fresh.to_vec();
            let reused_id: Vec<i64> = scratch.output.to_vec();
            assert_eq!(reused_id, fresh_id);
        }
    }
}
