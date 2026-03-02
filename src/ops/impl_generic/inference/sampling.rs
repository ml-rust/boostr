//! Generic implementation of sampling operations.
//!
//! Same algorithm on all backends. CPU backend has zero-overhead;
//! CUDA backend provides fused kernels that avoid D2H/H2D transfers.

use crate::error::Result;
use numr::ops::RandomOps;
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
    let mut logits_vec: Vec<f32> = logits.to_vec();

    // Temperature scaling
    if temperature != 1.0 {
        let inv_temp = 1.0 / temperature;
        for l in logits_vec.iter_mut() {
            *l *= inv_temp;
        }
    }

    // Softmax
    let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits_vec.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // Build sorted (index, prob) pairs
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Top-k filter
    if top_k > 0 && top_k < indexed.len() {
        indexed.truncate(top_k);
    }

    // Top-p filter
    if top_p < 1.0 {
        let mut cumsum = 0.0f32;
        let mut cutoff = indexed.len();
        for (i, (_, p)) in indexed.iter().enumerate() {
            cumsum += p;
            if cumsum > top_p {
                cutoff = i + 1;
                break;
            }
        }
        indexed.truncate(cutoff);
    }

    // Min-p filter
    if min_p > 0.0 && !indexed.is_empty() {
        let max_prob = indexed[0].1;
        let threshold = min_p * max_prob;
        indexed.retain(|(_, p)| *p >= threshold);
    }

    // Generate random value on-device via RandomOps, read back scalar
    let rand_tensor = client
        .rand(&[1], numr::dtype::DType::F32)
        .map_err(|e| crate::error::Error::Numr(e))?;
    let random_val: f32 = rand_tensor.to_vec::<f32>()[0];

    // Renormalize and sample
    let total: f32 = indexed.iter().map(|(_, p)| p).sum();
    let mut cumsum = 0.0f32;
    for (i, p) in &indexed {
        cumsum += p / total;
        if cumsum > random_val {
            return Ok(*i as u32);
        }
    }

    Ok(indexed.last().map(|(i, _)| *i as u32).unwrap_or(0))
}
