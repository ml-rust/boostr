//! Generic speculative decoding verification implementation
//!
//! THE algorithm — same for all backends.
//! Composes numr primitives for probability manipulation.
//! Sequential token verification (inherently serial per batch element).
//!
//! `to_vec()` here transfers probability and random data to CPU for the serial
//! accept/reject loop. All backends ultimately need CPU-side results
//! (Vec<VerificationResult>), so this is the correct place to do transfers.
//! Random values are generated via `philox_uniform` — same algorithm on CPU,
//! CUDA, and WebGPU — guaranteeing identical results given the same seed.

use crate::error::{Error, Result};
use crate::ops::traits::inference::speculative::VerificationResult;
use numr::dtype::DType;
use numr::ops::{AdvancedRandomOps, BinaryOps, ScalarOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Sample from a probability distribution using a pre-drawn uniform variate.
///
/// `r` must be in `[0, sum(probs))`. Falls back to index 0 if all probs are zero.
fn sample_with_uniform(probs: &[f32], r: f32) -> u32 {
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Verify speculative tokens: accept/reject + residual sampling.
///
/// Uses `philox_uniform` for reproducible random values — identical across
/// CPU, CUDA, and WebGPU backends given the same seed.
///
/// # Arguments
/// - `draft_probs`: `[batch, K, vocab_size]` F32
/// - `target_probs`: `[batch, K+1, vocab_size]` F32
/// - `draft_tokens`: `[batch, K]` I32
/// - `seed`: RNG seed (Philox key)
pub fn verify_speculative_tokens_impl<R, C>(
    client: &C,
    draft_probs: &Tensor<R>,
    target_probs: &Tensor<R>,
    draft_tokens: &Tensor<R>,
    seed: u64,
) -> Result<Vec<VerificationResult>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + AdvancedRandomOps<R>,
{
    // Validate shapes
    let dp_shape = draft_probs.shape();
    let tp_shape = target_probs.shape();
    let dt_shape = draft_tokens.shape();

    if dp_shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "draft_probs",
            reason: format!("expected 3D [batch, K, vocab], got {}D", dp_shape.len()),
        });
    }
    if tp_shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "target_probs",
            reason: format!("expected 3D [batch, K+1, vocab], got {}D", tp_shape.len()),
        });
    }
    if dt_shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "draft_tokens",
            reason: format!("expected 2D [batch, K], got {}D", dt_shape.len()),
        });
    }

    let batch_size = dp_shape[0];
    let num_draft = dp_shape[1];
    let vocab_size = dp_shape[2];

    if tp_shape[0] != batch_size || tp_shape[1] != num_draft + 1 || tp_shape[2] != vocab_size {
        return Err(Error::InvalidArgument {
            arg: "target_probs",
            reason: format!(
                "expected [{}, {}, {}], got {:?}",
                batch_size,
                num_draft + 1,
                vocab_size,
                tp_shape
            ),
        });
    }
    if dt_shape[0] != batch_size || dt_shape[1] != num_draft {
        return Err(Error::InvalidArgument {
            arg: "draft_tokens",
            reason: format!(
                "expected [{}, {}], got {:?}",
                batch_size, num_draft, dt_shape
            ),
        });
    }

    if draft_probs.dtype() != DType::F32 {
        return Err(Error::InvalidArgument {
            arg: "draft_probs",
            reason: format!("expected F32, got {:?}", draft_probs.dtype()),
        });
    }
    if draft_tokens.dtype() != DType::I32 {
        return Err(Error::InvalidArgument {
            arg: "draft_tokens",
            reason: format!("expected I32, got {:?}", draft_tokens.dtype()),
        });
    }

    // Generate uniform randoms via Philox4x32-10 — same algorithm on CPU, CUDA, WebGPU.
    // Layout: [batch, K+1] — first K values for accept/reject, last for bonus token.
    let randoms_tensor = client
        .philox_uniform(&[batch_size, num_draft + 1], seed, 0, DType::F32)
        .map_err(Error::Numr)?;

    let dp_data = draft_probs.contiguous().to_vec::<f32>();
    let tp_data = target_probs.contiguous().to_vec::<f32>();
    let dt_data = draft_tokens.contiguous().to_vec::<i32>();
    let rand_data = randoms_tensor.to_vec::<f32>();

    let mut results = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let rand_base = b * (num_draft + 1);
        let mut accepted_tokens = Vec::new();
        let mut first_rejection_pos = -1i32;

        // Sequential accept/reject using Philox randoms
        for pos in 0..num_draft {
            let draft_token = dt_data[b * num_draft + pos] as usize;

            let draft_prob = dp_data[b * num_draft * vocab_size + pos * vocab_size + draft_token];
            let target_prob =
                tp_data[b * (num_draft + 1) * vocab_size + pos * vocab_size + draft_token];

            let accept_prob = if draft_prob > 0.0 {
                (target_prob / draft_prob).min(1.0)
            } else {
                1.0
            };

            if rand_data[rand_base + pos] < accept_prob {
                accepted_tokens.push(draft_token as u32);
            } else {
                first_rejection_pos = pos as i32;
                break;
            }
        }

        // Sample bonus token using the K+1-th Philox random (independent of accept/reject)
        let bonus_r = rand_data[rand_base + num_draft];
        let bonus_token = if first_rejection_pos >= 0 {
            // Rejected: sample from residual max(0, p_target - p_draft)
            let pos = first_rejection_pos as usize;
            let dp_offset = b * num_draft * vocab_size + pos * vocab_size;
            let tp_offset = b * (num_draft + 1) * vocab_size + pos * vocab_size;

            let residual: Vec<f32> = (0..vocab_size)
                .map(|v| (tp_data[tp_offset + v] - dp_data[dp_offset + v]).max(0.0))
                .collect();

            let sum: f32 = residual.iter().sum();
            if sum > 1e-8 {
                sample_with_uniform(&residual, bonus_r * sum)
            } else {
                let target_slice = &tp_data[tp_offset..tp_offset + vocab_size];
                let target_sum: f32 = target_slice.iter().sum();
                sample_with_uniform(target_slice, bonus_r * target_sum)
            }
        } else {
            // All accepted: sample from target's K+1 position
            let tp_offset = b * (num_draft + 1) * vocab_size + num_draft * vocab_size;
            let target_slice = &tp_data[tp_offset..tp_offset + vocab_size];
            let target_sum: f32 = target_slice.iter().sum();
            sample_with_uniform(target_slice, bonus_r * target_sum)
        };

        results.push(VerificationResult {
            num_accepted: accepted_tokens.len(),
            accepted_tokens,
            bonus_token,
            first_rejection_pos,
        });
    }

    Ok(results)
}

/// Compute acceptance and residual probabilities element-wise.
///
/// - acceptance_prob[i] = min(1, target[i] / draft[i])
/// - residual_prob[i] = max(0, target[i] - draft[i])
pub fn compute_acceptance_probs_impl<R, C>(
    client: &C,
    draft_probs: &Tensor<R>,
    target_probs: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + ScalarOps<R>,
{
    let dp_shape = draft_probs.shape();
    let tp_shape = target_probs.shape();

    if dp_shape != tp_shape {
        return Err(Error::InvalidArgument {
            arg: "target_probs",
            reason: format!(
                "shape mismatch: draft {:?} vs target {:?}",
                dp_shape, tp_shape
            ),
        });
    }

    // acceptance = min(1, target / draft)
    // safe_draft = max(draft, eps) to avoid division by zero
    let eps = Tensor::<R>::full_scalar(dp_shape, DType::F32, 1e-10, draft_probs.device());
    let ones = Tensor::<R>::full_scalar(dp_shape, DType::F32, 1.0, draft_probs.device());
    let zeros = Tensor::<R>::full_scalar(dp_shape, DType::F32, 0.0, draft_probs.device());

    let safe_draft = client.maximum(draft_probs, &eps).map_err(Error::Numr)?;
    let ratio = client.div(target_probs, &safe_draft).map_err(Error::Numr)?;
    let acceptance = client.minimum(&ratio, &ones).map_err(Error::Numr)?;

    // residual = max(0, target - draft)
    let diff = client.sub(target_probs, draft_probs).map_err(Error::Numr)?;
    let residual = client.maximum(&diff, &zeros).map_err(Error::Numr)?;

    Ok((acceptance, residual))
}

/// Compute expected tokens per verification step.
///
/// expected[b] = sum_{i=0}^{K-1} prod_{j=0}^{i} rates[b,j] + 1
pub fn compute_expected_tokens_impl<R, C>(
    _client: &C,
    acceptance_rates: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>,
{
    let shape = acceptance_rates.shape();
    if shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "acceptance_rates",
            reason: format!("expected 2D [batch, K], got {}D", shape.len()),
        });
    }

    let batch_size = shape[0];
    let k = shape[1];

    let rates = acceptance_rates.contiguous().to_vec::<f32>();
    let mut expected = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let mut cumulative_prob = 1.0f32;
        let mut exp_tokens = 0.0f32;

        for i in 0..k {
            cumulative_prob *= rates[b * k + i];
            exp_tokens += cumulative_prob;
        }

        // +1 for bonus token (always sampled)
        expected.push(exp_tokens + 1.0);
    }

    Ok(Tensor::<R>::from_slice(
        &expected,
        &[batch_size],
        acceptance_rates.device(),
    ))
}
