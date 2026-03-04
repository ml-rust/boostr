//! MoE routing: top-k gating and weight normalization.
//!
//! THE algorithm — same for all backends.
//! Composes numr primitives: softmax, topk, sum, div.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{ActivationOps, ReduceOps, ScalarOps, SortingOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Top-k routing: softmax(logits) → topk → normalize weights.
///
/// # Arguments
/// - `logits`: `[num_tokens, num_experts]`
/// - `k`: experts per token
///
/// # Returns
/// `(indices [num_tokens, k] I32, weights [num_tokens, k] F32)`
pub fn moe_top_k_routing_impl<R, C>(
    client: &C,
    logits: &Tensor<R>,
    k: usize,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + ActivationOps<R>
        + SortingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>,
{
    let shape = logits.shape();
    if shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!(
                "expected 2D [num_tokens, num_experts], got {}D",
                shape.len()
            ),
        });
    }
    let num_experts = shape[1];
    if k == 0 || k > num_experts {
        return Err(Error::InvalidArgument {
            arg: "k",
            reason: format!("k={} must be in [1, num_experts={}]", k, num_experts),
        });
    }

    // softmax over experts dimension
    let probs = client.softmax(logits, -1).map_err(Error::Numr)?;

    // top-k selection (topk returns I64 indices, cast to I32 for uniform pipeline)
    let (top_values, top_indices_i64) = client
        .topk(&probs, k, -1, true, true)
        .map_err(Error::Numr)?;
    let top_indices = client
        .cast(&top_indices_i64, DType::I32)
        .map_err(Error::Numr)?;

    // normalize weights to sum to 1 per token
    let weight_sum = client.sum(&top_values, &[1], true).map_err(Error::Numr)?;
    let normalized = client.div(&top_values, &weight_sum).map_err(Error::Numr)?;

    Ok((top_indices, normalized))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_top_k_routing_shapes() {
        let (client, device) = cpu_setup();
        let num_tokens = 6;
        let num_experts = 4;
        let k = 2;

        let logits_data: Vec<f32> = (0..num_tokens * num_experts)
            .map(|i| (i as f32 * 0.3).sin())
            .collect();
        let logits =
            Tensor::<CpuRuntime>::from_slice(&logits_data, &[num_tokens, num_experts], &device);

        let (indices, weights) = moe_top_k_routing_impl(&client, &logits, k).unwrap();

        assert_eq!(indices.shape(), &[num_tokens, k]);
        assert_eq!(weights.shape(), &[num_tokens, k]);

        // Weights should sum to ~1 per token
        let w_vec = weights.to_vec::<f32>();
        for t in 0..num_tokens {
            let sum: f32 = (0..k).map(|j| w_vec[t * k + j]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "token {} weights sum to {}, expected 1.0",
                t,
                sum
            );
        }
    }

    #[test]
    fn test_routing_invalid_k() {
        let (client, device) = cpu_setup();
        let logits = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 8], &[2, 4], &device);

        assert!(moe_top_k_routing_impl(&client, &logits, 0).is_err());
        assert!(moe_top_k_routing_impl(&client, &logits, 5).is_err());
    }
}
