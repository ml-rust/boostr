//! Generic MoE implementation
//!
//! THE algorithm — same for all backends.
//! Composes numr primitives: softmax, topk, index_select, scatter_reduce, matmul.

use crate::error::{Error, Result};
use crate::ops::traits::architecture::moe::MoEActivation;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, IndexingOps, MatmulOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TensorOps, TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Top-k routing: softmax(logits) → topk → normalize weights.
///
/// # Arguments
/// - `logits`: `[num_tokens, num_experts]`
/// - `k`: experts per token
///
/// # Returns
/// `(indices [num_tokens, k] (I64 on CPU/CUDA, I32 on WebGPU), weights [num_tokens, k] F32)`
pub fn moe_top_k_routing_impl<R, C>(
    client: &C,
    logits: &Tensor<R>,
    k: usize,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ActivationOps<R> + SortingOps<R> + ReduceOps<R> + ScalarOps<R>,
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

    // top-k selection
    let (top_values, top_indices) = client
        .topk(&probs, k, -1, true, true)
        .map_err(Error::Numr)?;

    // normalize weights to sum to 1 per token
    let weight_sum = client.sum(&top_values, &[1], true).map_err(Error::Numr)?;
    let normalized = client.div(&top_values, &weight_sum).map_err(Error::Numr)?;

    Ok((top_indices, normalized))
}

/// Permute tokens into expert-grouped order.
///
/// Creates a flat token list where all tokens for expert 0 come first,
/// then expert 1, etc. Tokens assigned to multiple experts (top-k > 1)
/// are duplicated.
///
/// # Returns
/// `(permuted [N*k, hidden], expert_offsets [E+1] I32, sort_indices [N*k] I32)`
pub fn moe_permute_tokens_impl<R, C>(
    client: &C,
    tokens: &Tensor<R>,
    indices: &Tensor<R>,
    num_experts: usize,
) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + IndexingOps<R>
        + ScalarOps<R>
        + SortingOps<R>
        + ShapeOps<R>
        + TensorOps<R>
        + TypeConversionOps<R>,
{
    let tok_shape = tokens.shape();
    let idx_shape = indices.shape();
    if tok_shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "tokens",
            reason: format!("expected 2D [num_tokens, hidden], got {}D", tok_shape.len()),
        });
    }
    if idx_shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "indices",
            reason: format!("expected 2D [num_tokens, k], got {}D", idx_shape.len()),
        });
    }

    let num_tokens = tok_shape[0];
    let _hidden_dim = tok_shape[1];
    let k = idx_shape[1];
    let total = num_tokens * k;
    let device = tokens.device();

    // Flatten indices to [N*k] and cast to I32 for backend compatibility
    // (topk returns I64 but WebGPU argsort only supports I32/F32)
    let flat_indices_i64 = indices.reshape(&[total]).map_err(Error::Numr)?.contiguous();
    let flat_indices = client
        .cast(&flat_indices_i64, DType::I32)
        .map_err(Error::Numr)?;

    // Sort by expert index to group tokens by expert
    // argsort gives us the permutation that sorts flat_indices (stable sort)
    let sort_perm = client
        .argsort(&flat_indices, 0, false)
        .map_err(Error::Numr)?;

    // Build token indices: for each (token_i, expert_j), the source token is token_i
    // token_source[i*k + j] = i → we need to expand tokens by k
    // Create token source indices [N*k] where entry i*k+j = i
    let token_source_data: Vec<i32> = (0..num_tokens)
        .flat_map(|t| std::iter::repeat_n(t as i32, k))
        .collect();
    let token_source = Tensor::<R>::from_slice(&token_source_data, &[total], device);

    // Reorder token_source by sort_perm to get the token index for each position in sorted order
    let sorted_token_indices = client
        .index_select(&token_source, 0, &sort_perm)
        .map_err(Error::Numr)?;

    // Gather tokens in sorted order: permuted[i] = tokens[sorted_token_indices[i]]
    let permuted = client
        .index_select(tokens, 0, &sorted_token_indices)
        .map_err(Error::Numr)?;

    // Compute expert_offsets via bincount
    let sorted_expert_ids = client
        .index_select(&flat_indices, 0, &sort_perm)
        .map_err(Error::Numr)?;
    let counts = client
        .bincount(&sorted_expert_ids, None, num_experts)
        .map_err(Error::Numr)?;

    // Convert counts to CSR offsets: [0, count_0, count_0+count_1, ...]
    let counts_vec = counts.to_vec::<i32>();
    let mut offsets = Vec::with_capacity(num_experts + 1);
    offsets.push(0i32);
    let mut cumsum = 0i32;
    for &c in &counts_vec {
        cumsum += c;
        offsets.push(cumsum);
    }
    let expert_offsets = Tensor::<R>::from_slice(&offsets, &[num_experts + 1], device);

    Ok((permuted, expert_offsets, sort_perm))
}

/// Unpermute expert outputs back to original token order with weighted combination.
///
/// For each original token, gathers its k expert outputs and computes
/// a weighted sum using the routing weights.
pub fn moe_unpermute_tokens_impl<R, C>(
    client: &C,
    expert_output: &Tensor<R>,
    sort_indices: &Tensor<R>,
    weights: &Tensor<R>,
    num_tokens: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + IndexingOps<R> + ShapeOps<R> + ScalarOps<R> + ReduceOps<R> + TensorOps<R>,
{
    let out_shape = expert_output.shape();
    if out_shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "expert_output",
            reason: format!("expected 2D [N*k, hidden], got {}D", out_shape.len()),
        });
    }

    let total = out_shape[0];
    let hidden_dim = out_shape[1];
    let w_shape = weights.shape();
    if w_shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "weights",
            reason: format!("expected 2D [num_tokens, k], got {}D", w_shape.len()),
        });
    }
    let k = w_shape[1];

    if num_tokens * k != total {
        return Err(Error::InvalidArgument {
            arg: "num_tokens",
            reason: format!(
                "num_tokens({}) * k({}) = {} != expert_output rows({})",
                num_tokens,
                k,
                num_tokens * k,
                total
            ),
        });
    }

    // Compute inverse permutation: inv_perm[sort_indices[i]] = i
    // This tells us where each original (token, expert) pair ended up
    let device = expert_output.device();

    // Build inv_perm matching sort_indices dtype (I64 on CPU, I32 on WebGPU)
    let inv_perm = if sort_indices.dtype() == DType::I64 {
        let sort_vec = sort_indices.to_vec::<i64>();
        let mut inv_perm_data = vec![0i64; total];
        for (i, &si) in sort_vec.iter().enumerate() {
            inv_perm_data[si as usize] = i as i64;
        }
        Tensor::<R>::from_slice(&inv_perm_data, &[total], device)
    } else {
        let sort_vec = sort_indices.to_vec::<i32>();
        let mut inv_perm_data = vec![0i32; total];
        for (i, &si) in sort_vec.iter().enumerate() {
            inv_perm_data[si as usize] = i as i32;
        }
        Tensor::<R>::from_slice(&inv_perm_data, &[total], device)
    };

    // Gather back to original order: unsorted[i] = expert_output[inv_perm[i]]
    let unsorted = client
        .index_select(expert_output, 0, &inv_perm)
        .map_err(Error::Numr)?;

    // unsorted is now [N*k, hidden] in original (token, expert) order
    // Reshape to [N, k, hidden]
    let reshaped = unsorted
        .reshape(&[num_tokens, k, hidden_dim])
        .map_err(Error::Numr)?;

    // weights: [N, k] → [N, k, 1] for broadcasting
    let w_expanded = weights.reshape(&[num_tokens, k, 1]).map_err(Error::Numr)?;

    // Weighted sum: sum(reshaped * weights, dim=1) → [N, hidden]
    let weighted = client.mul(&reshaped, &w_expanded).map_err(Error::Numr)?;
    let output = client.sum(&weighted, &[1], false).map_err(Error::Numr)?;

    Ok(output)
}

/// Grouped GEMM: per-expert matmul on contiguous token groups.
///
/// For each expert e, computes:
///   output[offsets[e]..offsets[e+1]] = tokens[offsets[e]..offsets[e+1]] @ weights[e]
pub fn moe_grouped_gemm_impl<R, C>(
    client: &C,
    permuted_tokens: &Tensor<R>,
    expert_weights: &Tensor<R>,
    expert_offsets: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + MatmulOps<R> + ShapeOps<R> + TensorOps<R>,
{
    validate_grouped_gemm_args(permuted_tokens, expert_weights, expert_offsets)?;

    let ew_shape = expert_weights.shape();
    let num_experts = ew_shape[0];
    let out_dim = ew_shape[2];
    let total_tokens = permuted_tokens.shape()[0];

    let offsets = expert_offsets.to_vec::<i32>();

    let mut expert_outputs: Vec<Tensor<R>> = Vec::with_capacity(num_experts);

    for e in 0..num_experts {
        let start = offsets[e] as usize;
        let end = offsets[e + 1] as usize;
        let count = end - start;

        if count == 0 {
            continue;
        }

        let expert_tokens = permuted_tokens
            .narrow(0, start, count)
            .map_err(Error::Numr)?
            .contiguous();

        let weight = expert_weights
            .narrow(0, e, 1)
            .map_err(Error::Numr)?
            .contiguous()
            .reshape(&[ew_shape[1], out_dim])
            .map_err(Error::Numr)?;

        // matmul: [count, in_dim] @ [in_dim, out_dim] → [count, out_dim]
        let result = client
            .matmul(&expert_tokens, &weight)
            .map_err(Error::Numr)?;
        expert_outputs.push(result);
    }

    if expert_outputs.is_empty() {
        // All experts have zero tokens — return empty
        let device = permuted_tokens.device();
        return Ok(Tensor::<R>::empty(
            &[total_tokens, out_dim],
            permuted_tokens.dtype(),
            device,
        ));
    }

    // Concatenate all expert outputs along dim 0
    let refs: Vec<&Tensor<R>> = expert_outputs.iter().collect();
    let output = client.cat(&refs, 0).map_err(Error::Numr)?;

    Ok(output)
}

/// Fused grouped GEMM with activation.
///
/// Same as `moe_grouped_gemm_impl` but applies activation after each expert's matmul.
pub fn moe_grouped_gemm_fused_impl<R, C>(
    client: &C,
    permuted_tokens: &Tensor<R>,
    expert_weights: &Tensor<R>,
    expert_offsets: &Tensor<R>,
    activation: MoEActivation,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + MatmulOps<R> + ShapeOps<R> + TensorOps<R> + ActivationOps<R>,
{
    validate_grouped_gemm_args(permuted_tokens, expert_weights, expert_offsets)?;

    let ew_shape = expert_weights.shape();
    let num_experts = ew_shape[0];
    let out_dim = ew_shape[2];
    let total_tokens = permuted_tokens.shape()[0];

    let offsets = expert_offsets.to_vec::<i32>();

    let mut expert_outputs: Vec<Tensor<R>> = Vec::with_capacity(num_experts);

    for e in 0..num_experts {
        let start = offsets[e] as usize;
        let end = offsets[e + 1] as usize;
        let count = end - start;

        if count == 0 {
            continue;
        }

        let expert_tokens = permuted_tokens
            .narrow(0, start, count)
            .map_err(Error::Numr)?
            .contiguous();

        let weight = expert_weights
            .narrow(0, e, 1)
            .map_err(Error::Numr)?
            .contiguous()
            .reshape(&[ew_shape[1], out_dim])
            .map_err(Error::Numr)?;

        let result = client
            .matmul(&expert_tokens, &weight)
            .map_err(Error::Numr)?;

        // Apply activation
        let activated = match activation {
            MoEActivation::SiLU => client.silu(&result).map_err(Error::Numr)?,
            MoEActivation::GeLU => client.gelu(&result).map_err(Error::Numr)?,
            MoEActivation::None => result,
        };

        expert_outputs.push(activated);
    }

    if expert_outputs.is_empty() {
        let device = permuted_tokens.device();
        return Ok(Tensor::<R>::empty(
            &[total_tokens, out_dim],
            permuted_tokens.dtype(),
            device,
        ));
    }

    let refs: Vec<&Tensor<R>> = expert_outputs.iter().collect();
    let output = client.cat(&refs, 0).map_err(Error::Numr)?;

    Ok(output)
}

fn validate_grouped_gemm_args<R: Runtime>(
    permuted_tokens: &Tensor<R>,
    expert_weights: &Tensor<R>,
    expert_offsets: &Tensor<R>,
) -> Result<()> {
    let pt_shape = permuted_tokens.shape();
    let ew_shape = expert_weights.shape();
    let eo_shape = expert_offsets.shape();

    if pt_shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "permuted_tokens",
            reason: format!("expected 2D [total, in_dim], got {}D", pt_shape.len()),
        });
    }
    if ew_shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "expert_weights",
            reason: format!(
                "expected 3D [num_experts, in_dim, out_dim], got {}D",
                ew_shape.len()
            ),
        });
    }
    if eo_shape.len() != 1 {
        return Err(Error::InvalidArgument {
            arg: "expert_offsets",
            reason: format!("expected 1D [num_experts+1], got {}D", eo_shape.len()),
        });
    }
    if pt_shape[1] != ew_shape[1] {
        return Err(Error::InvalidArgument {
            arg: "expert_weights",
            reason: format!(
                "in_dim mismatch: tokens has {}, weights has {}",
                pt_shape[1], ew_shape[1]
            ),
        });
    }
    if eo_shape[0] != ew_shape[0] + 1 {
        return Err(Error::InvalidArgument {
            arg: "expert_offsets",
            reason: format!(
                "expected {} entries (num_experts+1), got {}",
                ew_shape[0] + 1,
                eo_shape[0]
            ),
        });
    }
    Ok(())
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
    fn test_permute_unpermute_roundtrip() {
        let (client, device) = cpu_setup();
        let num_tokens = 4;
        let hidden_dim = 8;
        let num_experts = 3;
        let k = 2;

        let tokens_data: Vec<f32> = (0..num_tokens * hidden_dim)
            .map(|i| i as f32 * 0.1)
            .collect();
        let tokens =
            Tensor::<CpuRuntime>::from_slice(&tokens_data, &[num_tokens, hidden_dim], &device);

        // Create expert indices [num_tokens, k]
        let indices_data: Vec<i32> = vec![0, 1, 2, 0, 1, 2, 0, 1];
        let indices = Tensor::<CpuRuntime>::from_slice(&indices_data, &[num_tokens, k], &device);

        // Equal weights
        let weights_data: Vec<f32> = vec![0.5; num_tokens * k];
        let weights = Tensor::<CpuRuntime>::from_slice(&weights_data, &[num_tokens, k], &device);

        let (permuted, offsets, sort_indices) =
            moe_permute_tokens_impl(&client, &tokens, &indices, num_experts).unwrap();

        assert_eq!(permuted.shape(), &[num_tokens * k, hidden_dim]);
        assert_eq!(offsets.shape(), &[num_experts + 1]);

        // Unpermute with identity-like weights should recover tokens
        // (each token appears k times with weight 1/k, so weighted sum = original)
        let result =
            moe_unpermute_tokens_impl(&client, &permuted, &sort_indices, &weights, num_tokens)
                .unwrap();

        assert_eq!(result.shape(), &[num_tokens, hidden_dim]);

        let result_vec = result.to_vec::<f32>();
        for (i, (&got, &expected)) in result_vec.iter().zip(tokens_data.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-5,
                "roundtrip mismatch at {}: got {}, expected {}",
                i,
                got,
                expected
            );
        }
    }

    #[test]
    fn test_grouped_gemm_shapes() {
        let (client, device) = cpu_setup();
        let num_experts = 3;
        let in_dim = 4;
        let out_dim = 6;
        let total_tokens = 8;

        let tokens_data: Vec<f32> = (0..total_tokens * in_dim)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let tokens =
            Tensor::<CpuRuntime>::from_slice(&tokens_data, &[total_tokens, in_dim], &device);

        let weights_data: Vec<f32> = (0..num_experts * in_dim * out_dim)
            .map(|i| (i as f32 * 0.05).cos() * 0.1)
            .collect();
        let weights = Tensor::<CpuRuntime>::from_slice(
            &weights_data,
            &[num_experts, in_dim, out_dim],
            &device,
        );

        let offsets = Tensor::<CpuRuntime>::from_slice(&[0i32, 3, 5, 8], &[4], &device);

        let output = moe_grouped_gemm_impl(&client, &tokens, &weights, &offsets).unwrap();
        assert_eq!(output.shape(), &[total_tokens, out_dim]);
    }

    #[test]
    fn test_grouped_gemm_fused_silu() {
        let (client, device) = cpu_setup();
        const NUM_EXPERTS: usize = 2;
        const IN_DIM: usize = 4;
        const OUT_DIM: usize = 4;
        const TOTAL: usize = 4;

        let tokens =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; TOTAL * IN_DIM], &[TOTAL, IN_DIM], &device);
        let weights = Tensor::<CpuRuntime>::from_slice(
            &[0.1f32; NUM_EXPERTS * IN_DIM * OUT_DIM],
            &[NUM_EXPERTS, IN_DIM, OUT_DIM],
            &device,
        );
        let offsets = Tensor::<CpuRuntime>::from_slice(&[0i32, 2, 4], &[3], &device);

        let plain = moe_grouped_gemm_impl(&client, &tokens, &weights, &offsets).unwrap();
        let fused =
            moe_grouped_gemm_fused_impl(&client, &tokens, &weights, &offsets, MoEActivation::SiLU)
                .unwrap();

        // Fused should equal silu(plain) element-wise
        let plain_vec = plain.to_vec::<f32>();
        let fused_vec = fused.to_vec::<f32>();
        for (i, (&p, &f)) in plain_vec.iter().zip(fused_vec.iter()).enumerate() {
            let expected = p * (1.0 / (1.0 + (-p).exp())); // silu
            assert!(
                (f - expected).abs() < 1e-5,
                "fused silu mismatch at {}: got {}, expected {}",
                i,
                f,
                expected
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
