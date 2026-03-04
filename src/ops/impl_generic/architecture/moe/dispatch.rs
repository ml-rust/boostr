//! MoE token permutation and unpermutation.
//!
//! THE algorithm — same for all backends.
//! Composes numr primitives: argsort, index_select, bincount, cumsum, scatter, cat.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps, TensorOps,
    TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

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

    // Flatten indices to [N*k] — already I32 from routing
    let flat_indices = indices.reshape(&[total]).map_err(Error::Numr)?.contiguous();

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

    // Convert counts to CSR offsets via GPU ops: [0, c0, c0+c1, ..., total]
    // bincount dtype varies by backend (I64 on CPU, I32 on WebGPU) — cast on GPU first.
    let counts_i32 = if counts.dtype() == DType::I32 {
        counts
    } else {
        client.cast(&counts, DType::I32).map_err(Error::Numr)?
    };
    let cumsum = client.cumsum(&counts_i32, 0).map_err(Error::Numr)?;
    // Prepend 0 to form CSR offsets [0, c0, c0+c1, ..., total]
    let zero = Tensor::<R>::zeros(&[1], DType::I32, device);
    let expert_offsets = client.cat(&[&zero, &cumsum], 0).map_err(Error::Numr)?;

    // Cast sort_perm to I32 (argsort returns I64 on CPU, I32 on WebGPU)
    let sort_perm_i32 = if sort_perm.dtype() != DType::I32 {
        client.cast(&sort_perm, DType::I32).map_err(Error::Numr)?
    } else {
        sort_perm
    };

    Ok((permuted, expert_offsets, sort_perm_i32))
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

    // Compute inverse permutation on GPU: inv_perm[sort_indices[i]] = i
    // Use scatter: dst[index[i]] = src[i] with src = arange(total).
    // arange is a static CPU-generated sequence of known size — not a GPU data download.
    let device = expert_output.device();
    let arange_data: Vec<i32> = (0..total as i32).collect();
    let values = Tensor::<R>::from_slice(&arange_data, &[total], device);
    let inv_perm_base = Tensor::<R>::zeros(&[total], DType::I32, device);
    let inv_perm = client
        .scatter(&inv_perm_base, 0, sort_indices, &values)
        .map_err(Error::Numr)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

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
}
