//! Flash Attention v2 parameter validation: Q/K/V shape and dtype checks, and
//! the resulting `AttentionParams` every launcher consumes.
//!
//! Split out of `flash_utils.rs`; the block config it resolves comes from
//! `flash_block_config.rs`.

use crate::error::{Error, Result};
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

use super::flash_block_config::block_config;

/// Validated attention parameters extracted from tensor shapes.
pub(super) struct AttentionParams {
    pub batch_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub seq_len_q: usize,
    pub seq_len_k: usize,
    pub head_dim: usize,
    pub block_m: usize,
    pub block_n: usize,
    /// Whether to use the small-memory kernel variant (_sm suffix)
    pub use_sm_kernel: bool,
}

/// Validate Q/K/V shapes and extract parameters.
pub(super) fn validate_qkv(
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<AttentionParams> {
    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();

    if q_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("expected 4D [B, H, S, D], got {}D", q_shape.len()),
        });
    }
    if k_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "k",
            reason: format!("expected 4D, got {}D", k_shape.len()),
        });
    }
    if v_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "v",
            reason: format!("expected 4D, got {}D", v_shape.len()),
        });
    }
    if q_shape[1] != num_heads {
        return Err(Error::InvalidArgument {
            arg: "num_heads",
            reason: format!("num_heads={} but q dim 1 is {}", num_heads, q_shape[1]),
        });
    }
    if k_shape[1] != num_kv_heads {
        return Err(Error::InvalidArgument {
            arg: "num_kv_heads",
            reason: format!(
                "num_kv_heads={} but k dim 1 is {}",
                num_kv_heads, k_shape[1]
            ),
        });
    }
    if q_shape[3] != head_dim || k_shape[3] != head_dim || v_shape[3] != head_dim {
        return Err(Error::InvalidArgument {
            arg: "head_dim",
            reason: format!(
                "head_dim={} but q.D={}, k.D={}, v.D={}",
                head_dim, q_shape[3], k_shape[3], v_shape[3]
            ),
        });
    }
    if q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0] {
        return Err(Error::InvalidArgument {
            arg: "batch_size",
            reason: format!(
                "batch mismatch: q.B={}, k.B={}, v.B={}",
                q_shape[0], k_shape[0], v_shape[0]
            ),
        });
    }
    if k_shape[2] != v_shape[2] {
        return Err(Error::InvalidArgument {
            arg: "v",
            reason: format!("k seq_len={} != v seq_len={}", k_shape[2], v_shape[2]),
        });
    }
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(Error::InvalidArgument {
            arg: "num_kv_heads",
            reason: format!(
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                num_heads, num_kv_heads
            ),
        });
    }

    let dtype = q.dtype();
    if k.dtype() != dtype || v.dtype() != dtype {
        return Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!(
                "Q/K/V dtype mismatch: Q={:?}, K={:?}, V={:?}",
                dtype,
                k.dtype(),
                v.dtype()
            ),
        });
    }
    if !q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() {
        return Err(Error::InvalidArgument {
            arg: "contiguity",
            reason: "Flash Attention requires contiguous Q, K, V tensors".into(),
        });
    }

    let elem_bytes = q.dtype().size_in_bytes();
    let seq_len_q = q_shape[2];
    let (block_m, block_n, use_sm_kernel) = block_config(head_dim, elem_bytes, seq_len_q)?;

    Ok(AttentionParams {
        batch_size: q_shape[0],
        num_heads,
        num_kv_heads,
        seq_len_q,
        seq_len_k: k_shape[2],
        head_dim,
        block_m,
        block_n,
        use_sm_kernel,
    })
}
