//! CUDA Paged Attention — vLLM-style non-contiguous KV cache
//!
//! Fused kernel — PRIMITIVE op. Block table indirection for KV blocks.
//! Supports: F32, F16, BF16, FP8E4M3, FP8E5M2. Head dimensions: 64, 128.
//!
//! Implementation is split across:
//! - `paged_attention_fwd.rs`: forward (standard + FP8)
//! - `paged_attention_bwd.rs`: backward
//! - `paged_decode.rs`: S_q=1 decode fast path

use crate::error::{Error, Result};
use crate::ops::traits::PagedAttentionOps;
use numr::dtype::DType;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::paged_attention_bwd::paged_attention_bwd_impl;
use super::paged_attention_fwd::{paged_attention_fwd_fp8_impl, paged_attention_fwd_impl};

/// Get block sizes for paged attention forward.
/// Uses smaller blocks that fit in 48KB shared memory.
pub(super) fn fwd_block_config(head_dim: usize, dtype: DType) -> Result<(usize, usize)> {
    match (dtype, head_dim) {
        // FP32: 4 bytes per element
        (DType::F32, 64) => Ok((64, 32)),  // (64+32+32)*64*4 = 32KB
        (DType::F32, 128) => Ok((32, 32)), // (32+32+32)*128*4 = 48KB
        // FP16/BF16: 2 bytes per element
        (DType::F16 | DType::BF16, 64) => Ok((64, 32)), // (64+32+32)*64*2 = 16KB
        (DType::F16 | DType::BF16, 128) => Ok((32, 32)), // (32+32+32)*128*2 = 24KB
        // FP8: uses FP32 smem for compute
        (DType::FP8E4M3 | DType::FP8E5M2, 64) => Ok((64, 32)), // (64+64)*64*4 = 32KB
        (DType::FP8E4M3 | DType::FP8E5M2, 128) => Ok((32, 32)), // (32+64)*128*4 = 49KB
        _ => Err(Error::InvalidArgument {
            arg: "head_dim",
            reason: format!(
                "unsupported head_dim={} for paged attention. Supported: 64, 128",
                head_dim
            ),
        }),
    }
}

/// Get block sizes for paged attention backward.
/// Backward needs more smem: Q + K + V + dO + O = (3*BLOCK_M + 2*BLOCK_N) * HD * dtype_size
pub(super) fn bwd_block_config(head_dim: usize, dtype: DType) -> Result<(usize, usize)> {
    match (dtype, head_dim) {
        (DType::F32, 64) => Ok((32, 32)),  // (96+64)*64*4 = 40KB
        (DType::F32, 128) => Ok((16, 16)), // (48+32)*128*4 = 40KB
        (DType::F16 | DType::BF16, 64) => Ok((64, 32)), // (192+64)*64*2 = 32KB
        (DType::F16 | DType::BF16, 128) => Ok((32, 32)), // (96+64)*128*2 = 40KB
        _ => Err(Error::InvalidArgument {
            arg: "head_dim",
            reason: format!(
                "unsupported head_dim={} for paged attention backward",
                head_dim
            ),
        }),
    }
}

impl PagedAttentionOps<CudaRuntime> for CudaClient {
    fn paged_attention_fwd(
        &self,
        q: &Tensor<CudaRuntime>,
        k_blocks: &Tensor<CudaRuntime>,
        v_blocks: &Tensor<CudaRuntime>,
        block_table: &Tensor<CudaRuntime>,
        num_heads: usize,
        num_kv_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
        block_size: usize,
        causal: bool,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        paged_attention_fwd_impl(
            self,
            q,
            k_blocks,
            v_blocks,
            block_table,
            num_heads,
            num_kv_heads,
            seq_len_q,
            seq_len_k,
            head_dim,
            block_size,
            causal,
        )
    }

    fn paged_attention_fwd_fp8(
        &self,
        q: &Tensor<CudaRuntime>,
        k_blocks: &Tensor<CudaRuntime>,
        v_blocks: &Tensor<CudaRuntime>,
        block_table: &Tensor<CudaRuntime>,
        num_heads: usize,
        num_kv_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
        block_size: usize,
        causal: bool,
        q_scale: f32,
        k_scale: f32,
        v_scale: f32,
        o_scale: f32,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        paged_attention_fwd_fp8_impl(
            self,
            q,
            k_blocks,
            v_blocks,
            block_table,
            num_heads,
            num_kv_heads,
            seq_len_q,
            seq_len_k,
            head_dim,
            block_size,
            causal,
            q_scale,
            k_scale,
            v_scale,
            o_scale,
        )
    }

    fn paged_attention_bwd(
        &self,
        dout: &Tensor<CudaRuntime>,
        q: &Tensor<CudaRuntime>,
        k_blocks: &Tensor<CudaRuntime>,
        v_blocks: &Tensor<CudaRuntime>,
        output: &Tensor<CudaRuntime>,
        lse: &Tensor<CudaRuntime>,
        block_table: &Tensor<CudaRuntime>,
        num_heads: usize,
        num_kv_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
        block_size: usize,
        causal: bool,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        paged_attention_bwd_impl(
            self,
            dout,
            q,
            k_blocks,
            v_blocks,
            output,
            lse,
            block_table,
            num_heads,
            num_kv_heads,
            seq_len_q,
            seq_len_k,
            head_dim,
            block_size,
            causal,
        )
    }
}
