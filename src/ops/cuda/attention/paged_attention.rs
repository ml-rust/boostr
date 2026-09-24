//! CUDA Paged Attention — vLLM-style non-contiguous KV cache
//!
//! Fused kernel — PRIMITIVE op. Block table indirection for KV blocks.
//! Supports: F32, F16, BF16, FP8E4M3, FP8E5M2. Head dimensions: 64, 128.
//!
//! Implementation is split across:
//! - `paged_attention_fwd_block_config.rs`: forward tile selection and
//!   shared-memory sizing
//! - `paged_attention_bwd_block_config.rs`: backward tile selection and
//!   shared-memory sizing
//! - `paged_attention_fwd.rs`: forward (standard)
//! - `paged_attention_fwd_fp8.rs`: forward (FP8)
//! - `paged_attention_bwd.rs`: backward
//! - `paged_decode.rs`: S_q=1 decode fast path
//!
//! A per-row `KvSeqLen::PerRow` key end reaches the decode fast path only. The
//! tiled prefill kernels and the FP8 kernel read one key length for the whole
//! batch and refuse it.

use crate::error::Result;
use crate::ops::impl_generic::attention::reject_per_row_kv_seq_len;
use crate::ops::traits::{KvSeqLen, PagedAttentionOps};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::paged_attention_bwd::paged_attention_bwd_impl;
use super::paged_attention_fwd::paged_attention_fwd_impl;
use super::paged_attention_fwd_fp8::paged_attention_fwd_fp8_impl;

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
        kv_seq_len: Option<KvSeqLen<'_, CudaRuntime>>,
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
            kv_seq_len,
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
        kv_seq_len: Option<KvSeqLen<'_, CudaRuntime>>,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        reject_per_row_kv_seq_len(kv_seq_len, "CUDA")?;
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
