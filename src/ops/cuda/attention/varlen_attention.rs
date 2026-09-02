//! Variable-length (ragged) attention CUDA launcher — trait wiring.
//!
//! Packed sequences with cu_seqlens indexing. Supports F32 and F16,
//! head dims 64, 128, and 256, with GQA (num_kv_heads <= num_heads).
//! Both forward and backward are implemented for all supported head dims.
//!
//! Kernel launch logic lives in `varlen_attention_fwd.rs` /
//! `varlen_attention_bwd.rs`; tile selection lives in
//! `varlen_attention_block_config.rs`. This file is wiring only, mirroring
//! `paged_attention.rs`'s split from `paged_attention_fwd.rs`/`paged_attention_bwd.rs`.

use crate::error::Result;
use crate::ops::traits::VarLenAttentionOps;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::varlen_attention_bwd::varlen_attention_bwd_impl;
use super::varlen_attention_fwd::varlen_attention_fwd_impl;

impl VarLenAttentionOps<CudaRuntime> for CudaClient {
    fn varlen_attention_fwd(
        &self,
        q: &Tensor<CudaRuntime>,
        k: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
        cu_seqlens_q: &Tensor<CudaRuntime>,
        cu_seqlens_k: &Tensor<CudaRuntime>,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        varlen_attention_fwd_impl(
            self,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            batch_size,
            num_heads,
            num_kv_heads,
            max_seqlen_q,
            max_seqlen_k,
            head_dim,
            causal,
        )
    }

    fn varlen_attention_bwd(
        &self,
        dout: &Tensor<CudaRuntime>,
        q: &Tensor<CudaRuntime>,
        k: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
        output: &Tensor<CudaRuntime>,
        lse: &Tensor<CudaRuntime>,
        cu_seqlens_q: &Tensor<CudaRuntime>,
        cu_seqlens_k: &Tensor<CudaRuntime>,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        varlen_attention_bwd_impl(
            self,
            dout,
            q,
            k,
            v,
            output,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            batch_size,
            num_heads,
            num_kv_heads,
            max_seqlen_q,
            max_seqlen_k,
            head_dim,
            causal,
        )
    }
}
