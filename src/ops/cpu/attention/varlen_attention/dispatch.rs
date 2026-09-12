//! `VarLenAttentionOps` for `CpuClient`: delegates to the forward and backward kernels.

use crate::error::Result;
use crate::ops::traits::VarLenAttentionOps;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use super::backward::varlen_attention_bwd_cpu;
use super::forward::varlen_attention_fwd_cpu;

impl VarLenAttentionOps<CpuRuntime> for CpuClient {
    fn varlen_attention_fwd(
        &self,
        q: &Tensor<CpuRuntime>,
        k: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
        cu_seqlens_q: &Tensor<CpuRuntime>,
        cu_seqlens_k: &Tensor<CpuRuntime>,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        _max_seqlen_q: usize,
        _max_seqlen_k: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        varlen_attention_fwd_cpu(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            batch_size,
            num_heads,
            num_kv_heads,
            head_dim,
            causal,
        )
    }

    fn varlen_attention_bwd(
        &self,
        dout: &Tensor<CpuRuntime>,
        q: &Tensor<CpuRuntime>,
        k: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
        output: &Tensor<CpuRuntime>,
        lse: &Tensor<CpuRuntime>,
        cu_seqlens_q: &Tensor<CpuRuntime>,
        cu_seqlens_k: &Tensor<CpuRuntime>,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        _max_seqlen_q: usize,
        _max_seqlen_k: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        varlen_attention_bwd_cpu(
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
            head_dim,
            causal,
        )
    }
}
