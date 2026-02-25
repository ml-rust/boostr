//! WebGPU implementation of MoEOps
//!
//! Delegates to impl_generic for all operations (which composes numr primitives).
//! WebGPU supports all required primitives (softmax, topk, index_select, matmul).
//! F32 only.

use crate::error::Result;
use crate::ops::impl_generic::architecture::moe::{
    moe_grouped_gemm_fused_impl, moe_grouped_gemm_impl, moe_permute_tokens_impl,
    moe_top_k_routing_impl, moe_unpermute_tokens_impl,
};
use crate::ops::traits::architecture::moe::{MoEActivation, MoEOps};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl MoEOps<WgpuRuntime> for WgpuClient {
    fn moe_top_k_routing(
        &self,
        logits: &Tensor<WgpuRuntime>,
        k: usize,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        moe_top_k_routing_impl(self, logits, k)
    }

    fn moe_permute_tokens(
        &self,
        tokens: &Tensor<WgpuRuntime>,
        indices: &Tensor<WgpuRuntime>,
        num_experts: usize,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        moe_permute_tokens_impl(self, tokens, indices, num_experts)
    }

    fn moe_unpermute_tokens(
        &self,
        expert_output: &Tensor<WgpuRuntime>,
        sort_indices: &Tensor<WgpuRuntime>,
        weights: &Tensor<WgpuRuntime>,
        num_tokens: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        moe_unpermute_tokens_impl(self, expert_output, sort_indices, weights, num_tokens)
    }

    fn moe_grouped_gemm(
        &self,
        permuted_tokens: &Tensor<WgpuRuntime>,
        expert_weights: &Tensor<WgpuRuntime>,
        expert_offsets: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        moe_grouped_gemm_impl(self, permuted_tokens, expert_weights, expert_offsets)
    }

    fn moe_grouped_gemm_fused(
        &self,
        permuted_tokens: &Tensor<WgpuRuntime>,
        expert_weights: &Tensor<WgpuRuntime>,
        expert_offsets: &Tensor<WgpuRuntime>,
        activation: MoEActivation,
    ) -> Result<Tensor<WgpuRuntime>> {
        moe_grouped_gemm_fused_impl(
            self,
            permuted_tokens,
            expert_weights,
            expert_offsets,
            activation,
        )
    }
}
