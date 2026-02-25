//! CPU implementation of MoEOps — delegates to impl_generic

use crate::error::Result;
use crate::ops::impl_generic::architecture::moe::{
    moe_grouped_gemm_fused_impl, moe_grouped_gemm_impl, moe_permute_tokens_impl,
    moe_top_k_routing_impl, moe_unpermute_tokens_impl,
};
use crate::ops::traits::architecture::moe::{MoEActivation, MoEOps};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl MoEOps<CpuRuntime> for CpuClient {
    fn moe_top_k_routing(
        &self,
        logits: &Tensor<CpuRuntime>,
        k: usize,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        moe_top_k_routing_impl(self, logits, k)
    }

    fn moe_permute_tokens(
        &self,
        tokens: &Tensor<CpuRuntime>,
        indices: &Tensor<CpuRuntime>,
        num_experts: usize,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        moe_permute_tokens_impl(self, tokens, indices, num_experts)
    }

    fn moe_unpermute_tokens(
        &self,
        expert_output: &Tensor<CpuRuntime>,
        sort_indices: &Tensor<CpuRuntime>,
        weights: &Tensor<CpuRuntime>,
        num_tokens: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        moe_unpermute_tokens_impl(self, expert_output, sort_indices, weights, num_tokens)
    }

    fn moe_grouped_gemm(
        &self,
        permuted_tokens: &Tensor<CpuRuntime>,
        expert_weights: &Tensor<CpuRuntime>,
        expert_offsets: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        moe_grouped_gemm_impl(self, permuted_tokens, expert_weights, expert_offsets)
    }

    fn moe_grouped_gemm_fused(
        &self,
        permuted_tokens: &Tensor<CpuRuntime>,
        expert_weights: &Tensor<CpuRuntime>,
        expert_offsets: &Tensor<CpuRuntime>,
        activation: MoEActivation,
    ) -> Result<Tensor<CpuRuntime>> {
        moe_grouped_gemm_fused_impl(
            self,
            permuted_tokens,
            expert_weights,
            expert_offsets,
            activation,
        )
    }
}
