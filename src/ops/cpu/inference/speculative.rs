//! CPU implementation of `SpeculativeOps`: delegates to impl_generic.

use crate::error::Result;
use crate::ops::impl_generic::inference::speculative::speculative_verify_impl;
use crate::ops::traits::inference::speculative::SpeculativeOps;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl SpeculativeOps<CpuRuntime> for CpuClient {
    fn speculative_verify(
        &self,
        target_probs: &Tensor<CpuRuntime>,
        draft_probs: Option<&Tensor<CpuRuntime>>,
        draft_tokens: &Tensor<CpuRuntime>,
        uniforms: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        speculative_verify_impl::<CpuRuntime>(
            self,
            target_probs,
            draft_probs,
            draft_tokens,
            uniforms,
        )
    }
}
