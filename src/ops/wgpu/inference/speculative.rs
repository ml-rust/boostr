//! WebGPU implementation of `SpeculativeOps`: delegates to impl_generic.
//!
//! `draft_tokens` and the result are I32, so every cast runs as a native WGSL shader.

use crate::error::Result;
use crate::ops::impl_generic::inference::speculative::speculative_verify_impl;
use crate::ops::traits::inference::speculative::SpeculativeOps;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl SpeculativeOps<WgpuRuntime> for WgpuClient {
    fn speculative_verify(
        &self,
        target_probs: &Tensor<WgpuRuntime>,
        draft_probs: Option<&Tensor<WgpuRuntime>>,
        draft_tokens: &Tensor<WgpuRuntime>,
        uniforms: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        speculative_verify_impl::<WgpuRuntime>(
            self,
            target_probs,
            draft_probs,
            draft_tokens,
            uniforms,
        )
    }
}
