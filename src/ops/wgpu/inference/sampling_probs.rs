//! WebGPU implementation of SamplingProbsOps — delegates to impl_generic.

use crate::error::Result;
use crate::ops::impl_generic::inference::sampling::sampling_probs_impl;
use crate::ops::traits::inference::sampling_probs::{FilterRule, SamplingProbsOps};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl SamplingProbsOps<WgpuRuntime> for WgpuClient {
    fn sampling_probs(
        &self,
        logits: &Tensor<WgpuRuntime>,
        rule: &FilterRule,
    ) -> Result<Tensor<WgpuRuntime>> {
        sampling_probs_impl::<WgpuRuntime>(self, logits, rule)
    }
}
