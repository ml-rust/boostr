//! CPU implementation of SamplingProbsOps — delegates to impl_generic.

use crate::error::Result;
use crate::ops::impl_generic::inference::sampling::sampling_probs_impl;
use crate::ops::traits::inference::sampling_probs::{FilterRule, SamplingProbsOps};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl SamplingProbsOps<CpuRuntime> for CpuClient {
    fn sampling_probs(
        &self,
        logits: &Tensor<CpuRuntime>,
        rule: &FilterRule,
    ) -> Result<Tensor<CpuRuntime>> {
        sampling_probs_impl::<CpuRuntime>(self, logits, rule)
    }
}
