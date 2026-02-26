//! CPU implementation of SpeculativeOps — delegates to impl_generic

use crate::error::Result;
use crate::ops::impl_generic::inference::speculative::{
    compute_acceptance_probs_impl, compute_expected_tokens_impl, verify_speculative_tokens_impl,
};
use crate::ops::traits::inference::speculative::{SpeculativeOps, VerificationResult};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl SpeculativeOps<CpuRuntime> for CpuClient {
    fn verify_speculative_tokens(
        &self,
        draft_probs: &Tensor<CpuRuntime>,
        target_probs: &Tensor<CpuRuntime>,
        draft_tokens: &Tensor<CpuRuntime>,
        seed: u64,
    ) -> Result<Vec<VerificationResult>> {
        verify_speculative_tokens_impl(self, draft_probs, target_probs, draft_tokens, seed)
    }

    fn compute_acceptance_probs(
        &self,
        draft_probs: &Tensor<CpuRuntime>,
        target_probs: &Tensor<CpuRuntime>,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        compute_acceptance_probs_impl(self, draft_probs, target_probs)
    }

    fn compute_expected_tokens(
        &self,
        acceptance_rates: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        compute_expected_tokens_impl(self, acceptance_rates)
    }
}
