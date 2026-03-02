//! CPU implementation of SamplingOps — delegates to impl_generic

use crate::error::Result;
use crate::ops::impl_generic::inference::sampling::{
    apply_sampling_penalties_impl, logits_to_token_impl, sample_token_impl,
};
use crate::ops::traits::inference::sampling::SamplingOps;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl SamplingOps<CpuRuntime> for CpuClient {
    fn apply_sampling_penalties(
        &self,
        logits: &Tensor<CpuRuntime>,
        token_ids: &Tensor<CpuRuntime>,
        token_counts: &Tensor<CpuRuntime>,
        repeat_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
    ) -> Result<()> {
        apply_sampling_penalties_impl::<CpuRuntime>(
            self,
            logits,
            token_ids,
            token_counts,
            repeat_penalty,
            frequency_penalty,
            presence_penalty,
        )
    }

    fn sample_token(
        &self,
        logits: &Tensor<CpuRuntime>,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
    ) -> Result<u32> {
        sample_token_impl::<CpuRuntime>(self, logits, temperature, top_k, top_p, min_p)
    }

    fn logits_to_token(
        &self,
        logits: &Tensor<CpuRuntime>,
        token_ids: &Tensor<CpuRuntime>,
        token_counts: &Tensor<CpuRuntime>,
        num_unique: usize,
        repeat_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
    ) -> Result<Tensor<CpuRuntime>> {
        logits_to_token_impl::<CpuRuntime>(
            self,
            logits,
            token_ids,
            token_counts,
            num_unique,
            repeat_penalty,
            frequency_penalty,
            presence_penalty,
            temperature,
            top_k,
            top_p,
            min_p,
        )
    }
}
