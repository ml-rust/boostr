//! CPU implementation of CalibrationOps — delegates to impl_generic

use crate::error::Result;
use crate::ops::impl_generic::quantization::calibration::{
    awq_channel_scores_impl, fisher_information_impl, gptq_hessian_update_impl,
    gptq_quantize_column_impl,
};
use crate::ops::traits::CalibrationOps;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl CalibrationOps<CpuRuntime> for CpuClient {
    fn awq_channel_scores(
        &self,
        activations: &Tensor<CpuRuntime>,
        weights: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        awq_channel_scores_impl(self, activations, weights)
    }

    fn fisher_information(&self, gradients: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        fisher_information_impl(self, gradients)
    }

    fn gptq_hessian_update(
        &self,
        hessian: &Tensor<CpuRuntime>,
        x_block: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        gptq_hessian_update_impl(self, hessian, x_block)
    }

    fn gptq_quantize_column(
        &self,
        weight: &Tensor<CpuRuntime>,
        h_inv: &Tensor<CpuRuntime>,
        num_bits: u32,
        group_size: u32,
        symmetric: bool,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        gptq_quantize_column_impl(self, weight, h_inv, num_bits, group_size, symmetric)
    }
}
