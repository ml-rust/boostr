//! WebGPU implementation of CalibrationOps
//!
//! Delegates to impl_generic for all methods.
//! numr's WebGPU matmul/reduce ops handle the heavy lifting.

use crate::error::Result;
use crate::ops::impl_generic::quantization::calibration::{
    awq_channel_scores_impl, fisher_information_impl, gptq_hessian_update_impl,
    gptq_quantize_column_impl,
};
use crate::ops::traits::CalibrationOps;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl CalibrationOps<WgpuRuntime> for WgpuClient {
    fn awq_channel_scores(
        &self,
        activations: &Tensor<WgpuRuntime>,
        weights: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        awq_channel_scores_impl(self, activations, weights)
    }

    fn fisher_information(&self, gradients: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        fisher_information_impl(self, gradients)
    }

    fn gptq_hessian_update(
        &self,
        hessian: &Tensor<WgpuRuntime>,
        x_block: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        gptq_hessian_update_impl(self, hessian, x_block)
    }

    fn gptq_quantize_column(
        &self,
        weight: &Tensor<WgpuRuntime>,
        h_inv: &Tensor<WgpuRuntime>,
        num_bits: u32,
        group_size: u32,
        symmetric: bool,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        gptq_quantize_column_impl(self, weight, h_inv, num_bits, group_size, symmetric)
    }
}
