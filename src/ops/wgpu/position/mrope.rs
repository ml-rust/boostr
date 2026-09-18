//! WebGPU implementation of MRopeOps — delegates to impl_generic.

use crate::error::Result;
use crate::ops::impl_generic::position::apply_mrope_interleaved_impl;
use crate::ops::traits::position::MRopeOps;
use numr::autograd::Var;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl MRopeOps<WgpuRuntime> for WgpuClient {
    fn apply_mrope_interleaved(
        &self,
        x: &Var<WgpuRuntime>,
        cos_cache: &Var<WgpuRuntime>,
        sin_cache: &Var<WgpuRuntime>,
        positions: &Tensor<WgpuRuntime>,
        selector: &Tensor<WgpuRuntime>,
        n_rot: usize,
    ) -> Result<Var<WgpuRuntime>> {
        apply_mrope_interleaved_impl(self, x, cos_cache, sin_cache, positions, selector, n_rot)
    }
}
