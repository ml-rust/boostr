//! CPU implementation of MRopeOps — delegates to impl_generic.

use crate::error::Result;
use crate::ops::impl_generic::position::apply_mrope_interleaved_impl;
use crate::ops::traits::position::MRopeOps;
use numr::autograd::Var;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl MRopeOps<CpuRuntime> for CpuClient {
    fn apply_mrope_interleaved(
        &self,
        x: &Var<CpuRuntime>,
        cos_cache: &Var<CpuRuntime>,
        sin_cache: &Var<CpuRuntime>,
        positions: &Tensor<CpuRuntime>,
        selector: &Tensor<CpuRuntime>,
        n_rot: usize,
    ) -> Result<Var<CpuRuntime>> {
        apply_mrope_interleaved_impl(self, x, cos_cache, sin_cache, positions, selector, n_rot)
    }
}
