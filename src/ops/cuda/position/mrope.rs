//! CUDA implementation of MRopeOps — delegates to impl_generic.

use crate::error::Result;
use crate::ops::impl_generic::position::apply_mrope_interleaved_impl;
use crate::ops::traits::position::MRopeOps;
use numr::autograd::Var;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl MRopeOps<CudaRuntime> for CudaClient {
    fn apply_mrope_interleaved(
        &self,
        x: &Var<CudaRuntime>,
        cos_cache: &Var<CudaRuntime>,
        sin_cache: &Var<CudaRuntime>,
        positions: &Tensor<CudaRuntime>,
        selector: &Tensor<CudaRuntime>,
        n_rot: usize,
    ) -> Result<Var<CudaRuntime>> {
        apply_mrope_interleaved_impl(self, x, cos_cache, sin_cache, positions, selector, n_rot)
    }
}
