//! CUDA implementation of MRopeOps: the composed op, plus the fused
//! `mrope_interleaved_f32` kernel for the F32 / I32 inference case.

use super::mrope_interleaved::{kernel_takes, mrope_interleaved_f32};
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

    /// One kernel launch when `x`, the tables and `selector` are `F32`,
    /// `positions` is `I32` and `x` needs no gradient; the composed op
    /// otherwise (it carries the backward graph, and the kernel has no
    /// other dtype).
    fn mrope_interleaved_fused(
        &self,
        x: &Var<CudaRuntime>,
        cos_cache: &Var<CudaRuntime>,
        sin_cache: &Var<CudaRuntime>,
        positions: &Tensor<CudaRuntime>,
        selector: &Tensor<CudaRuntime>,
        n_rot: usize,
    ) -> Result<Var<CudaRuntime>> {
        let (xt, cos, sin) = (x.tensor(), cos_cache.tensor(), sin_cache.tensor());
        if x.requires_grad() || !kernel_takes(xt, cos, sin, positions, selector) {
            return apply_mrope_interleaved_impl(
                self, x, cos_cache, sin_cache, positions, selector, n_rot,
            );
        }
        let out = mrope_interleaved_f32(self, xt, cos, sin, positions, selector, n_rot)?;
        Ok(Var::new(out, false))
    }
}
