//! CUDA implementation of RoPEPackedOps — delegates to impl_generic.

use crate::error::Result;
use crate::ops::impl_generic::position::apply_rope_packed_impl;
use crate::ops::traits::position::RoPEPackedOps;
use numr::autograd::Var;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl RoPEPackedOps<CudaRuntime> for CudaClient {
    fn apply_rope_packed(
        &self,
        x: &Var<CudaRuntime>,
        cos_cache: &Var<CudaRuntime>,
        sin_cache: &Var<CudaRuntime>,
        position_ids: &Tensor<CudaRuntime>,
    ) -> Result<Var<CudaRuntime>> {
        apply_rope_packed_impl(self, x, cos_cache, sin_cache, position_ids)
    }
}
