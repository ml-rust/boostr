//! WebGPU implementation of RoPEPackedOps — delegates to impl_generic.

use crate::error::Result;
use crate::ops::impl_generic::position::apply_rope_packed_impl;
use crate::ops::traits::position::RoPEPackedOps;
use numr::autograd::Var;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl RoPEPackedOps<WgpuRuntime> for WgpuClient {
    fn apply_rope_packed(
        &self,
        x: &Var<WgpuRuntime>,
        cos_cache: &Var<WgpuRuntime>,
        sin_cache: &Var<WgpuRuntime>,
        position_ids: &Tensor<WgpuRuntime>,
    ) -> Result<Var<WgpuRuntime>> {
        apply_rope_packed_impl(self, x, cos_cache, sin_cache, position_ids)
    }
}
