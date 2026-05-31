//! CPU implementation of RoPEPackedOps — delegates to impl_generic.

use crate::error::Result;
use crate::ops::impl_generic::position::apply_rope_packed_impl;
use crate::ops::traits::position::RoPEPackedOps;
use numr::autograd::Var;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl RoPEPackedOps<CpuRuntime> for CpuClient {
    fn apply_rope_packed(
        &self,
        x: &Var<CpuRuntime>,
        cos_cache: &Var<CpuRuntime>,
        sin_cache: &Var<CpuRuntime>,
        position_ids: &Tensor<CpuRuntime>,
    ) -> Result<Var<CpuRuntime>> {
        apply_rope_packed_impl(self, x, cos_cache, sin_cache, position_ids)
    }
}
