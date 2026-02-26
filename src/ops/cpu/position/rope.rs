//! CPU implementation of RoPEOps — delegates to impl_generic

use crate::error::Result;
use crate::ops::impl_generic::attention::rope::{
    apply_rope_impl, apply_rope_interleaved_impl, apply_rope_yarn_impl,
};
use crate::ops::traits::RoPEOps;
use numr::autograd::Var;
use numr::runtime::cpu::{CpuClient, CpuRuntime};

impl RoPEOps<CpuRuntime> for CpuClient {
    fn apply_rope(
        &self,
        x: &Var<CpuRuntime>,
        cos_cache: &Var<CpuRuntime>,
        sin_cache: &Var<CpuRuntime>,
    ) -> Result<Var<CpuRuntime>> {
        apply_rope_impl(self, x, cos_cache, sin_cache)
    }

    fn apply_rope_interleaved(
        &self,
        x: &Var<CpuRuntime>,
        cos_cache: &Var<CpuRuntime>,
        sin_cache: &Var<CpuRuntime>,
    ) -> Result<Var<CpuRuntime>> {
        apply_rope_interleaved_impl(self, x, cos_cache, sin_cache)
    }

    fn apply_rope_yarn(
        &self,
        x: &Var<CpuRuntime>,
        cos_cache: &Var<CpuRuntime>,
        sin_cache: &Var<CpuRuntime>,
        attn_scale: f32,
    ) -> Result<Var<CpuRuntime>> {
        apply_rope_yarn_impl(self, x, cos_cache, sin_cache, attn_scale)
    }
}
