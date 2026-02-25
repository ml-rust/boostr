//! WebGPU implementation of RoPEOps
//!
//! Delegates to impl_generic for the algorithm (which composes numr primitives).
//! All operations are F32 only (WebGPU limitation).

use crate::error::Result;
use crate::ops::impl_generic::attention::rope::apply_rope_impl;
use crate::ops::traits::RoPEOps;
use numr::autograd::Var;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};

impl RoPEOps<WgpuRuntime> for WgpuClient {
    fn apply_rope(
        &self,
        x: &Var<WgpuRuntime>,
        cos_cache: &Var<WgpuRuntime>,
        sin_cache: &Var<WgpuRuntime>,
    ) -> Result<Var<WgpuRuntime>> {
        // WebGPU delegates to impl_generic for now.
        // TODO: Implement fused WGSL kernel for better performance
        apply_rope_impl(self, x, cos_cache, sin_cache)
    }
}
