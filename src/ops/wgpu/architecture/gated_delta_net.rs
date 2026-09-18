//! WebGPU implementation of GatedDeltaNetOps
//!
//! Delegates to impl_generic: both paths are batched matmul plus
//! elementwise ops, which are already native on this backend.

use crate::error::Result;
use crate::ops::impl_generic::architecture::gated_delta_net::{
    gdn_chunk_prefill_impl, gdn_step_impl,
};
use crate::ops::traits::architecture::gated_delta_net::GatedDeltaNetOps;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl GatedDeltaNetOps<WgpuRuntime> for WgpuClient {
    fn gdn_step(
        &self,
        q: &Tensor<WgpuRuntime>,
        k: &Tensor<WgpuRuntime>,
        v: &Tensor<WgpuRuntime>,
        g: &Tensor<WgpuRuntime>,
        beta: &Tensor<WgpuRuntime>,
        state: &Tensor<WgpuRuntime>,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        gdn_step_impl(self, q, k, v, g, beta, state)
    }

    fn gdn_chunk_prefill(
        &self,
        q: &Tensor<WgpuRuntime>,
        k: &Tensor<WgpuRuntime>,
        v: &Tensor<WgpuRuntime>,
        g: &Tensor<WgpuRuntime>,
        beta: &Tensor<WgpuRuntime>,
        state: &Tensor<WgpuRuntime>,
        chunk_size: usize,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        gdn_chunk_prefill_impl(self, q, k, v, g, beta, state, chunk_size)
    }
}
