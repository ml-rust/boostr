//! WebGPU implementation of GatedDeltaNetOps
//!
//! Delegates to impl_generic: every path is batched matmul plus elementwise
//! ops. The chunked prefill and its backward add numr's batched triangular
//! solve. All of them are native on this backend.

use crate::error::Result;
use crate::ops::impl_generic::architecture::gated_delta_net::{
    gdn_chunk_backward_impl, gdn_chunk_prefill_impl, gdn_step_from_conv_impl, gdn_step_impl,
};
use crate::ops::traits::architecture::gated_delta_net::{GatedDeltaNetOps, GdnGrads};
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

    fn gdn_chunk_backward(
        &self,
        q: &Tensor<WgpuRuntime>,
        k: &Tensor<WgpuRuntime>,
        v: &Tensor<WgpuRuntime>,
        g: &Tensor<WgpuRuntime>,
        beta: &Tensor<WgpuRuntime>,
        state: &Tensor<WgpuRuntime>,
        d_o: &Tensor<WgpuRuntime>,
        d_state_final: Option<&Tensor<WgpuRuntime>>,
        chunk_size: usize,
    ) -> Result<GdnGrads<WgpuRuntime>> {
        gdn_chunk_backward_impl(
            self,
            q,
            k,
            v,
            g,
            beta,
            state,
            d_o,
            d_state_final,
            chunk_size,
        )
    }

    fn gdn_step_from_conv(
        &self,
        qkv: &Tensor<WgpuRuntime>,
        alpha_raw: &Tensor<WgpuRuntime>,
        beta_raw: &Tensor<WgpuRuntime>,
        dt_bias: &Tensor<WgpuRuntime>,
        ssm_a: &Tensor<WgpuRuntime>,
        state: &Tensor<WgpuRuntime>,
        h_k: usize,
        key_dim: usize,
        value_dim: usize,
        eps: f32,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        gdn_step_from_conv_impl(
            self, qkv, alpha_raw, beta_raw, dt_bias, ssm_a, state, h_k, key_dim, value_dim, eps,
        )
    }
}
