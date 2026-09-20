//! CPU implementation of GatedDeltaNetOps
//!
//! Delegates to impl_generic: both paths are batched matmul plus
//! elementwise ops, which are already native on this backend.

use crate::error::Result;
use crate::ops::impl_generic::architecture::gated_delta_net::{
    gdn_chunk_prefill_impl, gdn_step_from_conv_impl, gdn_step_impl,
};
use crate::ops::traits::architecture::gated_delta_net::GatedDeltaNetOps;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl GatedDeltaNetOps<CpuRuntime> for CpuClient {
    fn gdn_step(
        &self,
        q: &Tensor<CpuRuntime>,
        k: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
        g: &Tensor<CpuRuntime>,
        beta: &Tensor<CpuRuntime>,
        state: &Tensor<CpuRuntime>,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        gdn_step_impl(self, q, k, v, g, beta, state)
    }

    fn gdn_chunk_prefill(
        &self,
        q: &Tensor<CpuRuntime>,
        k: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
        g: &Tensor<CpuRuntime>,
        beta: &Tensor<CpuRuntime>,
        state: &Tensor<CpuRuntime>,
        chunk_size: usize,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        gdn_chunk_prefill_impl(self, q, k, v, g, beta, state, chunk_size)
    }

    fn gdn_step_from_conv(
        &self,
        qkv: &Tensor<CpuRuntime>,
        alpha_raw: &Tensor<CpuRuntime>,
        beta_raw: &Tensor<CpuRuntime>,
        dt_bias: &Tensor<CpuRuntime>,
        ssm_a: &Tensor<CpuRuntime>,
        state: &Tensor<CpuRuntime>,
        h_k: usize,
        key_dim: usize,
        value_dim: usize,
        eps: f32,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        gdn_step_from_conv_impl(
            self, qkv, alpha_raw, beta_raw, dt_bias, ssm_a, state, h_k, key_dim, value_dim, eps,
        )
    }
}
