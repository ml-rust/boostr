//! CUDA implementation of GatedDeltaNetOps
//!
//! `gdn_step` runs the fused kernel in `gdn_step` when the call fits it
//! (F32, one token, `S_k` in {32, 64, 128}) and delegates to impl_generic
//! otherwise. `gdn_step_from_conv` does the same with the kernel in
//! `gdn_step_from_conv`, which also absorbs the split, L2 norm, head tiling
//! and gate chain. `gdn_chunk_prefill` delegates to impl_generic: batched
//! matmul plus elementwise ops, already native on this backend.

use crate::error::Result;
use crate::ops::cuda::architecture::gdn_step::{gdn_step_fused, supports};
use crate::ops::cuda::architecture::gdn_step_from_conv::{
    gdn_step_from_conv_fused, supports_from_conv,
};
use crate::ops::impl_generic::architecture::gated_delta_net::{
    check_gdn_conv_shapes, check_gdn_shapes, gdn_chunk_prefill_impl, gdn_step_from_conv_impl,
    gdn_step_impl,
};
use crate::ops::traits::architecture::gated_delta_net::GatedDeltaNetOps;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl GatedDeltaNetOps<CudaRuntime> for CudaClient {
    fn gdn_step(
        &self,
        q: &Tensor<CudaRuntime>,
        k: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
        g: &Tensor<CudaRuntime>,
        beta: &Tensor<CudaRuntime>,
        state: &Tensor<CudaRuntime>,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let dims = check_gdn_shapes(q, k, v, g, beta, state)?;
        let one_dtype = [q, k, v, g, beta, state]
            .iter()
            .all(|t| t.dtype() == state.dtype());
        if one_dtype && supports(&dims, state.dtype()) {
            return gdn_step_fused(self, q, k, v, g, beta, state);
        }
        gdn_step_impl(self, q, k, v, g, beta, state)
    }

    fn gdn_chunk_prefill(
        &self,
        q: &Tensor<CudaRuntime>,
        k: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
        g: &Tensor<CudaRuntime>,
        beta: &Tensor<CudaRuntime>,
        state: &Tensor<CudaRuntime>,
        chunk_size: usize,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        gdn_chunk_prefill_impl(self, q, k, v, g, beta, state, chunk_size)
    }

    fn gdn_step_from_conv(
        &self,
        qkv: &Tensor<CudaRuntime>,
        alpha_raw: &Tensor<CudaRuntime>,
        beta_raw: &Tensor<CudaRuntime>,
        dt_bias: &Tensor<CudaRuntime>,
        ssm_a: &Tensor<CudaRuntime>,
        state: &Tensor<CudaRuntime>,
        h_k: usize,
        key_dim: usize,
        value_dim: usize,
        eps: f32,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let dims = check_gdn_conv_shapes(
            qkv, alpha_raw, beta_raw, dt_bias, ssm_a, state, h_k, key_dim, value_dim,
        )?;
        let one_dtype = [qkv, alpha_raw, beta_raw, dt_bias, ssm_a, state]
            .iter()
            .all(|t| t.dtype() == state.dtype());
        if one_dtype && supports_from_conv(&dims, state.dtype()) {
            return gdn_step_from_conv_fused(
                self, qkv, alpha_raw, beta_raw, dt_bias, ssm_a, state, h_k, key_dim, value_dim, eps,
            );
        }
        gdn_step_from_conv_impl(
            self, qkv, alpha_raw, beta_raw, dt_bias, ssm_a, state, h_k, key_dim, value_dim, eps,
        )
    }
}
