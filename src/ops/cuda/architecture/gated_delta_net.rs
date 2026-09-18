//! CUDA implementation of GatedDeltaNetOps
//!
//! Delegates to impl_generic: both paths are batched matmul plus
//! elementwise ops, which are already native on this backend.

use crate::error::Result;
use crate::ops::impl_generic::architecture::gated_delta_net::{
    gdn_chunk_prefill_impl, gdn_step_impl,
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
}
