//! CUDA implementation of FusedQkvOps
//!
//! Forward passes delegate to impl_generic (numr's CUDA matmul is already fast).
//! Backward passes also delegate to impl_generic for correctness.
//! Custom fused kernels can replace these when profiling shows benefit.

use crate::error::Result;
use crate::ops::impl_generic::attention::fused_qkv::{
    fused_output_projection_residual_bwd_impl, fused_output_projection_residual_impl,
    fused_qkv_projection_bwd_impl, fused_qkv_projection_impl,
};
use crate::ops::traits::attention::fused_qkv::FusedQkvOps;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl FusedQkvOps<CudaRuntime> for CudaClient {
    fn fused_qkv_projection(
        &self,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        bias: Option<&Tensor<CudaRuntime>>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        fused_qkv_projection_impl(self, input, weight, bias, num_heads, num_kv_heads, head_dim)
    }

    fn fused_output_projection_residual(
        &self,
        attn_out: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        bias: Option<&Tensor<CudaRuntime>>,
        residual: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        fused_output_projection_residual_impl(self, attn_out, weight, bias, residual)
    }

    fn fused_qkv_projection_bwd(
        &self,
        dq: &Tensor<CudaRuntime>,
        dk: &Tensor<CudaRuntime>,
        dv: &Tensor<CudaRuntime>,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        has_bias: bool,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Option<Tensor<CudaRuntime>>,
    )> {
        fused_qkv_projection_bwd_impl(
            self,
            dq,
            dk,
            dv,
            input,
            weight,
            has_bias,
            num_heads,
            num_kv_heads,
            head_dim,
        )
    }

    fn fused_output_projection_residual_bwd(
        &self,
        d_output: &Tensor<CudaRuntime>,
        attn_out: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        has_bias: bool,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Option<Tensor<CudaRuntime>>,
        Tensor<CudaRuntime>,
    )> {
        fused_output_projection_residual_bwd_impl(self, d_output, attn_out, weight, has_bias)
    }
}
