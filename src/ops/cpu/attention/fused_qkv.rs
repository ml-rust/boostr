//! CPU implementation of FusedQkvOps
//!
//! Delegates to impl_generic (composed from numr primitives).

use crate::error::Result;
use crate::ops::impl_generic::attention::fused_qkv::{
    fused_output_projection_residual_bwd_impl, fused_output_projection_residual_impl,
    fused_qkv_projection_bwd_impl, fused_qkv_projection_impl,
};
use crate::ops::traits::attention::fused_qkv::FusedQkvOps;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl FusedQkvOps<CpuRuntime> for CpuClient {
    fn fused_qkv_projection(
        &self,
        input: &Tensor<CpuRuntime>,
        weight: &Tensor<CpuRuntime>,
        bias: Option<&Tensor<CpuRuntime>>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        fused_qkv_projection_impl(self, input, weight, bias, num_heads, num_kv_heads, head_dim)
    }

    fn fused_output_projection_residual(
        &self,
        attn_out: &Tensor<CpuRuntime>,
        weight: &Tensor<CpuRuntime>,
        bias: Option<&Tensor<CpuRuntime>>,
        residual: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        fused_output_projection_residual_impl(self, attn_out, weight, bias, residual)
    }

    fn fused_qkv_projection_bwd(
        &self,
        dq: &Tensor<CpuRuntime>,
        dk: &Tensor<CpuRuntime>,
        dv: &Tensor<CpuRuntime>,
        input: &Tensor<CpuRuntime>,
        weight: &Tensor<CpuRuntime>,
        has_bias: bool,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(
        Tensor<CpuRuntime>,
        Tensor<CpuRuntime>,
        Option<Tensor<CpuRuntime>>,
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
        d_output: &Tensor<CpuRuntime>,
        attn_out: &Tensor<CpuRuntime>,
        weight: &Tensor<CpuRuntime>,
        has_bias: bool,
    ) -> Result<(
        Tensor<CpuRuntime>,
        Tensor<CpuRuntime>,
        Option<Tensor<CpuRuntime>>,
        Tensor<CpuRuntime>,
    )> {
        fused_output_projection_residual_bwd_impl(self, d_output, attn_out, weight, has_bias)
    }
}
