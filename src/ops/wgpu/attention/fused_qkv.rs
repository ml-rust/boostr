//! WebGPU implementation of FusedQkvOps
//!
//! Delegates to impl_generic (composed from numr primitives).
//! F32 only (WebGPU limitation).

use crate::error::Result;
use crate::ops::impl_generic::attention::fused_qkv::{
    fused_output_projection_residual_bwd_impl, fused_output_projection_residual_impl,
    fused_qkv_projection_bwd_impl, fused_qkv_projection_impl,
};
use crate::ops::traits::attention::fused_qkv::FusedQkvOps;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl FusedQkvOps<WgpuRuntime> for WgpuClient {
    fn fused_qkv_projection(
        &self,
        input: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        bias: Option<&Tensor<WgpuRuntime>>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        fused_qkv_projection_impl(self, input, weight, bias, num_heads, num_kv_heads, head_dim)
    }

    fn fused_output_projection_residual(
        &self,
        attn_out: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        bias: Option<&Tensor<WgpuRuntime>>,
        residual: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        fused_output_projection_residual_impl(self, attn_out, weight, bias, residual)
    }

    fn fused_qkv_projection_bwd(
        &self,
        dq: &Tensor<WgpuRuntime>,
        dk: &Tensor<WgpuRuntime>,
        dv: &Tensor<WgpuRuntime>,
        input: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        has_bias: bool,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Option<Tensor<WgpuRuntime>>,
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
        d_output: &Tensor<WgpuRuntime>,
        attn_out: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        has_bias: bool,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Option<Tensor<WgpuRuntime>>,
        Tensor<WgpuRuntime>,
    )> {
        fused_output_projection_residual_bwd_impl(self, d_output, attn_out, weight, has_bias)
    }
}
