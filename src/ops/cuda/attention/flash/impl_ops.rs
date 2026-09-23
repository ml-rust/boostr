//! CUDA Flash Attention v2 trait impl.
//!
//! Fused kernel — this is a PRIMITIVE op (kernel IS the algorithm).
//! Supports: F32, F16, BF16, FP8E4M3 with GQA and sliding window.
//! Head dimensions: 32, 64, 96, 128, 192, 256.
//!
//! `flash_attention_fwd` and `flash_attention_bwd` route through
//! [`flash_dispatch`]; see its module doc for which kernel family serves
//! which shape. The FP8 and INT4-KV methods here validate dtype/capability
//! and delegate straight to their one kernel family.

use crate::error::{Error, Result};
use crate::ops::traits::{AttnOutLayout, FlashAttentionOps, KvSeqLen};
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::flash_bwd_fp8;
use super::flash_dispatch;
use super::flash_fwd;
use super::flash_fwd_fp8_kv;
use super::flash_fwd_int4_kv;
use super::flash_utils::validate_qkv;
use crate::ops::traits::cache::kv_cache_quant::Int4GroupSize;

pub use super::flash_decode_graph::decode_attention_graph_fwd;
pub(crate) use super::flash_utils::set_smem_attribute;

impl FlashAttentionOps<CudaRuntime> for CudaClient {
    fn flash_attention_fwd(
        &self,
        q: &Tensor<CudaRuntime>,
        k: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
        window_size: usize,
        kv_seq_len: Option<KvSeqLen<'_, CudaRuntime>>,
        kv_start: Option<&Tensor<CudaRuntime>>,
        out_layout: AttnOutLayout,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        flash_dispatch::dispatch_fwd(
            self,
            q,
            k,
            v,
            num_heads,
            num_kv_heads,
            head_dim,
            causal,
            window_size,
            kv_seq_len,
            kv_start,
            out_layout,
        )
    }

    fn flash_attention_fwd_fp8(
        &self,
        q: &Tensor<CudaRuntime>,
        k: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
        q_scale: f32,
        k_scale: f32,
        v_scale: f32,
        o_scale: f32,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let p = validate_qkv(q, k, v, num_heads, num_kv_heads, head_dim)?;
        let dtype = q.dtype();

        if !matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!(
                    "flash_attention_fwd_fp8 requires FP8 dtype, got {:?}",
                    dtype
                ),
            });
        }

        // flash_v2_fp8.cu compiles at sm_80. Below that, the device has no
        // FP8 symbol to launch.
        if !numr::runtime::cuda::CudaDevice::new(q.device().id())
            .profile()
            .caps
            .fp8
        {
            return Err(Error::KernelError {
                reason: "flash_attention_fwd_fp8: device lacks FP8 support".into(),
            });
        }

        flash_fwd::flash_attention_fwd_fp8_impl(
            self, q, k, v, &p, causal, q_scale, k_scale, v_scale, o_scale,
        )
    }

    fn flash_attention_fwd_fp8_kv(
        &self,
        q: &Tensor<CudaRuntime>,
        k_quant: &Tensor<CudaRuntime>,
        v_quant: &Tensor<CudaRuntime>,
        k_scales: &Tensor<CudaRuntime>,
        v_scales: &Tensor<CudaRuntime>,
        num_heads: usize,
        head_dim: usize,
        causal: bool,
        per_token_scales: bool,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        flash_fwd_fp8_kv::flash_attention_fwd_fp8_kv_impl(
            self,
            q,
            k_quant,
            v_quant,
            k_scales,
            v_scales,
            num_heads,
            head_dim,
            causal,
            per_token_scales,
        )
    }

    fn flash_attention_fwd_int4_kv(
        &self,
        q: &Tensor<CudaRuntime>,
        k_quant: &Tensor<CudaRuntime>,
        v_quant: &Tensor<CudaRuntime>,
        k_scales: &Tensor<CudaRuntime>,
        k_zeros: &Tensor<CudaRuntime>,
        v_scales: &Tensor<CudaRuntime>,
        v_zeros: &Tensor<CudaRuntime>,
        num_heads: usize,
        head_dim: usize,
        causal: bool,
        group_size: Int4GroupSize,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        flash_fwd_int4_kv::flash_attention_fwd_int4_kv_impl(
            self, q, k_quant, v_quant, k_scales, k_zeros, v_scales, v_zeros, num_heads, head_dim,
            causal, group_size,
        )
    }

    fn flash_attention_bwd(
        &self,
        dout: &Tensor<CudaRuntime>,
        q: &Tensor<CudaRuntime>,
        k: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
        output: &Tensor<CudaRuntime>,
        lse: &Tensor<CudaRuntime>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
        window_size: usize,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        flash_dispatch::dispatch_bwd(
            self,
            dout,
            q,
            k,
            v,
            output,
            lse,
            num_heads,
            num_kv_heads,
            head_dim,
            causal,
            window_size,
        )
    }

    fn flash_attention_bwd_fp8(
        &self,
        dout: &Tensor<CudaRuntime>,
        q: &Tensor<CudaRuntime>,
        k: &Tensor<CudaRuntime>,
        v: &Tensor<CudaRuntime>,
        output: &Tensor<CudaRuntime>,
        lse: &Tensor<CudaRuntime>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
        q_scale: f32,
        k_scale: f32,
        v_scale: f32,
        do_scale: f32,
        o_scale: f32,
        dq_scale: f32,
        dk_scale: f32,
        dv_scale: f32,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let p = validate_qkv(q, k, v, num_heads, num_kv_heads, head_dim)?;
        let dtype = q.dtype();

        if !matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!(
                    "flash_attention_bwd_fp8 requires FP8 dtype, got {:?}",
                    dtype
                ),
            });
        }

        // flash_v2_bwd_fp8.cu compiles at sm_80, same arch requirement as the forward gate above.
        if !numr::runtime::cuda::CudaDevice::new(q.device().id())
            .profile()
            .caps
            .fp8
        {
            return Err(Error::KernelError {
                reason: "flash_attention_bwd_fp8: device lacks FP8 support".into(),
            });
        }

        flash_bwd_fp8::flash_attention_bwd_fp8_impl(
            self, dout, q, k, v, output, lse, &p, causal, q_scale, k_scale, v_scale, do_scale,
            o_scale, dq_scale, dk_scale, dv_scale,
        )
    }
}
