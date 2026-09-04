//! CUDA Flash Attention v2 — forward and backward
//!
//! Fused kernel — this is a PRIMITIVE op (kernel IS the algorithm).
//! Supports: F32, F16, BF16, FP8E4M3 with GQA and sliding window.
//! Head dimensions: 32, 64, 96, 128, 192, 256.

use crate::error::{Error, Result};
use crate::ops::traits::FlashAttentionOps;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::decode_split;
use super::flash_bwd;
use super::flash_bwd_fp8;
use super::flash_decode;
use super::flash_fwd;
use super::flash_fwd_fp8_kv;
use super::flash_utils::validate_qkv;
use super::flash_v3;
use super::mqa_gqa;

pub use super::flash_decode::decode_attention_graph_fwd;
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
        kv_seq_len: Option<usize>,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let mut p = validate_qkv(q, k, v, num_heads, num_kv_heads, head_dim)?;

        // kv_seq_len override: use actual seq len as loop bound, tensor dim-2 as stride
        let kv_seq_stride = p.seq_len_k; // memory stride = full capacity
        if let Some(seq_len) = kv_seq_len {
            p.seq_len_k = seq_len;
        }

        // Decode path: S_q=1, use lightweight vec kernel (supports separate stride).
        // Instantiated for F32/F16/BF16; anything else falls through to the
        // general kernel, which tiles a one-row query and is far slower here.
        if p.seq_len_q == 1
            && decode_split::decode_supports_dtype(q.dtype())
            && (head_dim == 64 || head_dim == 128)
            && window_size == 0
        {
            return flash_decode::decode_attention_fwd(self, q, k, v, &p, kv_seq_stride);
        }

        // Flash v2/v3 don't support separate kv_seq_stride — narrow if needed
        if kv_seq_stride != p.seq_len_k {
            let k_narrow = k.narrow(2, 0, p.seq_len_k)?.contiguous()?;
            let v_narrow = v.narrow(2, 0, p.seq_len_k)?.contiguous()?;
            return self.flash_attention_fwd(
                q,
                &k_narrow,
                &v_narrow,
                num_heads,
                num_kv_heads,
                head_dim,
                causal,
                window_size,
                None,
            );
        }

        // Flash v3 on SM 90+ for supported configs, when v3 is dispatchable
        // at all. `flash_v3::dispatch_enabled` is the single decision point and
        // is currently false — see its doc comment.
        if num_kv_heads == num_heads
            && window_size == 0
            && flash_v3::dispatch_enabled(self, q.device())
            && let Some(result) = flash_v3::flash_v3_fwd(
                self,
                q,
                k,
                v,
                p.batch_size,
                p.num_heads,
                p.seq_len_q,
                p.seq_len_k,
                p.head_dim,
                causal,
            )?
        {
            return Ok(result);
        }

        // Dedicated MQA/GQA kernels, for the shapes they're capable of.
        //
        // `should_use_mqa_gqa` is a capability gate, not a performance
        // heuristic — see its doc comment. Gated on two more things the
        // kernels actually support: `window_size == 0` because they have no
        // sliding-window path, and F32/F16/BF16 because those are the only
        // dtype variants instantiated. Anything else falls through to the
        // general kernel below, which is what ran for every shape before
        // this was wired up.
        // `caps.bf16` gates this because `mqa_gqa_bwd.cu` needs sm_80 for its
        // native bf16 backward kernels. The forward runs fine below sm_80, but
        // gating it here too keeps forward and backward on the same kernel
        // family. Falling through reaches the general flash kernel.
        if window_size == 0
            && matches!(q.dtype(), DType::F32 | DType::F16 | DType::BF16)
            && numr::runtime::cuda::CudaDevice::new(q.device().id())
                .profile()
                .caps
                .bf16
            && mqa_gqa::should_use_mqa_gqa(num_heads, num_kv_heads, head_dim)
        {
            return mqa_gqa::mqa_gqa_fwd(self, q, k, v, num_heads, num_kv_heads, head_dim, causal);
        }

        flash_fwd::flash_attention_fwd_impl(self, q, k, v, &p, causal, window_size)
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
        let p = validate_qkv(q, k, v, num_heads, num_kv_heads, head_dim)?;

        // Same v3 gate as the forward, through the same single decision
        // point: a v3 forward with a v2 backward would mix causal conventions.
        if num_kv_heads == num_heads
            && window_size == 0
            && flash_v3::dispatch_enabled(self, q.device())
            && let Some(result) = flash_v3::flash_v3_bwd(
                self,
                dout,
                q,
                k,
                v,
                output,
                lse,
                p.batch_size,
                p.num_heads,
                p.seq_len_q,
                p.seq_len_k,
                p.head_dim,
                causal,
            )?
        {
            return Ok(result);
        }

        // Same gate as the forward. Both halves must agree: routing the forward
        // to the MQA/GQA kernel and the backward to the general one would pair
        // kernels that were never parity-tested together.
        // `caps.bf16` gates this because `mqa_gqa_bwd.cu` needs sm_80 for its
        // native bf16 backward kernels. Falling through reaches the general
        // flash kernel.
        if window_size == 0
            && matches!(q.dtype(), DType::F32 | DType::F16 | DType::BF16)
            && numr::runtime::cuda::CudaDevice::new(q.device().id())
                .profile()
                .caps
                .bf16
            && mqa_gqa::should_use_mqa_gqa(num_heads, num_kv_heads, head_dim)
        {
            return mqa_gqa::mqa_gqa_bwd(
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
            );
        }

        flash_bwd::flash_attention_bwd_impl(
            self,
            dout,
            q,
            k,
            v,
            output,
            lse,
            &p,
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
