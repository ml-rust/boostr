//! CUDA Flash Attention v2 — forward and backward
//!
//! Fused kernel — this is a PRIMITIVE op (kernel IS the algorithm).
//! Supports: F32, F16, BF16, FP8E4M3 with GQA and sliding window.
//! Head dimensions: 32, 64, 96, 128, 192, 256.

use crate::error::{Error, Result};
use crate::ops::traits::FlashAttentionOps;
use numr::dtype::DType;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::flash_bwd;
use super::flash_bwd_fp8;
use super::flash_decode;
use super::flash_fwd;
use super::flash_utils::validate_qkv;
use super::flash_v3;

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

        // Decode path: S_q=1, use lightweight vec kernel (supports separate stride)
        if p.seq_len_q == 1
            && q.dtype() == DType::F32
            && (head_dim == 64 || head_dim == 128)
            && window_size == 0
        {
            return flash_decode::decode_attention_fwd(self, q, k, v, &p, kv_seq_stride);
        }

        // Flash v2/v3 don't support separate kv_seq_stride — narrow if needed
        if kv_seq_stride != p.seq_len_k {
            let k_narrow = k.narrow(2, 0, p.seq_len_k)?.contiguous();
            let v_narrow = v.narrow(2, 0, p.seq_len_k)?.contiguous();
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

        // Try Flash v3 on Hopper (SM 90+) for supported configs
        if num_kv_heads == num_heads && window_size == 0 && flash_v3::is_hopper(self, q.device()) {
            if let Some(result) = flash_v3::flash_v3_fwd(
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
            )? {
                return Ok(result);
            }
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

        flash_fwd::flash_attention_fwd_fp8_impl(
            self, q, k, v, &p, causal, q_scale, k_scale, v_scale, o_scale,
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

        // Try Flash v3 on Hopper (SM 90+) for supported configs
        if num_kv_heads == num_heads && window_size == 0 && flash_v3::is_hopper(self, q.device()) {
            if let Some(result) = flash_v3::flash_v3_bwd(
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
            )? {
                return Ok(result);
            }
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

        flash_bwd_fp8::flash_attention_bwd_fp8_impl(
            self, dout, q, k, v, output, lse, &p, causal, q_scale, k_scale, v_scale, do_scale,
            o_scale, dq_scale, dk_scale, dv_scale,
        )
    }
}
