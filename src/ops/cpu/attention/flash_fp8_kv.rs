//! CPU reference for `FlashAttentionOps::flash_attention_fwd_fp8_kv`.
//!
//! Not a fused kernel: dequantizes K/V from FP8 E4M3 to F32, then runs the
//! same standard attention path CPU `flash_attention_fwd` uses.

use crate::error::{Error, Result};
use crate::ops::impl_generic::attention::{StandardAttnConfig, standard_attention_fwd};
use numr::dtype::DType;
use numr::dtype::fp8::fp8_e4m3_to_f32;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

/// Dequantize an FP8 E4M3 KV tensor `[batch, heads, seq_len, head_dim]` to F32.
///
/// Matches the CUDA kernel's convention (`kv_cache_quant.cu`): a stored scale
/// is `448 / max_abs`, so dequantization is `fp8_e4m3_to_f32(byte) / scale`.
/// `scales` is `[batch, heads, seq_len]` for per-token, `[batch, heads]` for
/// per-head.
fn dequant_fp8_kv(
    quant: &Tensor<CpuRuntime>,
    scales: &Tensor<CpuRuntime>,
    per_token: bool,
) -> Result<Tensor<CpuRuntime>> {
    if quant.dtype() != DType::FP8E4M3 {
        return Err(Error::InvalidArgument {
            arg: "k_quant/v_quant",
            reason: format!(
                "flash_attention_fwd_fp8_kv requires FP8E4M3, got {:?}",
                quant.dtype()
            ),
        });
    }
    let shape = quant.shape().to_vec();
    if shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "k_quant/v_quant",
            reason: format!("expected 4D [B, H, S, D], got {}D", shape.len()),
        });
    }
    let (batch, heads, seq_len, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
    let bytes = quant.to_vec::<u8>();
    let scale_data = scales.to_vec::<f32>();

    let mut out = vec![0.0f32; bytes.len()];
    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_len {
                let scale = if per_token {
                    scale_data[(b * heads + h) * seq_len + s]
                } else {
                    scale_data[b * heads + h]
                };
                let base = ((b * heads + h) * seq_len + s) * head_dim;
                for d in 0..head_dim {
                    out[base + d] = fp8_e4m3_to_f32(bytes[base + d]) / scale;
                }
            }
        }
    }
    Ok(Tensor::<CpuRuntime>::from_slice(
        &out,
        &shape,
        quant.device(),
    )?)
}

/// CPU reference for `flash_attention_fwd_fp8_kv`: dequantize K/V, then run
/// `standard_attention_fwd` (the same path `flash_attention_fwd` falls back
/// to on CPU). `num_kv_heads` is fixed to `num_heads` — the kernel this
/// mirrors has no GQA.
#[allow(clippy::too_many_arguments)]
pub(super) fn flash_attention_fwd_fp8_kv_impl(
    client: &CpuClient,
    q: &Tensor<CpuRuntime>,
    k_quant: &Tensor<CpuRuntime>,
    v_quant: &Tensor<CpuRuntime>,
    k_scales: &Tensor<CpuRuntime>,
    v_scales: &Tensor<CpuRuntime>,
    num_heads: usize,
    causal: bool,
    per_token_scales: bool,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let k = dequant_fp8_kv(k_quant, k_scales, per_token_scales)?;
    let v = dequant_fp8_kv(v_quant, v_scales, per_token_scales)?;
    let cfg = StandardAttnConfig {
        num_heads,
        num_kv_heads: num_heads,
        causal,
        window_size: 0,
    };
    standard_attention_fwd(client, q, &k, &v, cfg)
}
