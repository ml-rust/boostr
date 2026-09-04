//! CPU reference for backward of FP8 KV-cache fake-quantization.
//!
//! Plain loops, written for clarity: this is the parity reference the CUDA
//! `kv_cache_fp8_bwd` kernels are checked against, not a fast path.
//!
//! F32 only. Any other dtype is refused rather than mis-computed.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::dtype::fp8::fp8_e4m3_to_f32;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

fn check_f32(t: &Tensor<CpuRuntime>, arg: &'static str, op: &str) -> Result<()> {
    if t.dtype() != DType::F32 {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!("{op} on CPU is F32 only, got {:?}", t.dtype()),
        });
    }
    Ok(())
}

fn check_kv_fp8(t: &Tensor<CpuRuntime>) -> Result<()> {
    if t.dtype() != DType::FP8E4M3 {
        return Err(Error::InvalidArgument {
            arg: "kv_fp8",
            reason: format!("kv_fp8_bwd: kv_fp8 must be FP8E4M3, got {:?}", t.dtype()),
        });
    }
    Ok(())
}

/// Backward for FP8 fake-quantization with a single tensor-wide scale.
///
/// `grad_kv[i] = grad_output[i]` (straight-through identity). `grad_scale =
/// sum(grad_output[i] * -c_i / scale^2)`, where `c_i` is the raw FP8 code
/// decoded with `fp8_e4m3_to_f32` (equivalent to the CUDA kernel's
/// `fp8_e4m3_to_f32(byte, 1.0)`, which recovers the code, not the
/// dequantized value). Returns `(grad_kv, grad_scale)` with `grad_scale` a
/// 1-element F32 tensor.
pub(super) fn kv_fp8_bwd_per_tensor_impl(
    _client: &CpuClient,
    grad_output: &Tensor<CpuRuntime>,
    kv_fp8: &Tensor<CpuRuntime>,
    scale: f32,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    check_f32(grad_output, "grad_output", "kv_fp8_bwd_per_tensor")?;
    check_kv_fp8(kv_fp8)?;

    let total_elements = grad_output.numel();
    if kv_fp8.numel() != total_elements {
        return Err(Error::InvalidArgument {
            arg: "kv_fp8",
            reason: format!(
                "kv_fp8_bwd_per_tensor: kv_fp8 has {} elements, grad_output has {total_elements}",
                kv_fp8.numel()
            ),
        });
    }

    let go = grad_output.to_vec::<f32>();
    let codes = kv_fp8.to_vec::<u8>();
    let inv_scale_sq = 1.0f32 / (scale * scale);

    let mut grad_scale_acc = 0.0f32;
    for (g, c) in go.iter().zip(codes.iter()) {
        let raw = fp8_e4m3_to_f32(*c);
        grad_scale_acc += g * (-raw * inv_scale_sq);
    }

    let device = grad_output.device();
    let grad_kv = Tensor::<CpuRuntime>::from_slice(&go, grad_output.shape(), device)?;
    let grad_scale = Tensor::<CpuRuntime>::from_slice(&[grad_scale_acc], &[1], device)?;

    Ok((grad_kv, grad_scale))
}

/// Backward for FP8 fake-quantization with one scale per token.
///
/// Same STE identity and per-element formula as `kv_fp8_bwd_per_tensor_impl`,
/// reduced per token instead of over the whole tensor:
/// `grad_scales[token] = sum_d(grad_output[d] * -c_d / scale[token]^2)`.
/// `total_tokens = batch * num_kv_heads * seq_len`, matching
/// `quantize_kv_fp8_per_token`'s flat `[num_tokens]` scale layout.
#[allow(clippy::too_many_arguments)]
pub(super) fn kv_fp8_bwd_per_token_impl(
    _client: &CpuClient,
    grad_output: &Tensor<CpuRuntime>,
    kv_fp8: &Tensor<CpuRuntime>,
    scales: &Tensor<CpuRuntime>,
    batch: usize,
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    check_f32(grad_output, "grad_output", "kv_fp8_bwd_per_token")?;
    check_kv_fp8(kv_fp8)?;
    check_f32(scales, "scales", "kv_fp8_bwd_per_token")?;

    let total_tokens = batch * num_kv_heads * seq_len;
    let expected_numel = total_tokens * head_dim;
    if grad_output.numel() != expected_numel {
        return Err(Error::InvalidArgument {
            arg: "grad_output",
            reason: format!(
                "kv_fp8_bwd_per_token: expected {expected_numel} elements (batch*num_kv_heads*seq_len*head_dim), got {}",
                grad_output.numel()
            ),
        });
    }
    if kv_fp8.numel() != expected_numel {
        return Err(Error::InvalidArgument {
            arg: "kv_fp8",
            reason: format!(
                "kv_fp8_bwd_per_token: kv_fp8 has {} elements, expected {expected_numel}",
                kv_fp8.numel()
            ),
        });
    }
    if scales.numel() != total_tokens {
        return Err(Error::InvalidArgument {
            arg: "scales",
            reason: format!(
                "kv_fp8_bwd_per_token: scales has {} elements, expected {total_tokens} (batch*num_kv_heads*seq_len)",
                scales.numel()
            ),
        });
    }

    let go = grad_output.to_vec::<f32>();
    let codes = kv_fp8.to_vec::<u8>();
    let scale_vals = scales.to_vec::<f32>();

    let mut grad_scales_acc = vec![0.0f32; total_tokens];
    for (token_idx, gs) in grad_scales_acc.iter_mut().enumerate() {
        let token_scale = scale_vals[token_idx];
        let inv_scale_sq = 1.0f32 / (token_scale * token_scale);
        let offset = token_idx * head_dim;
        let mut acc = 0.0f32;
        for d in 0..head_dim {
            let raw = fp8_e4m3_to_f32(codes[offset + d]);
            acc += go[offset + d] * (-raw * inv_scale_sq);
        }
        *gs = acc;
    }

    let device = grad_output.device();
    let grad_kv = Tensor::<CpuRuntime>::from_slice(&go, grad_output.shape(), device)?;
    let grad_scales = Tensor::<CpuRuntime>::from_slice(&grad_scales_acc, &[total_tokens], device)?;

    Ok((grad_kv, grad_scales))
}
