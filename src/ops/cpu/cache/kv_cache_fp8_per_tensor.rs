//! CPU reference for per-tensor FP8 KV cache quantize/dequantize.
//!
//! Split out of `kv_cache_quant.rs` to keep that file under the `cpu/*.rs`
//! line limit. Mirrors the CUDA `f32_to_fp8_e4m3_raw`/`fp8_e4m3_to_f32`
//! convention: a stored scale is `448/max_abs`.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::dtype::fp8::{f32_to_fp8_e4m3, fp8_e4m3_to_f32};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

const FP8_E4M3_MAX: f32 = 448.0;

pub(super) fn quantize_kv_fp8_per_tensor_impl(
    _client: &CpuClient,
    input: &Tensor<CpuRuntime>,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let data = input.to_vec::<f32>();
    let device = input.device();

    let max_abs = data.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let scale = if max_abs > 0.0 {
        FP8_E4M3_MAX / max_abs
    } else {
        1.0
    };

    let quantized: Vec<u8> = data.iter().map(|v| f32_to_fp8_e4m3(*v * scale)).collect();

    let q_tensor =
        Tensor::<CpuRuntime>::from_bytes(&quantized, input.shape(), DType::FP8E4M3, device)?;
    let s_tensor = Tensor::<CpuRuntime>::from_slice(&[scale], &[1], device)?;
    Ok((q_tensor, s_tensor))
}

pub(super) fn dequantize_kv_fp8_per_tensor_impl(
    _client: &CpuClient,
    quantized: &Tensor<CpuRuntime>,
    scale: &Tensor<CpuRuntime>,
    output_dtype: DType,
) -> Result<Tensor<CpuRuntime>> {
    if quantized.dtype() != DType::FP8E4M3 {
        return Err(Error::InvalidArgument {
            arg: "quantized",
            reason: format!(
                "FP8 per-tensor dequant: quantized must be FP8E4M3, got {:?}",
                quantized.dtype()
            ),
        });
    }
    let q_data = quantized.to_vec::<u8>();
    let scale_val = scale.to_vec::<f32>()[0];
    let device = quantized.device();

    let output: Vec<f32> = q_data
        .iter()
        .map(|&q| fp8_e4m3_to_f32(q) / scale_val)
        .collect();

    let f32_out = Tensor::<CpuRuntime>::from_slice(&output, quantized.shape(), device)?;
    match output_dtype {
        DType::F32 => Ok(f32_out),
        DType::F16 | DType::BF16 => Ok(f32_out.to_dtype(output_dtype)?),
        _ => Err(Error::InvalidArgument {
            arg: "output_dtype",
            reason: format!("FP8 per-tensor dequant: unsupported output dtype {output_dtype:?}"),
        }),
    }
}
