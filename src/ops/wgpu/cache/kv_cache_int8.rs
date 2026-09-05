//! WebGPU launchers for INT8 KV cache quantize/dequantize.
//!
//! Split out of `kv_cache_quant.rs` to keep that file under the `wgpu/*.rs`
//! line limit.

use super::kv_cache_quant::{QuantParams, create_params_buf, dispatch, validate_f32};
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;

const QUANT_INT8_SRC: &str = include_str!("../shaders/cache/kv_cache_quant_int8.wgsl");
const DEQUANT_INT8_SRC: &str = include_str!("../shaders/cache/kv_cache_dequant_int8.wgsl");

pub(super) fn quantize_kv_int8_impl(
    client: &WgpuClient,
    input: &Tensor<WgpuRuntime>,
    num_tokens: usize,
    head_dim: usize,
) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
    validate_f32(input, "quantize_kv_int8")?;

    let quantized =
        Tensor::<WgpuRuntime>::zeros(&[num_tokens, head_dim], DType::F32, input.device())?;
    let scales = Tensor::<WgpuRuntime>::zeros(&[num_tokens], DType::F32, input.device())?;

    let input_buf = get_buffer(input.storage().ptr()).ok_or_else(|| Error::KernelError {
        reason: "input buffer not found".into(),
    })?;
    let quant_buf = get_buffer(quantized.storage().ptr()).ok_or_else(|| Error::KernelError {
        reason: "quantized buffer not found".into(),
    })?;
    let scales_buf = get_buffer(scales.storage().ptr()).ok_or_else(|| Error::KernelError {
        reason: "scales buffer not found".into(),
    })?;

    let params = QuantParams {
        num_tokens: num_tokens as u32,
        head_dim: head_dim as u32,
        group_size: 0,
        mode: 1,
    };
    let params_buf = create_params_buf(client, &params);

    // Shader bindings: 0=input(read), 1=output(rw), 2=scales(rw)
    dispatch(
        client,
        QUANT_INT8_SRC,
        "quantize_kv_int8_f32",
        &[&input_buf, &quant_buf, &scales_buf, &params_buf],
        3,
        1,
        (num_tokens as u32).div_ceil(256),
    )?;

    Ok((quantized, scales))
}

pub(super) fn dequantize_kv_int8_impl(
    client: &WgpuClient,
    quantized: &Tensor<WgpuRuntime>,
    scales: &Tensor<WgpuRuntime>,
    num_tokens: usize,
    head_dim: usize,
) -> Result<Tensor<WgpuRuntime>> {
    validate_f32(quantized, "dequantize_kv_int8")?;
    validate_f32(scales, "dequantize_kv_int8")?;

    let output =
        Tensor::<WgpuRuntime>::zeros(&[num_tokens, head_dim], DType::F32, quantized.device())?;

    let quant_buf = get_buffer(quantized.storage().ptr()).ok_or_else(|| Error::KernelError {
        reason: "quantized buffer not found".into(),
    })?;
    let scales_buf = get_buffer(scales.storage().ptr()).ok_or_else(|| Error::KernelError {
        reason: "scales buffer not found".into(),
    })?;
    let out_buf = get_buffer(output.storage().ptr()).ok_or_else(|| Error::KernelError {
        reason: "output buffer not found".into(),
    })?;

    let params = QuantParams {
        num_tokens: num_tokens as u32,
        head_dim: head_dim as u32,
        group_size: 0,
        mode: 1,
    };
    let params_buf = create_params_buf(client, &params);

    // Shader bindings: 0=input(read), 1=scales(read), 2=output(rw)
    dispatch(
        client,
        DEQUANT_INT8_SRC,
        "dequantize_kv_int8_f32",
        &[&quant_buf, &scales_buf, &out_buf, &params_buf],
        3,
        2,
        (num_tokens as u32).div_ceil(256),
    )?;

    Ok(output)
}
