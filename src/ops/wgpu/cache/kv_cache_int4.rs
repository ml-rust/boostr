//! WebGPU launchers for INT4 KV cache quantize/dequantize.
//!
//! Split out of `kv_cache_quant.rs` to keep that file under the `wgpu/*.rs`
//! line limit. Shaders: kv_cache_quant_int4.wgsl, kv_cache_dequant_int4.wgsl.

use super::kv_cache_quant::{
    DEQUANT_INT4_SRC, QUANT_INT4_SRC, QuantParams, create_params_buf, dispatch, validate_f32,
};
use crate::error::{Error, Result};
use crate::ops::traits::Int4GroupSize;
use numr::dtype::DType;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;

pub(super) fn quantize_kv_int4_impl(
    client: &WgpuClient,
    input: &Tensor<WgpuRuntime>,
    num_tokens: usize,
    head_dim: usize,
    group_size: Int4GroupSize,
) -> Result<(
    Tensor<WgpuRuntime>,
    Tensor<WgpuRuntime>,
    Tensor<WgpuRuntime>,
)> {
    validate_f32(input, "quantize_kv_int4")?;

    let group_sz = group_size as usize;
    let num_groups = (num_tokens * head_dim) / group_sz;

    // Packed uses u32 via DType::I32 (same size), but we use F32 for WebGPU compatibility
    let packed =
        Tensor::<WgpuRuntime>::zeros(&[num_tokens, head_dim / 2], DType::F32, input.device())?;
    let scales = Tensor::<WgpuRuntime>::zeros(&[num_groups], DType::F32, input.device())?;
    let zeros = Tensor::<WgpuRuntime>::zeros(&[num_groups], DType::F32, input.device())?;

    let input_buf = get_buffer(input.storage().ptr()).ok_or_else(|| Error::KernelError {
        reason: "input buffer not found".into(),
    })?;
    let packed_buf = get_buffer(packed.storage().ptr()).ok_or_else(|| Error::KernelError {
        reason: "packed buffer not found".into(),
    })?;
    let scales_buf = get_buffer(scales.storage().ptr()).ok_or_else(|| Error::KernelError {
        reason: "scales buffer not found".into(),
    })?;
    let zeros_buf = get_buffer(zeros.storage().ptr()).ok_or_else(|| Error::KernelError {
        reason: "zeros buffer not found".into(),
    })?;

    let params = QuantParams {
        num_tokens: num_tokens as u32,
        head_dim: head_dim as u32,
        group_size: group_sz as u32,
        mode: 0,
    };
    let params_buf = create_params_buf(client, &params);

    // Shader bindings: 0=input(read), 1=packed(rw), 2=scales(rw), 3=zeros(rw)
    dispatch(
        client,
        QUANT_INT4_SRC,
        "quantize_kv_int4_f32",
        &[
            &input_buf,
            &packed_buf,
            &scales_buf,
            &zeros_buf,
            &params_buf,
        ],
        4,
        1,
        (num_groups as u32).div_ceil(256),
    )?;

    Ok((packed, scales, zeros))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dequantize_kv_int4_impl(
    client: &WgpuClient,
    packed: &Tensor<WgpuRuntime>,
    scales: &Tensor<WgpuRuntime>,
    zeros: &Tensor<WgpuRuntime>,
    num_tokens: usize,
    head_dim: usize,
    group_size: Int4GroupSize,
    output_dtype: DType,
) -> Result<Tensor<WgpuRuntime>> {
    validate_f32(packed, "dequantize_kv_int4")?;
    validate_f32(scales, "dequantize_kv_int4")?;
    validate_f32(zeros, "dequantize_kv_int4")?;
    if output_dtype != DType::F32 {
        return Err(Error::InvalidArgument {
            arg: "output_dtype",
            reason: format!(
                "dequantize_kv_int4: WebGPU only supports F32 output, got {output_dtype:?}"
            ),
        });
    }

    let output =
        Tensor::<WgpuRuntime>::zeros(&[num_tokens, head_dim], DType::F32, packed.device())?;

    let packed_buf = get_buffer(packed.storage().ptr()).ok_or_else(|| Error::KernelError {
        reason: "packed buffer not found".into(),
    })?;
    let scales_buf = get_buffer(scales.storage().ptr()).ok_or_else(|| Error::KernelError {
        reason: "scales buffer not found".into(),
    })?;
    let zeros_buf = get_buffer(zeros.storage().ptr()).ok_or_else(|| Error::KernelError {
        reason: "zeros buffer not found".into(),
    })?;
    let out_buf = get_buffer(output.storage().ptr()).ok_or_else(|| Error::KernelError {
        reason: "output buffer not found".into(),
    })?;

    let group_sz = group_size as usize;
    let num_groups = (num_tokens * head_dim) / group_sz;
    let params = QuantParams {
        num_tokens: num_tokens as u32,
        head_dim: head_dim as u32,
        group_size: group_sz as u32,
        mode: 0,
    };
    let params_buf = create_params_buf(client, &params);

    // Shader bindings: 0=packed(read), 1=scales(read), 2=zeros(read), 3=output(rw)
    dispatch(
        client,
        DEQUANT_INT4_SRC,
        "dequantize_kv_int4_f32",
        &[&packed_buf, &scales_buf, &zeros_buf, &out_buf, &params_buf],
        4,
        3,
        (num_groups as u32).div_ceil(256),
    )?;

    Ok(output)
}
