//! WebGPU implementation of KvCacheQuantOps
//!
//! FP8, INT4, and INT8 KV cache quantization.
//! Quantized values stored as F32 (WebGPU has no native INT8/FP8).

use crate::error::{Error, Result};
use crate::ops::traits::{Int4GroupSize, KvCacheQuantOps};
use numr::dtype::DType;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::BufferUsages;

const QUANT_FP8_SRC: &str = include_str!("../shaders/cache/kv_cache_quant_fp8.wgsl");
const DEQUANT_FP8_SRC: &str = include_str!("../shaders/cache/kv_cache_dequant_fp8.wgsl");
pub(super) const QUANT_INT4_SRC: &str = include_str!("../shaders/cache/kv_cache_quant_int4.wgsl");
pub(super) const DEQUANT_INT4_SRC: &str =
    include_str!("../shaders/cache/kv_cache_dequant_int4.wgsl");
const QUANT_INT8_SRC: &str = include_str!("../shaders/cache/kv_cache_quant_int8.wgsl");
const DEQUANT_INT8_SRC: &str = include_str!("../shaders/cache/kv_cache_dequant_int8.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct QuantParams {
    pub(super) num_tokens: u32,
    pub(super) head_dim: u32,
    pub(super) group_size: u32,
    pub(super) mode: u32,
}

pub(super) fn validate_f32(t: &Tensor<WgpuRuntime>, op: &str) -> Result<()> {
    if t.dtype() != DType::F32 {
        return Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!("{}: WebGPU requires F32, got {:?}", op, t.dtype()),
        });
    }
    Ok(())
}

pub(super) fn create_params_buf(client: &WgpuClient, params: &QuantParams) -> wgpu::Buffer {
    let buf = client.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("quant_params"),
        size: std::mem::size_of::<QuantParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .wgpu_queue()
        .write_buffer(&buf, 0, bytemuck::bytes_of(params));
    buf
}

pub(super) fn dispatch(
    client: &WgpuClient,
    shader_src: &'static str,
    entry: &'static str,
    bufs: &[&wgpu::Buffer],
    num_storage: u32,
    num_readonly: u32,
    workgroups: u32,
) -> Result<()> {
    let cache = client.pipeline_cache();
    let module = cache.get_or_create_module(entry, shader_src);
    let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
        num_storage_buffers: num_storage,
        num_uniform_buffers: 1,
        num_readonly_storage: num_readonly,
    });
    let pipeline = cache.get_or_create_pipeline(entry, entry, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, bufs);

    let mut encoder = client
        .wgpu_device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(entry) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(entry),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    client
        .wgpu_queue()
        .submit(std::iter::once(encoder.finish()));
    Ok(())
}

impl KvCacheQuantOps<WgpuRuntime> for WgpuClient {
    fn quantize_kv_fp8_per_token(
        &self,
        input: &Tensor<WgpuRuntime>,
        num_tokens: usize,
        head_dim: usize,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        validate_f32(input, "quantize_kv_fp8_per_token")?;

        let quantized =
            Tensor::<WgpuRuntime>::zeros(&[num_tokens, head_dim], DType::F32, input.device())?;
        let scales = Tensor::<WgpuRuntime>::zeros(&[num_tokens], DType::F32, input.device())?;

        let input_buf = get_buffer(input.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "input buffer not found".into(),
        })?;
        let quant_buf =
            get_buffer(quantized.storage().ptr()).ok_or_else(|| Error::KernelError {
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
        let params_buf = create_params_buf(self, &params);

        // Shader bindings: 0=input(read), 1=output(rw), 2=scales(rw)
        dispatch(
            self,
            QUANT_FP8_SRC,
            "quantize_kv_fp8_per_token_f32",
            &[&input_buf, &quant_buf, &scales_buf, &params_buf],
            3, // num_storage
            1, // num_readonly (binding 0 = input)
            (num_tokens as u32).div_ceil(256),
        )?;

        Ok((quantized, scales))
    }

    fn quantize_kv_fp8_per_head(
        &self,
        input: &Tensor<WgpuRuntime>,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        validate_f32(input, "quantize_kv_fp8_per_head")?;

        // The FP8 shader reduces over a per-unit contiguous span sized by
        // `head_dim` in its params. A head's `seq_len * head_dim` span is a
        // wider span of the same kind, so no per-head branch exists.
        // `quantize_kv_fp8_per_head` reuses the per-token entry point with
        // num_units = num_heads and span = seq_len * head_dim.
        let span = seq_len * head_dim;

        let quantized = Tensor::<WgpuRuntime>::zeros(
            &[num_heads, seq_len, head_dim],
            DType::F32,
            input.device(),
        )?;
        let scales = Tensor::<WgpuRuntime>::zeros(&[num_heads], DType::F32, input.device())?;

        let input_buf = get_buffer(input.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "input buffer not found".into(),
        })?;
        let quant_buf =
            get_buffer(quantized.storage().ptr()).ok_or_else(|| Error::KernelError {
                reason: "quantized buffer not found".into(),
            })?;
        let scales_buf = get_buffer(scales.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "scales buffer not found".into(),
        })?;

        let params = QuantParams {
            num_tokens: num_heads as u32,
            head_dim: span as u32,
            group_size: 0,
            mode: 1,
        };
        let params_buf = create_params_buf(self, &params);

        // Shader bindings: 0=input(read), 1=output(rw), 2=scales(rw)
        dispatch(
            self,
            QUANT_FP8_SRC,
            "quantize_kv_fp8_per_token_f32",
            &[&input_buf, &quant_buf, &scales_buf, &params_buf],
            3, // num_storage
            1, // num_readonly (binding 0 = input)
            (num_heads as u32).div_ceil(256),
        )?;

        Ok((quantized, scales))
    }

    fn dequantize_kv_fp8_per_token(
        &self,
        quantized: &Tensor<WgpuRuntime>,
        scales: &Tensor<WgpuRuntime>,
        num_tokens: usize,
        head_dim: usize,
        output_dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_f32(quantized, "dequantize_kv_fp8_per_token")?;
        validate_f32(scales, "dequantize_kv_fp8_per_token")?;
        if output_dtype != DType::F32 {
            return Err(Error::InvalidArgument {
                arg: "output_dtype",
                reason: format!(
                    "dequantize_kv_fp8_per_token: WebGPU only supports F32 output, got {output_dtype:?}"
                ),
            });
        }

        let output =
            Tensor::<WgpuRuntime>::zeros(&[num_tokens, head_dim], DType::F32, quantized.device())?;

        let quant_buf =
            get_buffer(quantized.storage().ptr()).ok_or_else(|| Error::KernelError {
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
        let params_buf = create_params_buf(self, &params);

        // Shader bindings: 0=input(read), 1=scales(read), 2=output(rw)
        dispatch(
            self,
            DEQUANT_FP8_SRC,
            "dequantize_kv_fp8_per_token_f32",
            &[&quant_buf, &scales_buf, &out_buf, &params_buf],
            3, // num_storage
            2, // num_readonly (bindings 0,1)
            (num_tokens as u32).div_ceil(256),
        )?;

        Ok(output)
    }

    fn quantize_kv_int4(
        &self,
        input: &Tensor<WgpuRuntime>,
        num_tokens: usize,
        head_dim: usize,
        group_size: Int4GroupSize,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        super::kv_cache_int4::quantize_kv_int4_impl(self, input, num_tokens, head_dim, group_size)
    }

    fn dequantize_kv_int4(
        &self,
        packed: &Tensor<WgpuRuntime>,
        scales: &Tensor<WgpuRuntime>,
        zeros: &Tensor<WgpuRuntime>,
        num_tokens: usize,
        head_dim: usize,
        group_size: Int4GroupSize,
        output_dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        super::kv_cache_int4::dequantize_kv_int4_impl(
            self,
            packed,
            scales,
            zeros,
            num_tokens,
            head_dim,
            group_size,
            output_dtype,
        )
    }

    fn quantize_kv_int8(
        &self,
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
        let quant_buf =
            get_buffer(quantized.storage().ptr()).ok_or_else(|| Error::KernelError {
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
        let params_buf = create_params_buf(self, &params);

        // Shader bindings: 0=input(read), 1=output(rw), 2=scales(rw)
        dispatch(
            self,
            QUANT_INT8_SRC,
            "quantize_kv_int8_f32",
            &[&input_buf, &quant_buf, &scales_buf, &params_buf],
            3,
            1,
            (num_tokens as u32).div_ceil(256),
        )?;

        Ok((quantized, scales))
    }

    fn dequantize_kv_int8(
        &self,
        quantized: &Tensor<WgpuRuntime>,
        scales: &Tensor<WgpuRuntime>,
        num_tokens: usize,
        head_dim: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_f32(quantized, "dequantize_kv_int8")?;
        validate_f32(scales, "dequantize_kv_int8")?;

        let output =
            Tensor::<WgpuRuntime>::zeros(&[num_tokens, head_dim], DType::F32, quantized.device())?;

        let quant_buf =
            get_buffer(quantized.storage().ptr()).ok_or_else(|| Error::KernelError {
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
        let params_buf = create_params_buf(self, &params);

        // Shader bindings: 0=input(read), 1=scales(read), 2=output(rw)
        dispatch(
            self,
            DEQUANT_INT8_SRC,
            "dequantize_kv_int8_f32",
            &[&quant_buf, &scales_buf, &out_buf, &params_buf],
            3,
            2,
            (num_tokens as u32).div_ceil(256),
        )?;

        Ok(output)
    }

    fn kv_fp8_bwd_per_tensor(
        &self,
        _grad_output: &Tensor<WgpuRuntime>,
        _kv_fp8: &Tensor<WgpuRuntime>,
        _scale: f32,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        Err(Error::KernelError {
            reason: "kv_fp8_bwd_per_tensor not implemented on WebGPU: no WGSL shader for it".into(),
        })
    }

    fn kv_fp8_bwd_per_token(
        &self,
        _grad_output: &Tensor<WgpuRuntime>,
        _kv_fp8: &Tensor<WgpuRuntime>,
        _scales: &Tensor<WgpuRuntime>,
        _batch: usize,
        _num_kv_heads: usize,
        _seq_len: usize,
        _head_dim: usize,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        Err(Error::KernelError {
            reason: "kv_fp8_bwd_per_token not implemented on WebGPU: no WGSL shader for it".into(),
        })
    }
}
