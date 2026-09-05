//! CUDA launcher for per-tensor FP8 KV cache quantize/dequantize.
//!
//! Split out of `kv_cache_quant.rs` to keep that file under the `cuda/*.rs`
//! line limit. Kernels: `kv_cache_fp8.cu`.
//!
//! Quantize is a three-launch pipeline (`_find_max`, `_finalize_scale`, then
//! the quantize kernel itself) because a max-abs reduced over the whole
//! tensor must finish before any element is quantized, and CUDA has no
//! grid-wide barrier inside one kernel. See the kernel file's header comment
//! for why a single-launch version is wrong past one block.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, KV_CACHE_FP8_MODULE};
use crate::readback::scalar::scalar_f32;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

const BLOCK: u32 = 256;

fn check_fp16(t: &Tensor<CudaRuntime>, arg: &'static str) -> Result<()> {
    if t.dtype() != DType::F16 {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!(
                "FP8 per-tensor: CUDA kernel only supports F16 input, got {:?}",
                t.dtype()
            ),
        });
    }
    Ok(())
}

/// Quantize a whole tensor to FP8 (E4M3) with a single tensor-wide scale.
/// Returns `(quantized, scale)` where `scale` is a 1-element F32 tensor.
pub(super) fn quantize_kv_fp8_per_tensor_impl(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    check_fp16(input, "input")?;

    let total_elements = input.numel();
    let device = input.device();
    let device_index = device.id();
    let module = kernels::get_or_load_module(client.context(), device_index, KV_CACHE_FP8_MODULE)?;

    // Zero-initialized: stage 1 folds every block's local max into this
    // scalar via atomicMax, so it must start at zero (all-block-max input is
    // fabsf, always non-negative).
    let scale = Tensor::<CudaRuntime>::zeros(&[1], DType::F32, device)?;
    let quantized = Tensor::<CudaRuntime>::empty(input.shape(), DType::FP8E4M3, device)?;

    let cfg = LaunchConfig {
        grid_dim: ((total_elements as u32).div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let total_i32 = total_elements as i32;
    let i_ptr = input.ptr();
    let s_ptr = scale.ptr();
    let q_ptr = quantized.ptr();

    unsafe {
        let find_max =
            kernels::get_kernel_function(&module, "quantize_kv_fp8_per_tensor_fp16_find_max")?;
        let mut builder = client.stream().launch_builder(&find_max);
        builder.arg(&i_ptr);
        builder.arg(&s_ptr);
        builder.arg(&total_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("FP8 per-tensor quant find_max failed: {e:?}"),
        })?;

        let finalize = kernels::get_kernel_function(
            &module,
            "quantize_kv_fp8_per_tensor_fp16_finalize_scale",
        )?;
        let mut builder = client.stream().launch_builder(&finalize);
        builder.arg(&s_ptr);
        builder
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(|e| Error::KernelError {
                reason: format!("FP8 per-tensor quant finalize_scale failed: {e:?}"),
            })?;

        let quantize = kernels::get_kernel_function(&module, "quantize_kv_fp8_per_tensor_fp16")?;
        let mut builder = client.stream().launch_builder(&quantize);
        builder.arg(&q_ptr);
        builder.arg(&i_ptr);
        builder.arg(&s_ptr);
        builder.arg(&total_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("FP8 per-tensor quant failed: {e:?}"),
        })?;
    }

    Ok((quantized, scale))
}

/// Dequantize an FP8 (E4M3) tensor back to `output_dtype` using a single
/// tensor-wide scale.
///
/// `dequantize_kv_fp8_per_tensor_fp16` takes `scale` BY VALUE (a plain
/// `float` parameter, not a device pointer), so the 1-element `scale` tensor
/// must be read back to the host before the launch. That is a single scalar,
/// not bulk tensor data, so it does not violate the no-GPU-to-CPU-transfer
/// rule for library code.
pub(super) fn dequantize_kv_fp8_per_tensor_impl(
    client: &CudaClient,
    quantized: &Tensor<CudaRuntime>,
    scale: &Tensor<CudaRuntime>,
    output_dtype: DType,
) -> Result<Tensor<CudaRuntime>> {
    if quantized.dtype() != DType::FP8E4M3 {
        return Err(Error::InvalidArgument {
            arg: "quantized",
            reason: format!(
                "FP8 per-tensor dequant: quantized must be FP8E4M3, got {:?}",
                quantized.dtype()
            ),
        });
    }
    if output_dtype != DType::F16 {
        return Err(Error::InvalidArgument {
            arg: "output_dtype",
            reason: format!(
                "FP8 per-tensor: CUDA kernel only supports F16 output, got {output_dtype:?}"
            ),
        });
    }

    let scale_host = scalar_f32(client, scale)?;
    let total_elements = quantized.numel();
    let device = quantized.device();
    let device_index = device.id();
    let module = kernels::get_or_load_module(client.context(), device_index, KV_CACHE_FP8_MODULE)?;
    let func = kernels::get_kernel_function(&module, "dequantize_kv_fp8_per_tensor_fp16")?;

    let output = Tensor::<CudaRuntime>::empty(quantized.shape(), DType::F16, device)?;

    let cfg = LaunchConfig {
        grid_dim: ((total_elements as u32).div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let o_ptr = output.ptr();
    let q_ptr = quantized.ptr();
    let total_i32 = total_elements as i32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&o_ptr);
        builder.arg(&q_ptr);
        builder.arg(&scale_host);
        builder.arg(&total_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("FP8 per-tensor dequant failed: {e:?}"),
        })?;
    }

    Ok(output)
}
