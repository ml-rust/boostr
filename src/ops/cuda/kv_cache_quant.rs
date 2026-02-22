//! KV cache quantization CUDA launchers
//!
//! Dispatches to compiled PTX kernels for FP8, INT4, and INT8 quantization.
//! Kernels: kv_cache_fp8.cu, kv_cache_int4.cu, kv_cache_quant.cu, kv_cache_fp8_bwd.cu

use crate::error::{Error, Result};
use crate::ops::traits::kv_cache_quant::{Int4GroupSize, KvCacheQuantOps};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::kernels::{self, KV_CACHE_FP8_MODULE, KV_CACHE_INT4_MODULE, KV_CACHE_QUANT_MODULE};

impl KvCacheQuantOps<CudaRuntime> for CudaClient {
    fn quantize_kv_fp8_per_token(
        &self,
        input: &Tensor<CudaRuntime>,
        num_tokens: usize,
        head_dim: usize,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let dtype = input.dtype();
        let kernel_name = match dtype {
            DType::F16 => "quantize_kv_fp8_per_token_fp16",
            DType::BF16 => "quantize_kv_fp8_per_token_bf16",
            _ => {
                return Err(Error::KernelError {
                    reason: format!("FP8 quant: unsupported input dtype {dtype:?}, need F16/BF16"),
                });
            }
        };

        let device = input.device();
        let device_index = device.id();
        let module =
            kernels::get_or_load_module(self.context(), device_index, KV_CACHE_FP8_MODULE)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        // Output: FP8 (u8) same shape, scales: [num_tokens] F32
        let quantized = Tensor::<CudaRuntime>::empty(&[num_tokens, head_dim], DType::U8, device);
        let scales = Tensor::<CudaRuntime>::empty(&[num_tokens], DType::F32, device);

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4,
        };

        let q_ptr = quantized.ptr();
        let i_ptr = input.ptr();
        let s_ptr = scales.ptr();
        let batch_i32 = 1i32; // Flattened as [num_tokens, head_dim]
        let nkh_i32 = 1i32;
        let sl_i32 = num_tokens as i32;
        let hd_i32 = head_dim as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&q_ptr);
            builder.arg(&i_ptr);
            builder.arg(&s_ptr);
            builder.arg(&batch_i32);
            builder.arg(&nkh_i32);
            builder.arg(&sl_i32);
            builder.arg(&hd_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("FP8 per-token quant failed: {e:?}"),
            })?;
        }

        Ok((quantized, scales))
    }

    fn dequantize_kv_fp8_per_token(
        &self,
        quantized: &Tensor<CudaRuntime>,
        scales: &Tensor<CudaRuntime>,
        num_tokens: usize,
        head_dim: usize,
        output_dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        let kernel_name = match output_dtype {
            DType::F16 => "dequantize_kv_fp8_per_token_fp16",
            DType::BF16 => "dequantize_kv_fp8_per_token_bf16",
            _ => {
                return Err(Error::KernelError {
                    reason: format!("FP8 dequant: unsupported output dtype {output_dtype:?}"),
                });
            }
        };

        let device = quantized.device();
        let device_index = device.id();
        let module =
            kernels::get_or_load_module(self.context(), device_index, KV_CACHE_FP8_MODULE)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        let output = Tensor::<CudaRuntime>::empty(&[num_tokens, head_dim], output_dtype, device);

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let o_ptr = output.ptr();
        let q_ptr = quantized.ptr();
        let s_ptr = scales.ptr();
        let batch_i32 = 1i32;
        let nkh_i32 = 1i32;
        let sl_i32 = num_tokens as i32;
        let hd_i32 = head_dim as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&o_ptr);
            builder.arg(&q_ptr);
            builder.arg(&s_ptr);
            builder.arg(&batch_i32);
            builder.arg(&nkh_i32);
            builder.arg(&sl_i32);
            builder.arg(&hd_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("FP8 per-token dequant failed: {e:?}"),
            })?;
        }

        Ok(output)
    }

    fn quantize_kv_int4(
        &self,
        input: &Tensor<CudaRuntime>,
        num_tokens: usize,
        head_dim: usize,
        group_size: Int4GroupSize,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let dtype = input.dtype();
        let dtype_suffix = match dtype {
            DType::F32 => "fp32",
            DType::F16 => "fp16",
            DType::BF16 => "bf16",
            _ => {
                return Err(Error::KernelError {
                    reason: format!("INT4 quant: unsupported dtype {dtype:?}"),
                });
            }
        };

        let kernel_name = format!("quantize_kv_int4_per_group_{dtype_suffix}");
        let device = input.device();
        let device_index = device.id();
        let module =
            kernels::get_or_load_module(self.context(), device_index, KV_CACHE_INT4_MODULE)?;
        let func = kernels::get_kernel_function(&module, &kernel_name)?;

        let gs = group_size as usize;
        let total = num_tokens * head_dim;
        let num_groups = (total + gs - 1) / gs;

        let packed = Tensor::<CudaRuntime>::empty(&[num_tokens, head_dim / 2], DType::U8, device);
        let scales_t = Tensor::<CudaRuntime>::empty(&[num_groups], DType::F32, device);
        let zeros_t = Tensor::<CudaRuntime>::empty(&[num_groups], DType::F32, device);

        let cfg = LaunchConfig {
            grid_dim: (num_groups as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4,
        };

        let i_ptr = input.ptr();
        let p_ptr = packed.ptr();
        let s_ptr = scales_t.ptr();
        let z_ptr = zeros_t.ptr();
        let nt_i32 = num_tokens as i32;
        let hd_i32 = head_dim as i32;
        let gs_i32 = gs as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&i_ptr);
            builder.arg(&p_ptr);
            builder.arg(&s_ptr);
            builder.arg(&z_ptr);
            builder.arg(&nt_i32);
            builder.arg(&hd_i32);
            builder.arg(&gs_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("INT4 quant failed: {e:?}"),
            })?;
        }

        Ok((packed, scales_t, zeros_t))
    }

    fn dequantize_kv_int4(
        &self,
        packed: &Tensor<CudaRuntime>,
        scales: &Tensor<CudaRuntime>,
        zeros: &Tensor<CudaRuntime>,
        num_tokens: usize,
        head_dim: usize,
        group_size: Int4GroupSize,
    ) -> Result<Tensor<CudaRuntime>> {
        let device = packed.device();
        let device_index = device.id();
        let module =
            kernels::get_or_load_module(self.context(), device_index, KV_CACHE_INT4_MODULE)?;
        let func = kernels::get_kernel_function(&module, "dequantize_kv_int4_per_group_fp32")?;

        let gs = group_size as usize;
        let total = num_tokens * head_dim;
        let num_groups = (total + gs - 1) / gs;

        let output = Tensor::<CudaRuntime>::empty(&[num_tokens, head_dim], DType::F32, device);

        let cfg = LaunchConfig {
            grid_dim: (num_groups as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let p_ptr = packed.ptr();
        let o_ptr = output.ptr();
        let s_ptr = scales.ptr();
        let z_ptr = zeros.ptr();
        let nt_i32 = num_tokens as i32;
        let hd_i32 = head_dim as i32;
        let gs_i32 = gs as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&p_ptr);
            builder.arg(&o_ptr);
            builder.arg(&s_ptr);
            builder.arg(&z_ptr);
            builder.arg(&nt_i32);
            builder.arg(&hd_i32);
            builder.arg(&gs_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("INT4 dequant failed: {e:?}"),
            })?;
        }

        Ok(output)
    }

    fn quantize_kv_int8(
        &self,
        input: &Tensor<CudaRuntime>,
        num_tokens: usize,
        head_dim: usize,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let dtype = input.dtype();
        let dtype_suffix = match dtype {
            DType::F32 => "fp32",
            DType::F16 => "fp16",
            DType::BF16 => "bf16",
            _ => {
                return Err(Error::KernelError {
                    reason: format!("INT8 quant: unsupported dtype {dtype:?}"),
                });
            }
        };

        let kernel_name = format!("quantize_kv_int8_per_token_{dtype_suffix}");
        let device = input.device();
        let device_index = device.id();
        let module =
            kernels::get_or_load_module(self.context(), device_index, KV_CACHE_QUANT_MODULE)?;
        let func = kernels::get_kernel_function(&module, &kernel_name)?;

        let quantized = Tensor::<CudaRuntime>::empty(&[num_tokens, head_dim], DType::I8, device);
        let scales = Tensor::<CudaRuntime>::empty(&[num_tokens], DType::F32, device);

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4,
        };

        let i_ptr = input.ptr();
        let q_ptr = quantized.ptr();
        let s_ptr = scales.ptr();
        let nt_i32 = num_tokens as i32;
        let hd_i32 = head_dim as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&i_ptr);
            builder.arg(&q_ptr);
            builder.arg(&s_ptr);
            builder.arg(&nt_i32);
            builder.arg(&hd_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("INT8 quant failed: {e:?}"),
            })?;
        }

        Ok((quantized, scales))
    }

    fn dequantize_kv_int8(
        &self,
        quantized: &Tensor<CudaRuntime>,
        scales: &Tensor<CudaRuntime>,
        num_tokens: usize,
        head_dim: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        let device = quantized.device();
        let device_index = device.id();
        let module =
            kernels::get_or_load_module(self.context(), device_index, KV_CACHE_QUANT_MODULE)?;
        let func = kernels::get_kernel_function(&module, "dequantize_kv_int8_per_token_fp32")?;

        let output = Tensor::<CudaRuntime>::empty(&[num_tokens, head_dim], DType::F32, device);

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let q_ptr = quantized.ptr();
        let o_ptr = output.ptr();
        let s_ptr = scales.ptr();
        let nt_i32 = num_tokens as i32;
        let hd_i32 = head_dim as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&q_ptr);
            builder.arg(&o_ptr);
            builder.arg(&s_ptr);
            builder.arg(&nt_i32);
            builder.arg(&hd_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("INT8 dequant failed: {e:?}"),
            })?;
        }

        Ok(output)
    }
}
