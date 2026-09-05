//! KV cache quantization CUDA launchers
//!
//! Dispatches to compiled PTX kernels for FP8, INT4, and INT8 quantization.
//! Kernels: kv_cache_fp8.cu, kv_cache_int4.cu, kv_cache_quant.cu, kv_cache_fp8_bwd.cu

use crate::error::{Error, Result};
use crate::ops::traits::cache::kv_cache_quant::{Int4GroupSize, KvCacheQuantOps};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::ops::cuda::kernels::{self, KV_CACHE_FP8_MODULE, KV_CACHE_QUANT_MODULE};

impl KvCacheQuantOps<CudaRuntime> for CudaClient {
    fn quantize_kv_fp8_per_token(
        &self,
        input: &Tensor<CudaRuntime>,
        num_tokens: usize,
        head_dim: usize,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        // kv_cache_quant.cu's quantize_kv_fp8_per_token_fp32 could quantize
        // F32 input directly, at full precision. It stays unused here: no
        // matching dequantize_kv_fp8_per_token_fp32 kernel exists anywhere
        // (only fp16/bf16 dequant, in kv_cache_fp8.cu), so a genuine fp32
        // quantize would produce output only readable back through a
        // different-precision dequant path. Casting to F16 first keeps
        // quantize and dequant symmetric at the same representable
        // precision.
        let input_to_use = if input.dtype() == DType::F32 {
            let vars = Var::new(input.clone(), false);
            let cast_var =
                numr::autograd::var_cast(&vars, DType::F16, self).map_err(Error::Numr)?;
            cast_var.tensor().clone()
        } else {
            input.clone()
        };

        let dtype = input_to_use.dtype();
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
        let quantized = Tensor::<CudaRuntime>::empty(&[num_tokens, head_dim], DType::U8, device)?;
        let scales = Tensor::<CudaRuntime>::empty(&[num_tokens], DType::F32, device)?;

        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4,
        };

        let q_ptr = quantized.ptr();
        let i_ptr = input_to_use.ptr();
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

    fn quantize_kv_fp8_per_head(
        &self,
        input: &Tensor<CudaRuntime>,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let dtype = input.dtype();
        let dtype_suffix = match dtype {
            DType::F32 => "fp32",
            DType::F16 => "fp16",
            DType::BF16 => "bf16",
            _ => {
                return Err(Error::KernelError {
                    reason: format!("FP8 per-head quant: unsupported dtype {dtype:?}"),
                });
            }
        };

        let kernel_name = format!("quantize_kv_fp8_per_head_{dtype_suffix}");
        let device = input.device();
        let device_index = device.id();

        let module =
            kernels::get_or_load_module(self.context(), device_index, KV_CACHE_QUANT_MODULE)?;
        let func = kernels::get_kernel_function(&module, &kernel_name)?;

        // Output: FP8 (u8) same shape, scales: [num_heads] F32
        let quantized =
            Tensor::<CudaRuntime>::empty(&[num_heads, seq_len, head_dim], DType::U8, device)?;
        let scales = Tensor::<CudaRuntime>::empty(&[num_heads], DType::F32, device)?;

        let cfg = LaunchConfig {
            grid_dim: (num_heads as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4,
        };

        let i_ptr = input.ptr();
        let q_ptr = quantized.ptr();
        let s_ptr = scales.ptr();
        let nh_i32 = num_heads as i32;
        let sl_i32 = seq_len as i32;
        let hd_i32 = head_dim as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&i_ptr);
            builder.arg(&q_ptr);
            builder.arg(&s_ptr);
            builder.arg(&nh_i32);
            builder.arg(&sl_i32);
            builder.arg(&hd_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("FP8 per-head quant failed: {e:?}"),
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
        // CUDA kernels only support F16/BF16 output; cast F32 to F16 then back
        let target_dtype = match output_dtype {
            DType::F32 => DType::F16, // We'll cast back to F32 later
            other => other,
        };

        let kernel_name = match target_dtype {
            DType::F16 => "dequantize_kv_fp8_per_token_fp16",
            DType::BF16 => "dequantize_kv_fp8_per_token_bf16",
            _ => {
                return Err(Error::KernelError {
                    reason: format!("FP8 dequant: unsupported output dtype {target_dtype:?}"),
                });
            }
        };

        let device = quantized.device();
        let device_index = device.id();

        let module =
            kernels::get_or_load_module(self.context(), device_index, KV_CACHE_FP8_MODULE)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        let output = Tensor::<CudaRuntime>::empty(&[num_tokens, head_dim], target_dtype, device)?;

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

        // Cast F16 back to F32 if needed
        if output_dtype == DType::F32 && target_dtype == DType::F16 {
            let output_var = Var::new(output, false);
            let cast_var =
                numr::autograd::var_cast(&output_var, DType::F32, self).map_err(Error::Numr)?;
            Ok(cast_var.tensor().clone())
        } else {
            Ok(output)
        }
    }

    fn quantize_kv_fp8_per_tensor(
        &self,
        input: &Tensor<CudaRuntime>,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        super::kv_cache_fp8_per_tensor::quantize_kv_fp8_per_tensor_impl(self, input)
    }

    fn dequantize_kv_fp8_per_tensor(
        &self,
        quantized: &Tensor<CudaRuntime>,
        scale: &Tensor<CudaRuntime>,
        output_dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        super::kv_cache_fp8_per_tensor::dequantize_kv_fp8_per_tensor_impl(
            self,
            quantized,
            scale,
            output_dtype,
        )
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
        super::kv_cache_int4::quantize_kv_int4_impl(self, input, num_tokens, head_dim, group_size)
    }

    fn dequantize_kv_int4(
        &self,
        packed: &Tensor<CudaRuntime>,
        scales: &Tensor<CudaRuntime>,
        zeros: &Tensor<CudaRuntime>,
        num_tokens: usize,
        head_dim: usize,
        group_size: Int4GroupSize,
        output_dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
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
        input: &Tensor<CudaRuntime>,
        num_tokens: usize,
        head_dim: usize,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        super::kv_cache_int8::quantize_kv_int8_impl(self, input, num_tokens, head_dim)
    }

    fn dequantize_kv_int8(
        &self,
        quantized: &Tensor<CudaRuntime>,
        scales: &Tensor<CudaRuntime>,
        num_tokens: usize,
        head_dim: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        super::kv_cache_int8::dequantize_kv_int8_impl(self, quantized, scales, num_tokens, head_dim)
    }

    fn kv_fp8_bwd_per_tensor(
        &self,
        grad_output: &Tensor<CudaRuntime>,
        kv_fp8: &Tensor<CudaRuntime>,
        scale: f32,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        super::kv_cache_fp8_bwd::kv_fp8_bwd_per_tensor_impl(self, grad_output, kv_fp8, scale)
    }

    fn kv_fp8_bwd_per_token(
        &self,
        grad_output: &Tensor<CudaRuntime>,
        kv_fp8: &Tensor<CudaRuntime>,
        scales: &Tensor<CudaRuntime>,
        batch: usize,
        num_kv_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        super::kv_cache_fp8_bwd::kv_fp8_bwd_per_token_impl(
            self,
            grad_output,
            kv_fp8,
            scales,
            batch,
            num_kv_heads,
            seq_len,
            head_dim,
        )
    }
}
