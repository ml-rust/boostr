//! CUDA implementation of CalibrationOps
//!
//! - awq_channel_scores: fused kernel (2 launches: act_scale + score_reduce)
//! - fisher_information: fused kernel (1 launch)
//! - gptq_hessian_update: delegates to impl_generic (numr matmul is CUDA-optimized)
//! - gptq_quantize_column: delegates to impl_generic (sequential column loop)

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, CALIBRATION_MODULE};
use crate::ops::impl_generic::quantization::calibration::{
    gptq_hessian_update_impl, gptq_quantize_column_impl,
};
use crate::ops::traits::CalibrationOps;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl CalibrationOps<CudaRuntime> for CudaClient {
    fn awq_channel_scores(
        &self,
        activations: &Tensor<CudaRuntime>,
        weights: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let act_shape = activations.shape();
        let w_shape = weights.shape();

        if act_shape.len() != 2 || w_shape.len() != 2 {
            return Err(Error::InvalidArgument {
                arg: "activations/weights",
                reason: format!("expected 2D, got act {:?}, w {:?}", act_shape, w_shape),
            });
        }
        if act_shape[1] != w_shape[1] {
            return Err(Error::InvalidArgument {
                arg: "weights",
                reason: format!("K mismatch: act K={}, w K={}", act_shape[1], w_shape[1]),
            });
        }

        let n = act_shape[0];
        let k = act_shape[1];
        let m = w_shape[0];
        let dtype = activations.dtype();

        let kernel_prefix = match dtype {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            _ => {
                return Err(Error::KernelError {
                    reason: format!("AWQ: unsupported dtype {:?}", dtype),
                });
            }
        };

        let device = activations.device();
        let device_index = device.id();
        let module = kernels::get_or_load_module(self.context(), device_index, CALIBRATION_MODULE)?;

        // Step 1: act_scale[j] = max_n(|act[n,j]|) → [K]
        let act_scale = Tensor::<CudaRuntime>::zeros(&[k], dtype, device);
        {
            let func_name = format!("awq_act_scale_{}", kernel_prefix);
            let func = kernels::get_kernel_function(&module, &func_name)?;

            let total = (n * k) as u32;
            let block = 256u32;
            let grid = total.div_ceil(block);
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };

            let act_ptr = activations.ptr();
            let out_ptr = act_scale.ptr();
            let n_i32 = n as i32;
            let k_i32 = k as i32;

            unsafe {
                let mut builder = self.stream().launch_builder(&func);
                builder.arg(&act_ptr);
                builder.arg(&out_ptr);
                builder.arg(&n_i32);
                builder.arg(&k_i32);
                builder.launch(cfg).map_err(|e| Error::KernelError {
                    reason: format!("awq_act_scale launch failed: {:?}", e),
                })?;
            }
        }

        // Step 2: score[j] = mean_i(act_scale[j] * |W[i,j]|) → [K]
        let scores = Tensor::<CudaRuntime>::zeros(&[k], dtype, device);
        {
            let func_name = format!("awq_score_reduce_{}", kernel_prefix);
            let func = kernels::get_kernel_function(&module, &func_name)?;

            let total = (m * k) as u32;
            let block = 256u32;
            let grid = total.div_ceil(block);
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            };

            let w_ptr = weights.ptr();
            let scale_ptr = act_scale.ptr();
            let out_ptr = scores.ptr();
            let m_i32 = m as i32;
            let k_i32 = k as i32;

            unsafe {
                let mut builder = self.stream().launch_builder(&func);
                builder.arg(&w_ptr);
                builder.arg(&scale_ptr);
                builder.arg(&out_ptr);
                builder.arg(&m_i32);
                builder.arg(&k_i32);
                builder.launch(cfg).map_err(|e| Error::KernelError {
                    reason: format!("awq_score_reduce launch failed: {:?}", e),
                })?;
            }
        }

        Ok(scores)
    }

    fn fisher_information(&self, gradients: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let shape = gradients.shape();
        if shape.len() != 2 {
            return Err(Error::InvalidArgument {
                arg: "gradients",
                reason: format!("expected 2D [N, P], got {:?}", shape),
            });
        }

        let n = shape[0];
        let p = shape[1];
        let dtype = gradients.dtype();

        let kernel_prefix = match dtype {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            _ => {
                return Err(Error::KernelError {
                    reason: format!("Fisher: unsupported dtype {:?}", dtype),
                });
            }
        };

        let device = gradients.device();
        let device_index = device.id();
        let module = kernels::get_or_load_module(self.context(), device_index, CALIBRATION_MODULE)?;

        let output = Tensor::<CudaRuntime>::zeros(&[p], dtype, device);
        let func_name = format!("fisher_accumulate_{}", kernel_prefix);
        let func = kernels::get_kernel_function(&module, &func_name)?;

        let total = (n * p) as u32;
        let block = 256u32;
        let grid = total.div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        let grad_ptr = gradients.ptr();
        let out_ptr = output.ptr();
        let n_i32 = n as i32;
        let p_i32 = p as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&grad_ptr);
            builder.arg(&out_ptr);
            builder.arg(&n_i32);
            builder.arg(&p_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("fisher_accumulate launch failed: {:?}", e),
            })?;
        }

        Ok(output)
    }

    fn gptq_hessian_update(
        &self,
        hessian: &Tensor<CudaRuntime>,
        x_block: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // Delegate to impl_generic — numr matmul is already CUDA-optimized
        gptq_hessian_update_impl(self, hessian, x_block)
    }

    fn gptq_quantize_column(
        &self,
        weight: &Tensor<CudaRuntime>,
        h_inv: &Tensor<CudaRuntime>,
        num_bits: u32,
        group_size: u32,
        symmetric: bool,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        // Delegate to impl_generic — sequential column loop
        gptq_quantize_column_impl(self, weight, h_inv, num_bits, group_size, symmetric)
    }
}
