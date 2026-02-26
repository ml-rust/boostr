//! CUDA implementation of SpeculativeOps
//!
//! verify_speculative_tokens: delegates to impl_generic (uses numr philox_uniform,
//!   results are Vec<VerificationResult> — CPU-side — so serial loop on CPU is correct).
//! compute_acceptance_probs: fused element-wise CUDA kernel (no RNG, fully parallel).
//! compute_expected_tokens: one thread per batch element CUDA kernel.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, SPECULATIVE_VERIFY_MODULE};
use crate::ops::impl_generic::inference::speculative::verify_speculative_tokens_impl;
use crate::ops::traits::inference::speculative::{SpeculativeOps, VerificationResult};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl SpeculativeOps<CudaRuntime> for CudaClient {
    fn verify_speculative_tokens(
        &self,
        draft_probs: &Tensor<CudaRuntime>,
        target_probs: &Tensor<CudaRuntime>,
        draft_tokens: &Tensor<CudaRuntime>,
        seed: u64,
    ) -> Result<Vec<VerificationResult>> {
        // Delegate to impl_generic: generates uniform randoms via philox_uniform
        // (same algorithm as CPU/WebGPU), then runs the serial accept/reject loop on CPU.
        // Results are Vec<VerificationResult> (CPU-side) regardless of backend.
        verify_speculative_tokens_impl(self, draft_probs, target_probs, draft_tokens, seed)
    }

    fn compute_acceptance_probs(
        &self,
        draft_probs: &Tensor<CudaRuntime>,
        target_probs: &Tensor<CudaRuntime>,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let dp_shape = draft_probs.shape();
        let tp_shape = target_probs.shape();

        if dp_shape != tp_shape {
            return Err(Error::InvalidArgument {
                arg: "target_probs",
                reason: format!(
                    "shape mismatch: draft {:?} vs target {:?}",
                    dp_shape, tp_shape
                ),
            });
        }

        let total_elements: usize = dp_shape.iter().product();
        let device = draft_probs.device();
        let device_index = device.id();

        let acceptance = Tensor::<CudaRuntime>::empty(dp_shape, DType::F32, device);
        let residual = Tensor::<CudaRuntime>::empty(dp_shape, DType::F32, device);

        let module =
            kernels::get_or_load_module(self.context(), device_index, SPECULATIVE_VERIFY_MODULE)?;
        let func = kernels::get_kernel_function(&module, "compute_acceptance_probs_kernel")?;

        let block_size = 256u32;
        let grid_size = ((total_elements as u32) + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let dp_ptr = draft_probs.ptr();
        let tp_ptr = target_probs.ptr();
        let acc_ptr = acceptance.ptr();
        let res_ptr = residual.ptr();
        let total_i32 = total_elements as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&dp_ptr);
            builder.arg(&tp_ptr);
            builder.arg(&acc_ptr);
            builder.arg(&res_ptr);
            builder.arg(&total_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("acceptance probs kernel launch failed: {:?}", e),
            })?;
        }

        Ok((acceptance, residual))
    }

    fn compute_expected_tokens(
        &self,
        acceptance_rates: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let shape = acceptance_rates.shape();
        if shape.len() != 2 {
            return Err(Error::InvalidArgument {
                arg: "acceptance_rates",
                reason: format!("expected 2D [batch, K], got {}D", shape.len()),
            });
        }

        let batch_size = shape[0];
        let k = shape[1];
        let device = acceptance_rates.device();
        let device_index = device.id();

        let output = Tensor::<CudaRuntime>::empty(&[batch_size], DType::F32, device);

        let module =
            kernels::get_or_load_module(self.context(), device_index, SPECULATIVE_VERIFY_MODULE)?;
        let func = kernels::get_kernel_function(&module, "compute_expected_tokens_kernel")?;

        let block_size = 256u32;
        let grid_size = ((batch_size as u32) + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let rates_ptr = acceptance_rates.ptr();
        let out_ptr = output.ptr();
        let bs_i32 = batch_size as i32;
        let k_i32 = k as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&rates_ptr);
            builder.arg(&out_ptr);
            builder.arg(&bs_i32);
            builder.arg(&k_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("expected tokens kernel launch failed: {:?}", e),
            })?;
        }

        Ok(output)
    }
}
