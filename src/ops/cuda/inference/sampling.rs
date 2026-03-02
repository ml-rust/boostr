//! CUDA implementation of SamplingOps — fused GPU kernels.
//!
//! Penalties: one thread per unique token, modifies logits in-place.
//! Sampling: single-block fused kernel for temperature → softmax → top-k → top-p → min-p → multinomial.
//! Random value generated on-device via numr's RandomOps::rand — no CPU RNG.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, SAMPLING_MODULE, SAMPLING_PENALTIES_MODULE};
use crate::ops::traits::inference::sampling::SamplingOps;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::ops::RandomOps;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl SamplingOps<CudaRuntime> for CudaClient {
    fn apply_sampling_penalties(
        &self,
        logits: &Tensor<CudaRuntime>,
        token_ids: &Tensor<CudaRuntime>,
        token_counts: &Tensor<CudaRuntime>,
        repeat_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
    ) -> Result<()> {
        let num_unique = token_ids.shape()[0];
        if num_unique == 0 {
            return Ok(());
        }

        let device = logits.device();
        let device_index = device.id();

        let module =
            kernels::get_or_load_module(self.context(), device_index, SAMPLING_PENALTIES_MODULE)?;
        let func = kernels::get_kernel_function(&module, "apply_sampling_penalties_kernel")?;

        let block_size = 256u32;
        let grid_size = (num_unique as u32).div_ceil(block_size);

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let logits_ptr = logits.ptr();
        let ids_ptr = token_ids.ptr();
        let counts_ptr = token_counts.ptr();
        let num_unique_i32 = num_unique as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&logits_ptr);
            builder.arg(&ids_ptr);
            builder.arg(&counts_ptr);
            builder.arg(&num_unique_i32);
            builder.arg(&repeat_penalty);
            builder.arg(&frequency_penalty);
            builder.arg(&presence_penalty);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("apply_sampling_penalties_kernel launch failed: {:?}", e),
            })?;
        }

        Ok(())
    }

    fn sample_token(
        &self,
        logits: &Tensor<CudaRuntime>,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
    ) -> Result<u32> {
        let vocab_size = logits.shape()[0];

        let device = logits.device();
        let device_index = device.id();

        let module = kernels::get_or_load_module(self.context(), device_index, SAMPLING_MODULE)?;
        let func = kernels::get_kernel_function(&module, "sample_token_kernel")?;

        // Allocate scratch buffer for probabilities (global mem, not shared)
        let probs_buf =
            Tensor::<CudaRuntime>::zeros(&[vocab_size], numr::dtype::DType::F32, device);
        // Generate random value on GPU via RandomOps
        let rand_tensor = self
            .rand(&[1], numr::dtype::DType::F32)
            .map_err(|e| Error::Numr(e))?;
        // Allocate output tensor for sampled token ID
        let output = Tensor::<CudaRuntime>::zeros(&[1], numr::dtype::DType::I32, device);

        let block_size = 1024u32;
        // shared mem: only block_size floats for reductions
        let shared_mem = (block_size as usize * 4) as u32;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_mem,
        };

        let logits_ptr = logits.ptr();
        let probs_ptr = probs_buf.ptr();
        let vocab_size_i32 = vocab_size as i32;
        let top_k_i32 = top_k as i32;
        let rand_ptr = rand_tensor.ptr();
        let output_ptr = output.ptr();

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&logits_ptr);
            builder.arg(&probs_ptr);
            builder.arg(&vocab_size_i32);
            builder.arg(&temperature);
            builder.arg(&top_k_i32);
            builder.arg(&top_p);
            builder.arg(&min_p);
            builder.arg(&rand_ptr);
            builder.arg(&output_ptr);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("sample_token_kernel launch failed: {:?}", e),
            })?;
        }

        // Read back the single i32 result
        let result: Vec<i32> = output.to_vec();
        Ok(result[0] as u32)
    }
}
