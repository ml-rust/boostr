//! `SamplingOps` for `CudaClient`: the penalties and sample-token kernel
//! launches, and the allocating wrappers over the `_into` variants.

use super::logits_to_token;
use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, SAMPLING_MODULE, SAMPLING_PENALTIES_MODULE};
use crate::ops::impl_generic::inference::sampling::{
    SamplingLogitsShape, validate_contiguous_arg, validate_sampling_logits, validate_scratch_vocab,
};
use crate::ops::traits::inference::sampling::SamplingOps;
use crate::ops::traits::inference::sampling_scratch::SamplingScratch;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
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
        validate_sampling_logits(
            "apply_sampling_penalties",
            logits,
            SamplingLogitsShape::Vocab,
        )?;
        validate_contiguous_arg("apply_sampling_penalties", "token_ids", token_ids)?;
        validate_contiguous_arg("apply_sampling_penalties", "token_counts", token_counts)?;
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
        validate_sampling_logits("sample_token", logits, SamplingLogitsShape::Vocab)?;
        let vocab_size = logits.shape()[0];
        let scratch = SamplingScratch::new(vocab_size, logits.device())?;
        self.sample_token_into(logits, temperature, top_k, top_p, min_p, &scratch)?;
        let result: Vec<i64> = scratch.output.to_vec();
        Ok(result[0] as u32)
    }

    fn sample_token_into(
        &self,
        logits: &Tensor<CudaRuntime>,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
        scratch: &SamplingScratch<CudaRuntime>,
    ) -> Result<()> {
        validate_sampling_logits("sample_token", logits, SamplingLogitsShape::Vocab)?;
        let vocab_size = logits.shape()[0];
        validate_scratch_vocab("sample_token", scratch, vocab_size)?;

        let device = logits.device();
        let device_index = device.id();

        let module = kernels::get_or_load_module(self.context(), device_index, SAMPLING_MODULE)?;
        let func = kernels::get_kernel_function(&module, "sample_token_kernel")?;

        // Draw into scratch.random instead of allocating a fresh tensor.
        scratch.draw_random(self, None)?;

        let block_size = 1024u32;
        // shared mem: only block_size floats for reductions
        let shared_mem = (block_size as usize * 4) as u32;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_mem,
        };

        let logits_ptr = logits.ptr();
        let probs_ptr = scratch.probs.ptr();
        let vocab_size_i32 = vocab_size as i32;
        let top_k_i32 = top_k as i32;
        let rand_ptr = scratch.random.ptr();
        let output_ptr = scratch.output.ptr();

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

        Ok(())
    }

    fn logits_to_token(
        &self,
        logits: &Tensor<CudaRuntime>,
        token_ids: &Tensor<CudaRuntime>,
        token_counts: &Tensor<CudaRuntime>,
        num_unique: usize,
        repeat_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
        seed: Option<u64>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_sampling_logits(
            "logits_to_token",
            logits,
            SamplingLogitsShape::BatchSeqVocab,
        )?;
        let vocab_size = logits.shape()[2];
        let scratch = SamplingScratch::new(vocab_size, logits.device())?;
        self.logits_to_token_into(
            logits,
            token_ids,
            token_counts,
            num_unique,
            repeat_penalty,
            frequency_penalty,
            presence_penalty,
            temperature,
            top_k,
            top_p,
            min_p,
            seed,
            &scratch,
        )?;
        Ok(scratch.output.clone())
    }

    fn logits_to_token_into(
        &self,
        logits: &Tensor<CudaRuntime>,
        token_ids: &Tensor<CudaRuntime>,
        token_counts: &Tensor<CudaRuntime>,
        num_unique: usize,
        repeat_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
        seed: Option<u64>,
        scratch: &SamplingScratch<CudaRuntime>,
    ) -> Result<()> {
        logits_to_token::logits_to_token_into(
            self,
            logits,
            token_ids,
            token_counts,
            num_unique,
            repeat_penalty,
            frequency_penalty,
            presence_penalty,
            temperature,
            top_k,
            top_p,
            min_p,
            seed,
            scratch,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{
        non_contiguous_counts, non_contiguous_ids, non_contiguous_vocab,
    };
    use super::*;
    use crate::test_utils::cuda_setup;

    #[test]
    fn apply_sampling_penalties_rejects_non_contiguous_token_ids() {
        let Some(cuda) = cuda_setup() else { return };
        let (client, device) = (&cuda.client, &cuda.device);
        let logits =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, -2.0], &[2], device).expect("logits");
        let token_ids = non_contiguous_ids(device);
        let token_counts =
            Tensor::<CudaRuntime>::from_slice(&[1i32, 1], &[2], device).expect("cnt");

        let err = client
            .apply_sampling_penalties(&logits, &token_ids, &token_counts, 1.1, 0.0, 0.0)
            .expect_err("non-contiguous token_ids must be rejected");
        assert!(format!("{err}").contains("token_ids"));
    }

    #[test]
    fn apply_sampling_penalties_rejects_non_contiguous_token_counts() {
        let Some(cuda) = cuda_setup() else { return };
        let (client, device) = (&cuda.client, &cuda.device);
        let logits =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, -2.0], &[2], device).expect("logits");
        let token_ids = Tensor::<CudaRuntime>::from_slice(&[0i64, 1], &[2], device).expect("ids");
        let token_counts = non_contiguous_counts(device);

        let err = client
            .apply_sampling_penalties(&logits, &token_ids, &token_counts, 1.1, 0.0, 0.0)
            .expect_err("non-contiguous token_counts must be rejected");
        assert!(format!("{err}").contains("token_counts"));
    }

    #[test]
    fn apply_sampling_penalties_rejects_non_contiguous_logits() {
        let Some(cuda) = cuda_setup() else { return };
        let (client, device) = (&cuda.client, &cuda.device);
        let logits = non_contiguous_vocab(device);
        let token_ids = Tensor::<CudaRuntime>::from_slice(&[0i64], &[1], device).expect("ids");
        let token_counts = Tensor::<CudaRuntime>::from_slice(&[1i32], &[1], device).expect("cnt");

        let err = client
            .apply_sampling_penalties(&logits, &token_ids, &token_counts, 1.1, 0.0, 0.0)
            .expect_err("non-contiguous logits must be rejected");
        assert!(format!("{err}").contains("apply_sampling_penalties"));
    }

    #[test]
    fn sample_token_rejects_non_contiguous_logits() {
        let Some(cuda) = cuda_setup() else { return };
        let (client, device) = (&cuda.client, &cuda.device);
        let logits = non_contiguous_vocab(device);

        let err = client
            .sample_token(&logits, 1.0, 0, 1.0, 0.0)
            .expect_err("non-contiguous logits must be rejected");
        assert!(format!("{err}").contains("sample_token"));
    }
}
