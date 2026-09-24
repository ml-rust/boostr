//! CUDA implementation of SamplingOps — fused GPU kernels.
//!
//! Penalties: one thread per unique token, modifies logits in-place.
//! Sampling: single-block fused kernel for temperature → softmax → top-k → top-p → min-p → multinomial.
//! Random value generated on-device via numr's RandomOps::rand — no CPU RNG.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{
    self, LOGITS_TO_TOKEN_MODULE, SAMPLING_MODULE, SAMPLING_PENALTIES_MODULE,
};
use crate::ops::impl_generic::inference::sampling::{
    SamplingLogitsShape, validate_contiguous_arg, validate_sampling_logits, validate_scratch_vocab,
};
use crate::ops::traits::inference::sampling::SamplingOps;
use crate::ops::traits::inference::sampling_scratch::SamplingScratch;
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

        // Generate random value on GPU via RandomOps. numr has no
        // `rand_into`, so this one-element tensor still allocates every
        // call — the probs and output buffers below do not.
        let rand_tensor = self
            .rand(&[1], numr::dtype::DType::F32)
            .map_err(Error::Numr)?;

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
        let rand_ptr = rand_tensor.ptr();
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
        validate_sampling_logits(
            "logits_to_token",
            logits,
            SamplingLogitsShape::BatchSeqVocab,
        )?;
        validate_contiguous_arg("logits_to_token", "token_ids", token_ids)?;
        validate_contiguous_arg("logits_to_token", "token_counts", token_counts)?;
        let shape = logits.shape();
        let seq_len = shape[1];
        let vocab_size = shape[2];
        validate_scratch_vocab("logits_to_token", scratch, vocab_size)?;

        let device_index = logits.device().id();

        let module =
            kernels::get_or_load_module(self.context(), device_index, LOGITS_TO_TOKEN_MODULE)?;
        let func = kernels::get_kernel_function(&module, "logits_to_token_kernel")?;

        // Generate random value on-device (only used for non-greedy, but always generated
        // to avoid branching on host — the kernel ignores it when temperature == 0).
        // numr has no `rand_into`, so this one-element tensor still allocates
        // every call — the probs and output buffers below do not.
        let rand_tensor = if let Some(s) = seed {
            self.rand_seeded(&[1], numr::dtype::DType::F32, s)
                .map_err(Error::Numr)?
        } else {
            self.rand(&[1], numr::dtype::DType::F32)
                .map_err(Error::Numr)?
        };

        // Determine dtype code: 0 = F32, 1 = F16, 2 = BF16.
        let dtype_code: i32 = match logits.dtype() {
            numr::dtype::DType::F32 => 0,
            numr::dtype::DType::F16 => 1,
            numr::dtype::DType::BF16 => 2,
            other => {
                return Err(Error::InvalidArgument {
                    arg: "logits",
                    reason: format!(
                        "logits_to_token: unsupported dtype {other:?}, expected F32/F16/BF16"
                    ),
                });
            }
        };

        let block_size = 1024u32;
        // Shared mem: block_size floats for reductions + block_size ints for argmax indices
        let shared_mem = (block_size as usize * 4 * 2) as u32;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: shared_mem,
        };

        let logits_ptr = logits.ptr();
        let probs_ptr = scratch.probs.ptr();
        let ids_ptr = token_ids.ptr();
        let counts_ptr = token_counts.ptr();
        let output_ptr = scratch.output.ptr();
        let rand_ptr = rand_tensor.ptr();
        let seq_len_i32 = seq_len as i32;
        let vocab_size_i32 = vocab_size as i32;
        let num_unique_i32 = num_unique as i32;
        let top_k_i32 = top_k as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&logits_ptr);
            builder.arg(&probs_ptr);
            builder.arg(&ids_ptr);
            builder.arg(&counts_ptr);
            builder.arg(&output_ptr);
            builder.arg(&seq_len_i32);
            builder.arg(&vocab_size_i32);
            builder.arg(&num_unique_i32);
            builder.arg(&repeat_penalty);
            builder.arg(&frequency_penalty);
            builder.arg(&presence_penalty);
            builder.arg(&temperature);
            builder.arg(&top_k_i32);
            builder.arg(&top_p);
            builder.arg(&min_p);
            builder.arg(&rand_ptr);
            builder.arg(&dtype_code);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("logits_to_token_kernel launch failed: {:?}", e),
            })?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cuda_setup;

    /// A `[vocab_size]` view with stride 2, mirroring the CPU fixture: every
    /// other element of a `[vocab_size, 2]` buffer.
    fn non_contiguous_vocab(device: &numr::runtime::cuda::CudaDevice) -> Tensor<CudaRuntime> {
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let base = Tensor::<CudaRuntime>::from_slice(&data, &[4, 2], device).expect("base");
        let view = base.narrow(1, 0, 1).expect("narrow").squeeze(Some(1));
        assert_eq!(view.shape(), &[4]);
        assert!(!view.is_contiguous());
        view
    }

    /// A `[1, seq_len, vocab_size]` view whose vocab axis is narrowed out of
    /// a wider buffer, so it keeps the parent's stride instead of a
    /// contiguous one.
    fn non_contiguous_batch_seq_vocab(
        device: &numr::runtime::cuda::CudaDevice,
    ) -> Tensor<CudaRuntime> {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let base = Tensor::<CudaRuntime>::from_slice(&data, &[1, 2, 8], device).expect("base");
        let view = base.narrow(2, 0, 4).expect("narrow");
        assert_eq!(view.shape(), &[1, 2, 4]);
        assert!(!view.is_contiguous());
        view
    }

    /// A `[len]` i64 view with stride 2, mirroring the CPU fixture: every
    /// other element of a `[len, 2]` buffer of token ids.
    fn non_contiguous_ids(device: &numr::runtime::cuda::CudaDevice) -> Tensor<CudaRuntime> {
        let data: Vec<i64> = vec![0, 9, 1, 9];
        let base = Tensor::<CudaRuntime>::from_slice(&data, &[2, 2], device).expect("base");
        let view = base.narrow(1, 0, 1).expect("narrow").squeeze(Some(1));
        assert_eq!(view.shape(), &[2]);
        assert!(!view.is_contiguous());
        view
    }

    /// A `[len]` i32 view with stride 2, mirroring the CPU fixture: every
    /// other element of a `[len, 2]` buffer of token counts.
    fn non_contiguous_counts(device: &numr::runtime::cuda::CudaDevice) -> Tensor<CudaRuntime> {
        let data: Vec<i32> = vec![1, 9, 2, 9];
        let base = Tensor::<CudaRuntime>::from_slice(&data, &[2, 2], device).expect("base");
        let view = base.narrow(1, 0, 1).expect("narrow").squeeze(Some(1));
        assert_eq!(view.shape(), &[2]);
        assert!(!view.is_contiguous());
        view
    }

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
    fn logits_to_token_rejects_non_contiguous_token_ids() {
        let Some(cuda) = cuda_setup() else { return };
        let (client, device) = (&cuda.client, &cuda.device);
        let data: Vec<f32> = vec![0.0, 1.0, 0.0, 2.0];
        let logits = Tensor::<CudaRuntime>::from_slice(&data, &[1, 1, 4], device).expect("logits");
        let token_ids = non_contiguous_ids(device);
        let token_counts =
            Tensor::<CudaRuntime>::from_slice(&[1i32, 1], &[2], device).expect("cnt");

        let err = client
            .logits_to_token(
                &logits,
                &token_ids,
                &token_counts,
                1,
                1.0,
                0.0,
                0.0,
                0.0,
                0,
                1.0,
                0.0,
                None,
            )
            .expect_err("non-contiguous token_ids must be rejected");
        assert!(format!("{err}").contains("token_ids"));
    }

    #[test]
    fn logits_to_token_rejects_non_contiguous_token_counts() {
        let Some(cuda) = cuda_setup() else { return };
        let (client, device) = (&cuda.client, &cuda.device);
        let data: Vec<f32> = vec![0.0, 1.0, 0.0, 2.0];
        let logits = Tensor::<CudaRuntime>::from_slice(&data, &[1, 1, 4], device).expect("logits");
        let token_ids = Tensor::<CudaRuntime>::from_slice(&[0i64, 1], &[2], device).expect("ids");
        let token_counts = non_contiguous_counts(device);

        let err = client
            .logits_to_token(
                &logits,
                &token_ids,
                &token_counts,
                1,
                1.0,
                0.0,
                0.0,
                0.0,
                0,
                1.0,
                0.0,
                None,
            )
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

    #[test]
    fn logits_to_token_rejects_non_contiguous_logits() {
        let Some(cuda) = cuda_setup() else { return };
        let (client, device) = (&cuda.client, &cuda.device);
        let logits = non_contiguous_batch_seq_vocab(device);
        let token_ids = Tensor::<CudaRuntime>::from_slice(&[0i64], &[1], device).expect("ids");
        let token_counts = Tensor::<CudaRuntime>::from_slice(&[1i32], &[1], device).expect("cnt");

        let err = client
            .logits_to_token(
                &logits,
                &token_ids,
                &token_counts,
                0,
                1.0,
                0.0,
                0.0,
                0.0,
                0,
                1.0,
                0.0,
                None,
            )
            .expect_err("non-contiguous logits must be rejected");
        assert!(format!("{err}").contains("logits_to_token"));
    }

    /// A scratch reused across calls (same seed, same inputs) produces the
    /// same token as a fresh allocation per call — nothing leaks between
    /// calls through the shared probs/output buffers.
    #[test]
    fn a_reused_scratch_matches_fresh_allocation_per_call() {
        let Some(cuda) = cuda_setup() else { return };
        let (client, device) = (&cuda.client, &cuda.device);
        let data: Vec<f32> = vec![0.0, 1.0, 0.0, 2.0, 9.0, -1.0, 0.5, 3.0];
        let logits = Tensor::<CudaRuntime>::from_slice(&data, &[1, 2, 4], device).expect("logits");
        let token_ids = Tensor::<CudaRuntime>::from_slice(&[0i64], &[1], device).expect("ids");
        let token_counts = Tensor::<CudaRuntime>::from_slice(&[1i32], &[1], device).expect("cnt");
        let scratch = SamplingScratch::<CudaRuntime>::new(4, device).expect("scratch");

        for seed in [1u64, 2u64, 3u64] {
            let fresh = client
                .logits_to_token(
                    &logits,
                    &token_ids,
                    &token_counts,
                    1,
                    1.1,
                    0.1,
                    0.1,
                    0.7,
                    0,
                    1.0,
                    0.0,
                    Some(seed),
                )
                .expect("fresh alloc path");
            client
                .logits_to_token_into(
                    &logits,
                    &token_ids,
                    &token_counts,
                    1,
                    1.1,
                    0.1,
                    0.1,
                    0.7,
                    0,
                    1.0,
                    0.0,
                    Some(seed),
                    &scratch,
                )
                .expect("scratch path");
            let fresh_id: Vec<i64> = fresh.to_vec();
            let reused_id: Vec<i64> = scratch.output.to_vec();
            assert_eq!(reused_id, fresh_id);
        }
    }
}
