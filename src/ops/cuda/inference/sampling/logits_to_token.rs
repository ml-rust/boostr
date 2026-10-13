//! Fused logits-to-token kernel launch: penalties, temperature, top-k,
//! top-p, min-p and the multinomial draw (or argmax) in one block.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, LOGITS_TO_TOKEN_MODULE};
use crate::ops::impl_generic::inference::sampling::{
    SamplingLogitsShape, validate_contiguous_arg, validate_sampling_logits, validate_scratch_vocab,
};
use crate::ops::traits::inference::sampling_scratch::SamplingScratch;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// `SamplingOps::logits_to_token_into` for `CudaClient`: writes the token
/// into `scratch.output`.
#[allow(clippy::too_many_arguments)]
pub(super) fn logits_to_token_into(
    client: &CudaClient,
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
        kernels::get_or_load_module(client.context(), device_index, LOGITS_TO_TOKEN_MODULE)?;
    let func = kernels::get_kernel_function(&module, "logits_to_token_kernel")?;

    // Draw into scratch.random instead of allocating a fresh tensor
    // (only used for non-greedy, but always drawn to avoid branching on
    // host — the kernel ignores it when temperature == 0).
    scratch.draw_random(client, seed)?;

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
    let rand_ptr = scratch.random.ptr();
    let seq_len_i32 = seq_len as i32;
    let vocab_size_i32 = vocab_size as i32;
    let num_unique_i32 = num_unique as i32;
    let top_k_i32 = top_k as i32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
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

#[cfg(test)]
mod tests {
    use super::super::test_support::{
        non_contiguous_batch_seq_vocab, non_contiguous_counts, non_contiguous_ids,
    };
    use crate::ops::traits::inference::sampling::SamplingOps;
    use crate::ops::traits::inference::sampling_scratch::SamplingScratch;
    use crate::test_utils::cuda_setup;
    use numr::runtime::cuda::CudaRuntime;
    use numr::tensor::Tensor;

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

    /// Two successive calls with the same seed, through one reused scratch,
    /// draw the same random value and produce the same token both times —
    /// the first draw does not leak into or bias the second.
    #[test]
    fn same_seed_through_one_reused_scratch_repeats_the_token() {
        let Some(cuda) = cuda_setup() else { return };
        let (client, device) = (&cuda.client, &cuda.device);
        let data: Vec<f32> = vec![0.0, 1.0, 0.0, 2.0, 9.0, -1.0, 0.5, 3.0];
        let logits = Tensor::<CudaRuntime>::from_slice(&data, &[1, 2, 4], device).expect("logits");
        let token_ids = Tensor::<CudaRuntime>::from_slice(&[0i64], &[1], device).expect("ids");
        let token_counts = Tensor::<CudaRuntime>::from_slice(&[1i32], &[1], device).expect("cnt");
        let scratch = SamplingScratch::<CudaRuntime>::new(4, device).expect("scratch");

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
                Some(7),
                &scratch,
            )
            .expect("first scratch call");
        let first_id: Vec<i64> = scratch.output.to_vec();

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
                Some(7),
                &scratch,
            )
            .expect("second scratch call");
        let second_id: Vec<i64> = scratch.output.to_vec();

        assert_eq!(first_id, second_id);
    }
}
