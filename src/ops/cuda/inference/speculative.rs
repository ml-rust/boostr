//! CUDA implementation of `SpeculativeOps`: one fused block per batch row.

use super::sampling_probs::block_size;
use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, SPECULATIVE_VERIFY_MODULE};
use crate::ops::impl_generic::inference::speculative::{VerifyShape, validate_verify_args};
use crate::ops::traits::inference::speculative::SpeculativeOps;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

const KERNEL: &str = "speculative_verify_f32";

impl SpeculativeOps<CudaRuntime> for CudaClient {
    fn speculative_verify(
        &self,
        target_probs: &Tensor<CudaRuntime>,
        draft_probs: Option<&Tensor<CudaRuntime>>,
        draft_tokens: &Tensor<CudaRuntime>,
        uniforms: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let VerifyShape {
            batch,
            drafts,
            vocab,
        } = validate_verify_args(target_probs, draft_probs, draft_tokens, uniforms)?;
        let device = target_probs.device();
        let out = Tensor::<CudaRuntime>::empty(&[batch, 2], DType::I32, device)?;
        if batch == 0 {
            return Ok(out);
        }
        let grid = u32::try_from(batch).map_err(|_| Error::InvalidArgument {
            arg: "target_probs",
            reason: format!(
                "speculative_verify: {batch} rows exceed the grid limit of {} — split the batch",
                u32::MAX
            ),
        })?;
        // `validate_verify_args` bounds both: K+1 fits in i32, V is at most 2^24.
        let drafts_i32 = drafts as i32;
        let vocab_i32 = vocab as i32;

        let target = target_probs.contiguous()?;
        let draft = draft_probs.map(Tensor::contiguous).transpose()?;
        let tokens = draft_tokens.contiguous()?;
        let uniforms = uniforms.contiguous()?;

        let module =
            kernels::get_or_load_module(self.context(), device.id(), SPECULATIVE_VERIFY_MODULE)?;
        let func = kernels::get_kernel_function(&module, KERNEL)?;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block_size(vocab), 1, 1),
            shared_mem_bytes: 0,
        };

        let target_ptr = target.ptr();
        let draft_ptr = draft.as_ref().map_or(0u64, Tensor::ptr);
        let has_draft = i32::from(draft.is_some());
        let tokens_ptr = tokens.ptr();
        let uniforms_ptr = uniforms.ptr();
        let out_ptr = out.ptr();
        // SAFETY: every pointer is a live contiguous device buffer of the
        // checked shape, or null for an absent draft with `has_draft` 0.
        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&target_ptr);
            builder.arg(&draft_ptr);
            builder.arg(&tokens_ptr);
            builder.arg(&uniforms_ptr);
            builder.arg(&out_ptr);
            builder.arg(&drafts_i32);
            builder.arg(&vocab_i32);
            builder.arg(&has_draft);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("{KERNEL} launch failed: {e:?}"),
            })?;
        }
        Ok(out)
    }
}
