//! CUDA implementation of `SamplingProbsOps`: one fused block per logits row.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, SAMPLING_PROBS_MODULE};
use crate::ops::traits::inference::sampling_probs::{FilterRule, SamplingProbsOps};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// The `top_k` argument of the filter kernels: 0 when top-k is disabled.
///
/// The cast is exact, since an active `top_k` stays below `vocab`.
pub(crate) fn kernel_top_k(rule: &FilterRule, vocab: usize) -> i32 {
    if rule.top_k_active(vocab) {
        rule.top_k as i32
    } else {
        0
    }
}

/// The vocab size as the `int` argument of a sampling kernel.
///
/// # Errors
///
/// Returns `Error::InvalidArgument` if `vocab` does not fit in an `i32`.
pub(crate) fn kernel_vocab(op: &'static str, vocab: usize) -> Result<i32> {
    i32::try_from(vocab).map_err(|_| Error::InvalidArgument {
        arg: "logits",
        reason: format!(
            "{op}: vocab {vocab} exceeds the kernel limit of {}",
            i32::MAX
        ),
    })
}

/// The `dtype` argument of the fused sampling kernels: 0 for F32, 1 for F16, 2 for BF16.
///
/// # Errors
///
/// Returns `Error::InvalidArgument` that names `op` for any other dtype.
pub(crate) fn kernel_dtype_code(op: &'static str, dtype: DType) -> Result<i32> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        other => Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!(
                "{op}: unsupported dtype {other:?}, expected F32/F16/BF16 — cast the logits first"
            ),
        }),
    }
}

/// The power-of-two block size for a row of `vocab` tokens, from 32 to 1024.
pub(crate) fn block_size(vocab: usize) -> u32 {
    vocab.next_power_of_two().clamp(32, 1024) as u32
}

impl SamplingProbsOps<CudaRuntime> for CudaClient {
    fn sampling_probs(
        &self,
        logits: &Tensor<CudaRuntime>,
        rule: &FilterRule,
    ) -> Result<Tensor<CudaRuntime>> {
        rule.check()?;
        let shape = logits.shape().to_vec();
        let vocab = shape.last().copied().unwrap_or(0);
        if vocab == 0 {
            return Err(Error::InvalidArgument {
                arg: "logits",
                reason: format!(
                    "sampling_probs: logits must be [.., vocab] with vocab above 0, got shape {shape:?}"
                ),
            });
        }
        let dtype = logits.dtype();
        if !dtype.is_float() {
            return Err(Error::InvalidArgument {
                arg: "logits",
                reason: format!(
                    "sampling_probs: logits must be a float dtype, got {dtype:?} — cast the logits first"
                ),
            });
        }
        // The kernel reads F32, F16 and BF16. Any other float dtype casts to F32 on the device.
        let input = match dtype {
            DType::F32 | DType::F16 | DType::BF16 => logits.contiguous()?,
            _ => self.cast(logits, DType::F32)?,
        };
        let kernel_name = match input.dtype() {
            DType::F16 => "sampling_probs_f16",
            DType::BF16 => "sampling_probs_bf16",
            _ => "sampling_probs_f32",
        };
        let vocab_i32 = kernel_vocab("sampling_probs", vocab)?;
        let rows = shape[..shape.len() - 1].iter().product::<usize>();
        let rows_u32 = u32::try_from(rows).map_err(|_| Error::InvalidArgument {
            arg: "logits",
            reason: format!(
                "sampling_probs: {rows} rows exceed the grid limit of {} — split the batch",
                u32::MAX
            ),
        })?;

        let device = logits.device();
        let out = Tensor::<CudaRuntime>::empty(&shape, DType::F32, device)?;
        if rows == 0 {
            return Ok(out);
        }

        let module =
            kernels::get_or_load_module(self.context(), device.id(), SAMPLING_PROBS_MODULE)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;
        let block = block_size(vocab);
        let cfg = LaunchConfig {
            grid_dim: (rows_u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * 4,
        };

        let in_ptr = input.ptr();
        let out_ptr = out.ptr();
        let top_k = kernel_top_k(rule, vocab);
        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&in_ptr);
            builder.arg(&out_ptr);
            builder.arg(&vocab_i32);
            builder.arg(&rule.temperature);
            builder.arg(&top_k);
            builder.arg(&rule.top_p);
            builder.arg(&rule.min_p);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("{kernel_name} launch failed: {e:?}"),
            })?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size_is_a_power_of_two_in_range() {
        assert_eq!(block_size(1), 32);
        assert_eq!(block_size(33), 64);
        assert_eq!(block_size(1000), 1024);
        assert_eq!(block_size(128_256), 1024);
    }

    #[test]
    fn kernel_top_k_is_zero_when_disabled() {
        let rule = FilterRule {
            temperature: 1.0,
            top_k: 4,
            top_p: 1.0,
            min_p: 0.0,
        };
        assert_eq!(kernel_top_k(&rule, 5), 4);
        assert_eq!(kernel_top_k(&rule, 4), 0);
        assert_eq!(kernel_top_k(&FilterRule { top_k: 0, ..rule }, 5), 0);
        assert_eq!(
            kernel_top_k(
                &FilterRule {
                    top_k: usize::MAX,
                    ..rule
                },
                5
            ),
            0
        );
    }
}
