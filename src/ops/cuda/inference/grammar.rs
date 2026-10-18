//! CUDA implementation of GrammarDfaOps — GPU-side DFA logit masking.
//!
//! One thread per vocabulary token. The mask writes a device copy of the
//! logits, so the input stays unchanged and nothing returns to the host.

use crate::error::{Error, Result};
use crate::inference::grammar::device::check_logits;
use crate::ops::cuda::kernels::{self, GRAMMAR_DFA_MODULE};
use crate::ops::traits::inference::grammar::{DeviceGrammarDfa, GrammarDfaOps};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use cudarc::driver::sys;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// `value` as a kernel `int32_t`, or an error that names it.
fn kernel_i32(value: usize, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::InvalidArgument {
        arg: "grammar",
        reason: format!("grammar_dfa_mask_logits: {what} {value} exceeds the int32 range"),
    })
}

impl GrammarDfaOps<CudaRuntime> for CudaClient {
    fn grammar_dfa_mask_logits(
        &self,
        logits: &Tensor<CudaRuntime>,
        grammar: &DeviceGrammarDfa<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let vocab_size = grammar.vocab_size();
        check_logits(logits, vocab_size)?;
        let total = logits.numel();
        let output = Tensor::<CudaRuntime>::empty(logits.shape(), DType::F32, logits.device())?;
        if total == 0 {
            return Ok(output);
        }
        let f32_bytes = std::mem::size_of::<f32>();
        // The copy holds the enqueue permit for its own call only. The launch
        // below takes its own permit.
        {
            let _permit = self
                .stream()
                .enqueue_permit()
                .map_err(|e| Error::KernelError {
                    reason: format!("grammar_dfa_mask_logits: enqueue permit failed: {e:?}"),
                })?;
            // SAFETY: both pointers address `total` F32 values on this device.
            // `check_logits` checks the source is contiguous.
            let result = unsafe {
                sys::cuMemcpyDtoDAsync_v2(
                    output.ptr(),
                    logits.ptr(),
                    total * f32_bytes,
                    self.stream().raw().cu_stream(),
                )
            };
            if result != sys::CUresult::CUDA_SUCCESS {
                return Err(Error::KernelError {
                    reason: format!("grammar_dfa_mask_logits: logits copy failed: {result:?}"),
                });
            }
        }

        let module =
            kernels::get_or_load_module(self.context(), logits.device().id(), GRAMMAR_DFA_MODULE)?;
        let func = kernels::get_kernel_function(&module, "grammar_dfa_mask_logits_kernel")?;

        let block_size = 256u32;
        let grid_size = (vocab_size as u32).div_ceil(block_size);
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // The mask covers the last `vocab_size` logits.
        let logits_ptr = output.ptr() + ((total - vocab_size) * f32_bytes) as u64;
        let vocab = grammar.vocab();
        let transition_ptr = grammar.transition_table().ptr();
        let accepting_ptr = grammar.accepting_mask().ptr();
        let vocab_bytes_ptr = vocab.bytes().ptr();
        let vocab_offsets_ptr = vocab.offsets().ptr();
        let eos_ptr = grammar.eos_tokens().ptr();
        let current_state = i32::try_from(grammar.current_state()).unwrap_or(-1);
        let num_states = kernel_i32(grammar.num_states(), "state count")?;
        let vocab_size_i32 = kernel_i32(vocab_size, "vocab size")?;
        let bytes_len = kernel_i32(vocab.total_bytes(), "vocab byte count")?;
        let num_eos = kernel_i32(grammar.num_eos(), "EOS count")?;

        // SAFETY: every pointer is a live device tensor that `grammar` or
        // `output` owns, and each length argument bounds its buffer.
        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&logits_ptr);
            builder.arg(&transition_ptr);
            builder.arg(&accepting_ptr);
            builder.arg(&vocab_bytes_ptr);
            builder.arg(&vocab_offsets_ptr);
            builder.arg(&eos_ptr);
            builder.arg(&current_state);
            builder.arg(&num_states);
            builder.arg(&vocab_size_i32);
            builder.arg(&bytes_len);
            builder.arg(&num_eos);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("grammar_dfa_mask_logits_kernel launch failed: {:?}", e),
            })?;
        }

        Ok(output)
    }
}
