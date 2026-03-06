//! CUDA implementation of GrammarDfaOps — GPU-side DFA logit masking.
//!
//! One thread per vocabulary token. No CPU<->GPU roundtrip needed.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, GRAMMAR_DFA_MODULE};
use crate::ops::traits::inference::grammar::{DeviceGrammarDfa, GrammarDfaOps};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl GrammarDfaOps<CudaRuntime> for CudaClient {
    fn grammar_dfa_mask_logits(
        &self,
        logits: &Tensor<CudaRuntime>,
        grammar: &DeviceGrammarDfa<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let vocab_size = grammar.vocab_size;

        // Clone logits so we can modify in-place
        let output = logits.clone();

        // Get the pointer to the last vocab_size logits
        let total_elements = output.shape().iter().product::<usize>();
        let logits_offset = total_elements.saturating_sub(vocab_size);

        let device = logits.device();
        let device_index = device.id();

        let module = kernels::get_or_load_module(self.context(), device_index, GRAMMAR_DFA_MODULE)?;
        let func = kernels::get_kernel_function(&module, "grammar_dfa_mask_logits_kernel")?;

        let block_size = 256u32;
        let grid_size = (vocab_size as u32).div_ceil(block_size);

        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Offset the logits pointer to point at the last position's logits
        let logits_ptr = output.ptr();
        let logits_ptr_offset = unsafe { logits_ptr.offset(logits_offset as isize * 4) };
        let transition_ptr = grammar.transition_table.ptr();
        let accepting_ptr = grammar.accepting_mask.ptr();
        let vocab_bytes_ptr = grammar.vocab_bytes.ptr();
        let vocab_offsets_ptr = grammar.vocab_offsets.ptr();
        let current_state = grammar.current_state as i32;
        let num_states = grammar.num_states as i32;
        let vocab_size_i32 = vocab_size as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&logits_ptr_offset);
            builder.arg(&transition_ptr);
            builder.arg(&accepting_ptr);
            builder.arg(&vocab_bytes_ptr);
            builder.arg(&vocab_offsets_ptr);
            builder.arg(&current_state);
            builder.arg(&num_states);
            builder.arg(&vocab_size_i32);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("grammar_dfa_mask_logits_kernel launch failed: {:?}", e),
            })?;
        }

        Ok(output)
    }
}
