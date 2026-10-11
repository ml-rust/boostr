//! WebGPU implementation of GrammarDfaOps — WGSL DFA logit mask, F32 only.
//!
//! One invocation per vocabulary token. The mask stays on the device.

use crate::error::{Error, Result};
use crate::ops::traits::inference::grammar::{DeviceGrammarDfa, GrammarDfaOps};
use numr::dtype::DType;
use numr::runtime::wgpu::shaders::LayoutKey;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

use super::sampling::{tensor_buffer, uniform_buffer};

const GRAMMAR_SHADER: &str = include_str!("../shaders/inference/grammar_dfa.wgsl");

/// Matches `GrammarParams` in `grammar_dfa.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GrammarParams {
    logits_offset: u32,
    vocab_size: u32,
    current_state: i32,
    num_states: u32,
}

impl GrammarDfaOps<WgpuRuntime> for WgpuClient {
    fn grammar_dfa_mask_logits(
        &self,
        logits: &Tensor<WgpuRuntime>,
        grammar: &DeviceGrammarDfa<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        if logits.dtype() != DType::F32 {
            return Err(Error::InvalidArgument {
                arg: "logits",
                reason: format!(
                    "grammar_dfa_mask_logits: WebGPU needs F32 logits, got {:?}",
                    logits.dtype()
                ),
            });
        }
        let vocab_size = grammar.vocab_size;
        let total = logits.numel();
        if total < vocab_size {
            return Err(Error::InvalidArgument {
                arg: "logits",
                reason: format!(
                    "grammar_dfa_mask_logits: logits hold {total} values, fewer than the \
                     grammar vocab of {vocab_size}"
                ),
            });
        }
        if !logits.is_contiguous() {
            return Err(Error::InvalidArgument {
                arg: "logits",
                reason: format!(
                    "grammar_dfa_mask_logits: logits are not contiguous (shape {:?}, strides \
                     {:?}). Pass a contiguous tensor",
                    logits.shape(),
                    logits.strides()
                ),
            });
        }
        // The mask writes a fresh copy. The caller's logits stay unchanged.
        let output = Tensor::<WgpuRuntime>::empty(logits.shape(), DType::F32, logits.device())?;
        if total == 0 {
            return Ok(output);
        }
        let source_buf = tensor_buffer(logits, "logits")?;
        let table_buf = tensor_buffer(&grammar.transition_table, "transition_table")?;
        let accepting_buf = tensor_buffer(&grammar.accepting_mask, "accepting_mask")?;
        let bytes_buf = tensor_buffer(&grammar.vocab_bytes, "vocab_bytes")?;
        let offsets_buf = tensor_buffer(&grammar.vocab_offsets, "vocab_offsets")?;
        let logits_buf = tensor_buffer(&output, "logits")?;

        let params = GrammarParams {
            logits_offset: (total - vocab_size) as u32,
            vocab_size: vocab_size as u32,
            current_state: grammar.current_state as i32,
            num_states: grammar.num_states as u32,
        };
        let params_buf = uniform_buffer(self, "grammar_dfa_params", &params);

        let cache = self.pipeline_cache();
        let module = cache.get_or_create_module("grammar_dfa", GRAMMAR_SHADER);
        let layout = cache.get_or_create_layout(LayoutKey {
            num_storage_buffers: 5,
            num_uniform_buffers: 1,
            num_readonly_storage: 4,
        });
        let pipeline = cache.get_or_create_pipeline(
            "grammar_dfa",
            "grammar_dfa_mask_logits",
            &module,
            &layout,
        );
        let bind_group = cache.create_bind_group(
            &layout,
            &[
                &table_buf,
                &accepting_buf,
                &bytes_buf,
                &offsets_buf,
                &logits_buf,
                &params_buf,
            ],
        );
        let f32_bytes = std::mem::size_of::<f32>() as u64;
        let mut encoder =
            self.wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("grammar_dfa_mask_logits"),
                });
        encoder.copy_buffer_to_buffer(
            &source_buf,
            logits.offset() as u64 * f32_bytes,
            &logits_buf,
            0,
            total as u64 * f32_bytes,
        );
        if vocab_size > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grammar_dfa_mask_logits"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups((vocab_size as u32).div_ceil(256), 1, 1);
        }
        self.wgpu_queue().submit(std::iter::once(encoder.finish()));
        Ok(output)
    }
}
