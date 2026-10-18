//! WebGPU implementation of GrammarDfaOps — WGSL DFA logit mask, F32 only.
//!
//! One invocation per vocabulary token. The mask stays on the device.

use crate::error::{Error, Result};
use crate::inference::grammar::device::check_logits;
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
    current_state: u32,
    num_states: u32,
    num_eos: u32,
    vocab_bytes_len: u32,
    _pad0: u32,
    _pad1: u32,
}

/// `value` as a shader `u32`, or an error that names it.
fn shader_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::InvalidArgument {
        arg: "grammar",
        reason: format!("grammar_dfa_mask_logits: {what} {value} exceeds the u32 range"),
    })
}

impl GrammarDfaOps<WgpuRuntime> for WgpuClient {
    fn grammar_dfa_mask_logits(
        &self,
        logits: &Tensor<WgpuRuntime>,
        grammar: &DeviceGrammarDfa<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        let vocab_size = grammar.vocab_size();
        check_logits(logits, vocab_size)?;
        let total = logits.numel();
        // The mask writes a fresh copy. The caller's logits stay unchanged.
        let output = Tensor::<WgpuRuntime>::empty(logits.shape(), DType::F32, logits.device())?;
        if total == 0 {
            return Ok(output);
        }
        let vocab = grammar.vocab();
        let source_buf = tensor_buffer(logits, "logits")?;
        let table_buf = tensor_buffer(grammar.transition_table(), "transition_table")?;
        let accepting_buf = tensor_buffer(grammar.accepting_mask(), "accepting_mask")?;
        let bytes_buf = tensor_buffer(vocab.bytes(), "vocab_bytes")?;
        let offsets_buf = tensor_buffer(vocab.offsets(), "vocab_offsets")?;
        let eos_buf = tensor_buffer(grammar.eos_tokens(), "eos_tokens")?;
        let logits_buf = tensor_buffer(&output, "logits")?;

        let params = GrammarParams {
            logits_offset: shader_u32(total - vocab_size, "logits offset")?,
            vocab_size: shader_u32(vocab_size, "vocab size")?,
            current_state: grammar.current_state(),
            num_states: shader_u32(grammar.num_states(), "state count")?,
            num_eos: shader_u32(grammar.num_eos(), "EOS count")?,
            vocab_bytes_len: shader_u32(vocab.total_bytes(), "vocab byte count")?,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buf = uniform_buffer(self, "grammar_dfa_params", &params);

        let cache = self.pipeline_cache();
        let module = cache.get_or_create_module("grammar_dfa", GRAMMAR_SHADER);
        let layout = cache.get_or_create_layout(LayoutKey {
            num_storage_buffers: 6,
            num_uniform_buffers: 1,
            num_readonly_storage: 5,
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
                &eos_buf,
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
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grammar_dfa_mask_logits"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(params.vocab_size.div_ceil(256), 1, 1);
        }
        self.wgpu_queue().submit(std::iter::once(encoder.finish()));
        Ok(output)
    }
}
