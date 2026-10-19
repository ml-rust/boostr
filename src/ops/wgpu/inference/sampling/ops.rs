//! `SamplingOps` for `WgpuClient`: argument checks and the penalty pass.

use super::dispatch::{
    SamplingArgs, dispatch_logits_to_token, submit_pass, tensor_buffer, uniform_buffer,
};
use crate::error::{Error, Result};
use crate::ops::impl_generic::inference::sampling::{
    SamplingLogitsShape, validate_contiguous_arg, validate_sampling_logits, validate_scratch_vocab,
};
use crate::ops::traits::inference::sample_seed::SampleSeed;
use crate::ops::traits::inference::sampling::SamplingOps;
use crate::ops::traits::inference::sampling_probs::FilterRule;
use crate::ops::traits::inference::sampling_scratch::SamplingScratch;
use numr::dtype::DType;
use numr::runtime::wgpu::shaders::LayoutKey;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

const PENALTIES_SHADER: &str = include_str!("../../shaders/inference/sampling_penalties.wgsl");

/// Matches `PenaltyParams` in `sampling_penalties.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PenaltyParams {
    logits_offset: u32,
    vocab_size: u32,
    num_unique: u32,
    _pad0: u32,
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    _pad1: f32,
}

/// The canonical filter config of a stochastic sampling call.
fn filter_rule(temperature: f32, top_k: usize, top_p: f32, min_p: f32) -> FilterRule {
    FilterRule {
        temperature,
        top_k,
        top_p,
        min_p,
    }
}

/// Reject any logits dtype but F32.
fn require_f32(op: &str, logits: &Tensor<WgpuRuntime>) -> Result<()> {
    if logits.dtype() != DType::F32 {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!(
                "{op}: WebGPU sampling needs F32 logits, got {:?}. Cast the logits to F32 first",
                logits.dtype()
            ),
        });
    }
    Ok(())
}

/// The penalty count the kernel reads: never past either input tensor.
fn penalty_count(
    num_unique: usize,
    token_ids: &Tensor<WgpuRuntime>,
    token_counts: &Tensor<WgpuRuntime>,
) -> usize {
    num_unique.min(token_ids.numel()).min(token_counts.numel())
}

/// Reject penalty tensors whose dtype the kernel does not read.
fn check_penalty_dtypes(
    op: &'static str,
    token_ids: &Tensor<WgpuRuntime>,
    token_counts: &Tensor<WgpuRuntime>,
) -> Result<()> {
    if token_ids.numel() > 0 && token_ids.dtype() != DType::I64 {
        return Err(Error::InvalidArgument {
            arg: "token_ids",
            reason: format!("{op}: token_ids must be I64, got {:?}", token_ids.dtype()),
        });
    }
    if token_counts.numel() > 0 && token_counts.dtype() != DType::I32 {
        return Err(Error::InvalidArgument {
            arg: "token_counts",
            reason: format!(
                "{op}: token_counts must be I32, got {:?}",
                token_counts.dtype()
            ),
        });
    }
    Ok(())
}

impl SamplingOps<WgpuRuntime> for WgpuClient {
    fn apply_sampling_penalties(
        &self,
        logits: &Tensor<WgpuRuntime>,
        token_ids: &Tensor<WgpuRuntime>,
        token_counts: &Tensor<WgpuRuntime>,
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
        require_f32("apply_sampling_penalties", logits)?;
        check_penalty_dtypes("apply_sampling_penalties", token_ids, token_counts)?;
        let num_unique = penalty_count(token_ids.shape()[0], token_ids, token_counts);
        if num_unique == 0 {
            return Ok(());
        }

        let ids_buf = tensor_buffer(token_ids, "token_ids")?;
        let counts_buf = tensor_buffer(token_counts, "token_counts")?;
        let logits_buf = tensor_buffer(logits, "logits")?;
        let params = PenaltyParams {
            logits_offset: logits.offset() as u32,
            vocab_size: logits.shape()[0] as u32,
            num_unique: num_unique as u32,
            _pad0: 0,
            repeat_penalty,
            frequency_penalty,
            presence_penalty,
            _pad1: 0.0,
        };
        let params_buf = uniform_buffer(self, "sampling_penalty_params", &params);

        let cache = self.pipeline_cache();
        let module = cache.get_or_create_module("sampling_penalties", PENALTIES_SHADER);
        let layout = cache.get_or_create_layout(LayoutKey {
            num_storage_buffers: 3,
            num_uniform_buffers: 1,
            num_readonly_storage: 2,
        });
        let pipeline = cache.get_or_create_pipeline(
            "sampling_penalties",
            "apply_sampling_penalties",
            &module,
            &layout,
        );
        let bind_group =
            cache.create_bind_group(&layout, &[&ids_buf, &counts_buf, &logits_buf, &params_buf]);
        submit_pass(
            self,
            "apply_sampling_penalties",
            &pipeline,
            &bind_group,
            (num_unique as u32).div_ceil(256),
        );
        Ok(())
    }

    fn sample_token(
        &self,
        logits: &Tensor<WgpuRuntime>,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
    ) -> Result<u32> {
        validate_sampling_logits("sample_token", logits, SamplingLogitsShape::Vocab)?;
        let scratch = SamplingScratch::new(logits.shape()[0], logits.device())?;
        self.sample_token_into(logits, temperature, top_k, top_p, min_p, &scratch)?;
        let token: Vec<i64> = scratch.output.to_vec();
        let token = token.first().ok_or_else(|| Error::KernelError {
            reason: "sample_token: the kernel wrote no token".to_string(),
        })?;
        Ok(*token as u32)
    }

    fn sample_token_into(
        &self,
        logits: &Tensor<WgpuRuntime>,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
        scratch: &SamplingScratch<WgpuRuntime>,
    ) -> Result<()> {
        validate_sampling_logits("sample_token", logits, SamplingLogitsShape::Vocab)?;
        require_f32("sample_token", logits)?;
        let vocab_size = logits.shape()[0];
        validate_scratch_vocab("sample_token", scratch, vocab_size)?;
        filter_rule(temperature, top_k, top_p, min_p).check()?;
        scratch.draw_random(self, None)?;
        let args = SamplingArgs {
            num_unique: 0,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            temperature,
            top_k,
            top_p,
            min_p,
            greedy: false,
        };
        dispatch_logits_to_token(
            self,
            logits,
            0,
            vocab_size,
            &scratch.random,
            &scratch.random,
            &args,
            scratch,
        )
    }

    fn logits_to_token(
        &self,
        logits: &Tensor<WgpuRuntime>,
        token_ids: &Tensor<WgpuRuntime>,
        token_counts: &Tensor<WgpuRuntime>,
        num_unique: usize,
        repeat_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
        seed: Option<SampleSeed>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_sampling_logits(
            "logits_to_token",
            logits,
            SamplingLogitsShape::BatchSeqVocab,
        )?;
        let vocab_size = logits.shape()[logits.ndim() - 1];
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
        Ok(scratch.output)
    }

    fn logits_to_token_into(
        &self,
        logits: &Tensor<WgpuRuntime>,
        token_ids: &Tensor<WgpuRuntime>,
        token_counts: &Tensor<WgpuRuntime>,
        num_unique: usize,
        repeat_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
        seed: Option<SampleSeed>,
        scratch: &SamplingScratch<WgpuRuntime>,
    ) -> Result<()> {
        validate_sampling_logits(
            "logits_to_token",
            logits,
            SamplingLogitsShape::BatchSeqVocab,
        )?;
        validate_contiguous_arg("logits_to_token", "token_ids", token_ids)?;
        validate_contiguous_arg("logits_to_token", "token_counts", token_counts)?;
        require_f32("logits_to_token", logits)?;
        check_penalty_dtypes("logits_to_token", token_ids, token_counts)?;
        let vocab_size = logits.shape()[logits.ndim() - 1];
        if vocab_size == 0 || logits.numel() == 0 {
            return Err(Error::InvalidArgument {
                arg: "logits",
                reason: format!(
                    "logits_to_token: seq_len and vocab_size must be > 0, got shape {:?}",
                    logits.shape()
                ),
            });
        }
        validate_scratch_vocab("logits_to_token", scratch, vocab_size)?;
        let greedy = temperature == 0.0;
        if !greedy {
            filter_rule(temperature, top_k, top_p, min_p).check()?;
            scratch.draw_random(self, seed)?;
        }
        let args = SamplingArgs {
            num_unique: penalty_count(num_unique, token_ids, token_counts),
            repeat_penalty,
            frequency_penalty,
            presence_penalty,
            temperature,
            top_k,
            top_p,
            min_p,
            greedy,
        };
        let last_row = logits.numel() - vocab_size;
        dispatch_logits_to_token(
            self,
            logits,
            last_row,
            vocab_size,
            token_ids,
            token_counts,
            &args,
            scratch,
        )
    }
}
