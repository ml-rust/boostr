//! WebGPU implementation of SamplingOps — WGSL kernels, F32 only.
//!
//! - Penalties: one invocation per unique token, in place.
//! - `sample_token` and `logits_to_token`: one fused single-workgroup
//!   kernel. It copies the last position, applies penalties, then takes the
//!   argmax or runs temperature → softmax → top-k → min-p → top-p →
//!   multinomial. The random draw is made on the device with
//!   `RandomOps::rand_into`.
//!
//! Every token stays on the device. The caller reads it back once.

use std::sync::Arc;

use crate::error::{Error, Result};
use crate::ops::impl_generic::inference::sampling::{
    SamplingLogitsShape, validate_contiguous_arg, validate_sampling_logits, validate_scratch_vocab,
};
use crate::ops::traits::inference::sampling::SamplingOps;
use crate::ops::traits::inference::sampling_scratch::SamplingScratch;
use numr::dtype::DType;
use numr::runtime::wgpu::shaders::LayoutKey;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::{Buffer, BufferUsages};

const LOGITS_TO_TOKEN_SHADER: &str = include_str!("../shaders/inference/logits_to_token.wgsl");
const PENALTIES_SHADER: &str = include_str!("../shaders/inference/sampling_penalties.wgsl");

/// Matches `LogitsToTokenParams` in `logits_to_token.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LogitsToTokenParams {
    logits_offset: u32,
    vocab_size: u32,
    num_unique: u32,
    top_k: u32,
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    temperature: f32,
    top_p: f32,
    min_p: f32,
    greedy: u32,
    _pad: u32,
}

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

/// Penalty settings plus the sampling chain, as one kernel call takes them.
struct SamplingArgs {
    num_unique: usize,
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    greedy: bool,
}

/// The storage buffer behind `tensor`.
pub(super) fn tensor_buffer(tensor: &Tensor<WgpuRuntime>, what: &str) -> Result<Arc<Buffer>> {
    get_buffer(tensor.storage().ptr()).ok_or_else(|| Error::KernelError {
        reason: format!("{what} buffer is not registered with the WebGPU client"),
    })
}

/// The buffer behind `tensor`, or a one-word stand-in when it is empty.
///
/// An empty tensor owns no buffer. WebGPU still needs a binding.
fn buffer_or_placeholder(
    client: &WgpuClient,
    tensor: &Tensor<WgpuRuntime>,
    what: &str,
) -> Result<Arc<Buffer>> {
    if tensor.numel() == 0 {
        return Ok(Arc::new(client.wgpu_device().create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("sampling_placeholder"),
                size: 8,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
        )));
    }
    tensor_buffer(tensor, what)
}

/// A uniform buffer holding `params`.
pub(super) fn uniform_buffer<T: bytemuck::Pod>(
    client: &WgpuClient,
    label: &'static str,
    params: &T,
) -> Buffer {
    let buffer = client.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<T>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .wgpu_queue()
        .write_buffer(&buffer, 0, bytemuck::bytes_of(params));
    buffer
}

/// Encode and submit one compute pass.
pub(super) fn submit_pass(
    client: &WgpuClient,
    label: &'static str,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups: u32,
) {
    let mut encoder = client
        .wgpu_device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, Some(bind_group), &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    client
        .wgpu_queue()
        .submit(std::iter::once(encoder.finish()));
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

/// Run the fused kernel over the vocab row at `logits_offset`.
#[allow(clippy::too_many_arguments)]
fn dispatch_logits_to_token(
    client: &WgpuClient,
    logits: &Tensor<WgpuRuntime>,
    logits_offset: usize,
    vocab_size: usize,
    token_ids: &Tensor<WgpuRuntime>,
    token_counts: &Tensor<WgpuRuntime>,
    args: &SamplingArgs,
    scratch: &SamplingScratch<WgpuRuntime>,
) -> Result<()> {
    let logits_buf = tensor_buffer(logits, "logits")?;
    let ids_buf = buffer_or_placeholder(client, token_ids, "token_ids")?;
    let counts_buf = buffer_or_placeholder(client, token_counts, "token_counts")?;
    let random_buf = tensor_buffer(&scratch.random, "scratch.random")?;
    let probs_buf = tensor_buffer(&scratch.probs, "scratch.probs")?;
    let output_buf = tensor_buffer(&scratch.output, "scratch.output")?;

    let params = LogitsToTokenParams {
        logits_offset: (logits.offset() + logits_offset) as u32,
        vocab_size: vocab_size as u32,
        num_unique: args.num_unique as u32,
        top_k: args.top_k.min(u32::MAX as usize) as u32,
        repeat_penalty: args.repeat_penalty,
        frequency_penalty: args.frequency_penalty,
        presence_penalty: args.presence_penalty,
        temperature: args.temperature,
        top_p: args.top_p,
        min_p: args.min_p,
        greedy: u32::from(args.greedy),
        _pad: 0,
    };
    let params_buf = uniform_buffer(client, "logits_to_token_params", &params);

    let cache = client.pipeline_cache();
    let module = cache.get_or_create_module("logits_to_token", LOGITS_TO_TOKEN_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 6,
        num_uniform_buffers: 1,
        num_readonly_storage: 4,
    });
    let pipeline =
        cache.get_or_create_pipeline("logits_to_token", "logits_to_token", &module, &layout);
    let bind_group = cache.create_bind_group(
        &layout,
        &[
            &logits_buf,
            &ids_buf,
            &counts_buf,
            &random_buf,
            &probs_buf,
            &output_buf,
            &params_buf,
        ],
    );
    submit_pass(client, "logits_to_token", &pipeline, &bind_group, 1);
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
        Ok(token[0] as u32)
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
        seed: Option<u64>,
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
        seed: Option<u64>,
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
