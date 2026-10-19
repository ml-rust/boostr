//! Buffer helpers and the fused logits-to-token dispatch for WebGPU sampling.

use std::sync::Arc;

use crate::error::{Error, Result};
use crate::ops::traits::inference::sampling_scratch::SamplingScratch;
use numr::runtime::wgpu::shaders::LayoutKey;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::{Buffer, BufferUsages};

const LOGITS_TO_TOKEN_SHADER: &str = include_str!("../../shaders/inference/logits_to_token.wgsl");

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

/// Penalty settings plus the sampling chain, as one kernel call takes them.
pub(super) struct SamplingArgs {
    pub(super) num_unique: usize,
    pub(super) repeat_penalty: f32,
    pub(super) frequency_penalty: f32,
    pub(super) presence_penalty: f32,
    pub(super) temperature: f32,
    pub(super) top_k: usize,
    pub(super) top_p: f32,
    pub(super) min_p: f32,
    pub(super) greedy: bool,
}

/// The storage buffer behind `tensor`.
pub(in crate::ops::wgpu::inference) fn tensor_buffer(
    tensor: &Tensor<WgpuRuntime>,
    what: &str,
) -> Result<Arc<Buffer>> {
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
pub(in crate::ops::wgpu::inference) fn uniform_buffer<T: bytemuck::Pod>(
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

/// Run the fused kernel over the vocab row at `logits_offset`.
#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_logits_to_token(
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
