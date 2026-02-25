//! WebGPU implementation of VarLenAttentionOps
//!
//! Variable-length (packed) Flash Attention with cu_seqlens indexing.
//! F32 only (WebGPU limitation).

use crate::error::{Error, Result};
use crate::ops::traits::VarLenAttentionOps;
use numr::dtype::DType;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::BufferUsages;

const VARLEN_SHADER_SOURCE: &str = include_str!("../shaders/attention/varlen_attention.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VarlenParams {
    total_tokens_q: u32,
    total_tokens_k: u32,
    num_heads: u32,
    head_dim: u32,
    batch_size: u32,
    causal: u32,
    scale: f32,
    _pad: u32,
}

fn validate_f32(t: &Tensor<WgpuRuntime>, op: &str) -> Result<()> {
    if t.dtype() != DType::F32 {
        return Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!("{}: WebGPU requires F32, got {:?}", op, t.dtype()),
        });
    }
    Ok(())
}

impl VarLenAttentionOps<WgpuRuntime> for WgpuClient {
    fn varlen_attention_fwd(
        &self,
        q: &Tensor<WgpuRuntime>,
        k: &Tensor<WgpuRuntime>,
        v: &Tensor<WgpuRuntime>,
        cu_seqlens_q: &Tensor<WgpuRuntime>,
        cu_seqlens_k: &Tensor<WgpuRuntime>,
        batch_size: usize,
        num_heads: usize,
        _max_seqlen_q: usize,
        _max_seqlen_k: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        validate_f32(q, "varlen_attention_fwd")?;
        validate_f32(k, "varlen_attention_fwd")?;
        validate_f32(v, "varlen_attention_fwd")?;
        // cu_seqlens are I32 — do NOT validate as F32

        let total_tokens_q = q.shape()[0];
        let total_tokens_k = k.shape()[0];

        // Create output tensors
        let output = Tensor::<WgpuRuntime>::zeros(q.shape(), DType::F32, q.device());
        let lse_shape = vec![total_tokens_q, num_heads];
        let lse = Tensor::<WgpuRuntime>::zeros(&lse_shape, DType::F32, q.device());

        // Get buffers
        let q_buf = get_buffer(q.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "q buffer not found".into(),
        })?;
        let k_buf = get_buffer(k.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "k buffer not found".into(),
        })?;
        let v_buf = get_buffer(v.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "v buffer not found".into(),
        })?;
        let cu_q_buf =
            get_buffer(cu_seqlens_q.storage().ptr()).ok_or_else(|| Error::KernelError {
                reason: "cu_seqlens_q buffer not found".into(),
            })?;
        let cu_k_buf =
            get_buffer(cu_seqlens_k.storage().ptr()).ok_or_else(|| Error::KernelError {
                reason: "cu_seqlens_k buffer not found".into(),
            })?;
        let out_buf = get_buffer(output.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "output buffer not found".into(),
        })?;
        let lse_buf = get_buffer(lse.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "lse buffer not found".into(),
        })?;

        let params = VarlenParams {
            total_tokens_q: total_tokens_q as u32,
            total_tokens_k: total_tokens_k as u32,
            num_heads: num_heads as u32,
            head_dim: head_dim as u32,
            batch_size: batch_size as u32,
            causal: if causal { 1 } else { 0 },
            scale: 1.0f32 / (head_dim as f32).sqrt(),
            _pad: 0,
        };

        let params_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("varlen_params"),
            size: std::mem::size_of::<VarlenParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.wgpu_queue()
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let cache = self.pipeline_cache();
        let module = cache.get_or_create_module("varlen_attention_fwd_f32", VARLEN_SHADER_SOURCE);

        let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
            num_storage_buffers: 7,
            num_uniform_buffers: 1,
            num_readonly_storage: 5,
        });
        let pipeline = cache.get_or_create_pipeline(
            "varlen_attention_fwd_f32",
            "varlen_attention_fwd_f32",
            &module,
            &layout,
        );

        let bind_group = cache.create_bind_group(
            &layout,
            &[
                &q_buf,
                &k_buf,
                &v_buf,
                &cu_q_buf,
                &cu_k_buf,
                &out_buf,
                &lse_buf,
                &params_buf,
            ],
        );

        let workgroups = (total_tokens_q as u32).div_ceil(256);

        let mut encoder =
            self.wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("varlen_attention_fwd"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("varlen_attention_fwd"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.wgpu_queue().submit(std::iter::once(encoder.finish()));

        Ok((output, lse))
    }

    fn varlen_attention_bwd(
        &self,
        _dout: &Tensor<WgpuRuntime>,
        _q: &Tensor<WgpuRuntime>,
        _k: &Tensor<WgpuRuntime>,
        _v: &Tensor<WgpuRuntime>,
        _output: &Tensor<WgpuRuntime>,
        _lse: &Tensor<WgpuRuntime>,
        _cu_seqlens_q: &Tensor<WgpuRuntime>,
        _cu_seqlens_k: &Tensor<WgpuRuntime>,
        _batch_size: usize,
        _num_heads: usize,
        _max_seqlen_q: usize,
        _max_seqlen_k: usize,
        _head_dim: usize,
        _causal: bool,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        Err(Error::InvalidArgument {
            arg: "op",
            reason: "varlen_attention_bwd not yet implemented on WebGPU".into(),
        })
    }
}
