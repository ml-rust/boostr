//! WebGPU implementation of PagedAttentionOps
//!
//! vLLM-style paged attention with block-table KV cache indirection.
//! F32 only (WebGPU limitation).

use crate::error::{Error, Result};
use crate::ops::traits::PagedAttentionOps;
use numr::dtype::DType;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::BufferUsages;

const PAGED_SHADER_SOURCE: &str = include_str!("../shaders/attention/paged_attention.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PagedParams {
    batch_size: u32,
    num_heads: u32,
    num_kv_heads: u32,
    seq_len_q: u32,
    seq_len_k: u32,
    head_dim: u32,
    block_size: u32,
    max_num_blocks: u32,
    scale: f32,
    causal: u32,
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

impl PagedAttentionOps<WgpuRuntime> for WgpuClient {
    fn paged_attention_fwd(
        &self,
        q: &Tensor<WgpuRuntime>,
        k_blocks: &Tensor<WgpuRuntime>,
        v_blocks: &Tensor<WgpuRuntime>,
        block_table: &Tensor<WgpuRuntime>,
        num_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
        head_dim: usize,
        block_size: usize,
        causal: bool,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        validate_f32(q, "paged_attention_fwd")?;
        validate_f32(k_blocks, "paged_attention_fwd")?;
        validate_f32(v_blocks, "paged_attention_fwd")?;

        let q_shape = q.shape();
        let batch_size = q_shape[0];

        let output = Tensor::<WgpuRuntime>::zeros(q_shape, DType::F32, q.device());
        let lse_shape = vec![batch_size, num_heads, seq_len_q];
        let lse = Tensor::<WgpuRuntime>::zeros(&lse_shape, DType::F32, q.device());

        let q_buf = get_buffer(q.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "q buffer not found".into(),
        })?;
        let k_buf = get_buffer(k_blocks.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "k_blocks buffer not found".into(),
        })?;
        let v_buf = get_buffer(v_blocks.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "v_blocks buffer not found".into(),
        })?;
        let bt_buf = get_buffer(block_table.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "block_table buffer not found".into(),
        })?;
        let out_buf = get_buffer(output.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "output buffer not found".into(),
        })?;
        let lse_buf = get_buffer(lse.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "lse buffer not found".into(),
        })?;

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        // Infer num_kv_heads from k_blocks shape: [num_blocks, block_size, num_kv_heads, head_dim]
        let k_shape = k_blocks.shape();
        let num_kv_heads = if k_shape.len() == 4 { k_shape[2] } else { 1 };
        let max_num_blocks = block_table.shape()[1];
        let params = PagedParams {
            batch_size: batch_size as u32,
            num_heads: num_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            seq_len_q: seq_len_q as u32,
            seq_len_k: seq_len_k as u32,
            head_dim: head_dim as u32,
            block_size: block_size as u32,
            max_num_blocks: max_num_blocks as u32,
            scale,
            causal: if causal { 1 } else { 0 },
        };

        let params_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("paged_params"),
            size: std::mem::size_of::<PagedParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.wgpu_queue()
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let cache = self.pipeline_cache();
        let module = cache.get_or_create_module("paged_attention_fwd_f32", PAGED_SHADER_SOURCE);

        let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
            num_storage_buffers: 6,
            num_uniform_buffers: 1,
            num_readonly_storage: 4,
        });
        let pipeline = cache.get_or_create_pipeline(
            "paged_attention_fwd_f32",
            "paged_attention_fwd_f32",
            &module,
            &layout,
        );

        let bind_group = cache.create_bind_group(
            &layout,
            &[
                &q_buf,
                &k_buf,
                &v_buf,
                &bt_buf,
                &out_buf,
                &lse_buf,
                &params_buf,
            ],
        );

        let total_queries = (batch_size * num_heads * seq_len_q) as u32;
        let workgroups = (total_queries + 255) / 256;

        let mut encoder =
            self.wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("paged_attention_fwd"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("paged_attention_fwd"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.wgpu_queue().submit(std::iter::once(encoder.finish()));

        Ok((output, lse))
    }

    fn paged_attention_fwd_fp8(
        &self,
        _q: &Tensor<WgpuRuntime>,
        _k_blocks: &Tensor<WgpuRuntime>,
        _v_blocks: &Tensor<WgpuRuntime>,
        _block_table: &Tensor<WgpuRuntime>,
        _num_heads: usize,
        _seq_len_q: usize,
        _seq_len_k: usize,
        _head_dim: usize,
        _block_size: usize,
        _causal: bool,
        _q_scale: f32,
        _k_scale: f32,
        _v_scale: f32,
        _o_scale: f32,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        Err(Error::InvalidArgument {
            arg: "dtype",
            reason: "WebGPU does not support FP8, use F32 instead".into(),
        })
    }

    fn paged_attention_bwd(
        &self,
        _dout: &Tensor<WgpuRuntime>,
        _q: &Tensor<WgpuRuntime>,
        _k_blocks: &Tensor<WgpuRuntime>,
        _v_blocks: &Tensor<WgpuRuntime>,
        _output: &Tensor<WgpuRuntime>,
        _lse: &Tensor<WgpuRuntime>,
        _block_table: &Tensor<WgpuRuntime>,
        _num_heads: usize,
        _seq_len_q: usize,
        _seq_len_k: usize,
        _head_dim: usize,
        _block_size: usize,
        _causal: bool,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        Err(Error::InvalidArgument {
            arg: "op",
            reason: "paged_attention_bwd not yet implemented on WebGPU".into(),
        })
    }
}
