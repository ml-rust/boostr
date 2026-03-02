//! WebGPU implementation of AttentionOps and FlashAttentionOps
//!
//! AttentionOps: delegates to impl_generic (Var-based, for autograd)
//! FlashAttentionOps: standard O(N²) GPU implementation (Tensor-based, no autograd)
//!
//! F32 only (WebGPU limitation).

use crate::error::{Error, Result};
use crate::ops::impl_generic::attention::multi_head_attention_impl;
use crate::ops::traits::{AttentionOps, FlashAttentionOps};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::BufferUsages;

const FLASH_SHADER_SOURCE: &str = include_str!("../shaders/attention/flash.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashParams {
    batch_size: u32,
    num_heads: u32,
    num_kv_heads: u32,
    seq_len_q: u32,
    seq_len_k: u32,
    head_dim: u32,
    scale: f32,
    causal: u32,
    window_size: u32,
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

impl AttentionOps<WgpuRuntime> for WgpuClient {
    fn multi_head_attention(
        &self,
        q: &Var<WgpuRuntime>,
        k: &Var<WgpuRuntime>,
        v: &Var<WgpuRuntime>,
        mask: Option<&Var<WgpuRuntime>>,
        num_heads: usize,
    ) -> Result<Var<WgpuRuntime>> {
        // Delegate to impl_generic for autograd support
        multi_head_attention_impl(self, q, k, v, mask, num_heads)
    }
}

impl FlashAttentionOps<WgpuRuntime> for WgpuClient {
    fn flash_attention_fwd(
        &self,
        q: &Tensor<WgpuRuntime>,
        k: &Tensor<WgpuRuntime>,
        v: &Tensor<WgpuRuntime>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
        window_size: usize,
        kv_seq_len: Option<usize>,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        // kv_seq_len override not optimized for WGPU — narrow if needed
        if let Some(seq_len) = kv_seq_len {
            let k_narrow = k.narrow(2, 0, seq_len)?.contiguous();
            let v_narrow = v.narrow(2, 0, seq_len)?.contiguous();
            return self.flash_attention_fwd(
                q,
                &k_narrow,
                &v_narrow,
                num_heads,
                num_kv_heads,
                head_dim,
                causal,
                window_size,
                None,
            );
        }
        validate_f32(q, "flash_attention_fwd")?;
        validate_f32(k, "flash_attention_fwd")?;
        validate_f32(v, "flash_attention_fwd")?;

        let q_shape = q.shape();
        let batch_size = q_shape[0];
        let seq_len_q = q_shape[2];
        let seq_len_k = k.shape()[2];

        // Create output tensors
        let output = Tensor::<WgpuRuntime>::zeros(q_shape, DType::F32, q.device());
        let lse_shape = vec![batch_size, num_heads, seq_len_q];
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
        let out_buf = get_buffer(output.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "output buffer not found".into(),
        })?;
        let lse_buf = get_buffer(lse.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "lse buffer not found".into(),
        })?;

        // Create params
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let params = FlashParams {
            batch_size: batch_size as u32,
            num_heads: num_heads as u32,
            num_kv_heads: num_kv_heads as u32,
            seq_len_q: seq_len_q as u32,
            seq_len_k: seq_len_k as u32,
            head_dim: head_dim as u32,
            scale,
            causal: if causal { 1 } else { 0 },
            window_size: window_size as u32,
            _pad: 0,
        };

        let params_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("flash_params"),
            size: std::mem::size_of::<FlashParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.wgpu_queue()
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        // Setup pipeline
        let cache = self.pipeline_cache();
        let module = cache.get_or_create_module("flash_attention_fwd_f32", FLASH_SHADER_SOURCE);

        let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
            num_storage_buffers: 5,
            num_uniform_buffers: 1,
            num_readonly_storage: 3,
        });
        let pipeline = cache.get_or_create_pipeline(
            "flash_attention_fwd_f32",
            "flash_attention_fwd_f32",
            &module,
            &layout,
        );

        let bind_group = cache.create_bind_group(
            &layout,
            &[&q_buf, &k_buf, &v_buf, &out_buf, &lse_buf, &params_buf],
        );

        // Dispatch
        let total_queries = (batch_size * num_heads * seq_len_q) as u32;
        let workgroups = total_queries.div_ceil(256);

        let mut encoder =
            self.wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("flash_attention_fwd"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flash_attention_fwd"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.wgpu_queue().submit(std::iter::once(encoder.finish()));

        Ok((output, lse))
    }

    fn flash_attention_fwd_fp8(
        &self,
        _q: &Tensor<WgpuRuntime>,
        _k: &Tensor<WgpuRuntime>,
        _v: &Tensor<WgpuRuntime>,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _causal: bool,
        _q_scale: f32,
        _k_scale: f32,
        _v_scale: f32,
        _o_scale: f32,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        // FP8 not supported in WebGPU (no native FP8 support in wgpu)
        // Dequantize to F32 and run standard attention
        Err(Error::InvalidArgument {
            arg: "dtype",
            reason: "WebGPU does not support FP8, use F32 instead".into(),
        })
    }

    fn flash_attention_bwd(
        &self,
        _dout: &Tensor<WgpuRuntime>,
        _q: &Tensor<WgpuRuntime>,
        _k: &Tensor<WgpuRuntime>,
        _v: &Tensor<WgpuRuntime>,
        _output: &Tensor<WgpuRuntime>,
        _lse: &Tensor<WgpuRuntime>,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _causal: bool,
        _window_size: usize,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        Err(Error::InvalidArgument {
            arg: "op",
            reason: "flash_attention_bwd not yet implemented on WebGPU".into(),
        })
    }

    fn flash_attention_bwd_fp8(
        &self,
        _dout: &Tensor<WgpuRuntime>,
        _q: &Tensor<WgpuRuntime>,
        _k: &Tensor<WgpuRuntime>,
        _v: &Tensor<WgpuRuntime>,
        _output: &Tensor<WgpuRuntime>,
        _lse: &Tensor<WgpuRuntime>,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _causal: bool,
        _q_scale: f32,
        _k_scale: f32,
        _v_scale: f32,
        _do_scale: f32,
        _o_scale: f32,
        _dq_scale: f32,
        _dk_scale: f32,
        _dv_scale: f32,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        Err(Error::InvalidArgument {
            arg: "op",
            reason: "flash_attention_bwd_fp8 not yet implemented on WebGPU".into(),
        })
    }
}
