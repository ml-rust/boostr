//! WebGPU implementation of MlaOps
//!
//! Uses the SDPA WGSL shader to avoid 4D matmul limitation in WebGPU.

use crate::error::{Error, Result};
use crate::ops::traits::MlaOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::BufferUsages;

const SDPA_SHADER_SOURCE: &str = include_str!("../shaders/attention/sdpa.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SdpaParams {
    batch_size: u32,
    num_heads: u32,
    seq_len_q: u32,
    seq_len_k: u32,
    head_dim_k: u32,
    head_dim_v: u32,
    scale: f32,
    causal: u32,
}

impl MlaOps<WgpuRuntime> for WgpuClient {
    fn scaled_dot_product_attention(
        &self,
        q: &Var<WgpuRuntime>,
        k: &Var<WgpuRuntime>,
        v: &Var<WgpuRuntime>,
        scale: f64,
        causal: bool,
    ) -> Result<Var<WgpuRuntime>> {
        let q_tensor = q.tensor();
        let k_tensor = k.tensor();
        let v_tensor = v.tensor();

        let q_shape = q_tensor.shape();
        let k_shape = k_tensor.shape();
        let v_shape = v_tensor.shape();

        if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
            return Err(Error::InvalidArgument {
                arg: "q/k/v",
                reason: "expected 4D tensors [B, H, S, D]".into(),
            });
        }

        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let seq_len_q = q_shape[2];
        let head_dim_k = q_shape[3];
        let seq_len_k = k_shape[2];
        let head_dim_v = v_shape[3];

        if q_tensor.dtype() != DType::F32 {
            return Err(Error::InvalidArgument {
                arg: "dtype",
                reason: format!("WebGPU SDPA requires F32, got {:?}", q_tensor.dtype()),
            });
        }

        // Create output tensor: [B, H, S_q, D_v]
        let output = Tensor::<WgpuRuntime>::zeros(
            &[batch_size, num_heads, seq_len_q, head_dim_v],
            DType::F32,
            q_tensor.device(),
        );

        let q_buf = get_buffer(q_tensor.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "q buffer not found".into(),
        })?;
        let k_buf = get_buffer(k_tensor.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "k buffer not found".into(),
        })?;
        let v_buf = get_buffer(v_tensor.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "v buffer not found".into(),
        })?;
        let out_buf = get_buffer(output.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "output buffer not found".into(),
        })?;

        let params = SdpaParams {
            batch_size: batch_size as u32,
            num_heads: num_heads as u32,
            seq_len_q: seq_len_q as u32,
            seq_len_k: seq_len_k as u32,
            head_dim_k: head_dim_k as u32,
            head_dim_v: head_dim_v as u32,
            scale: scale as f32,
            causal: if causal { 1 } else { 0 },
        };

        let params_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("sdpa_params"),
            size: std::mem::size_of::<SdpaParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.wgpu_queue()
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let cache = self.pipeline_cache();
        let module = cache.get_or_create_module("sdpa_forward_f32", SDPA_SHADER_SOURCE);

        // Shader bindings: 0=q(read), 1=k(read), 2=v(read), 3=out(rw)
        let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
            num_storage_buffers: 4,
            num_uniform_buffers: 1,
            num_readonly_storage: 3,
        });
        let pipeline =
            cache.get_or_create_pipeline("sdpa_forward_f32", "sdpa_forward_f32", &module, &layout);

        let bind_group =
            cache.create_bind_group(&layout, &[&q_buf, &k_buf, &v_buf, &out_buf, &params_buf]);

        let total_queries = (batch_size * num_heads * seq_len_q) as u32;
        let workgroups = (total_queries + 255) / 256;

        let mut encoder =
            self.wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sdpa"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sdpa"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.wgpu_queue().submit(std::iter::once(encoder.finish()));

        Ok(Var::new(output, false))
    }
}
