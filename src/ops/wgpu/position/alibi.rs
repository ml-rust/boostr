//! WebGPU implementation of AlibiOps
//!
//! F32 only (WebGPU limitation).

use crate::error::{Error, Result};
use crate::ops::traits::AlibiOps;
use numr::dtype::DType;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::BufferUsages;

const SHADER_SOURCE: &str = include_str!("../shaders/position/alibi.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AlibiParams {
    batch_size: u32,
    num_heads: u32,
    seq_len_q: u32,
    seq_len_k: u32,
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

impl AlibiOps<WgpuRuntime> for WgpuClient {
    fn alibi_add_bias(
        &self,
        scores: &Tensor<WgpuRuntime>,
        batch_size: usize,
        num_heads: usize,
        seq_len_q: usize,
        seq_len_k: usize,
    ) -> Result<()> {
        validate_f32(scores, "alibi_add_bias")?;

        let scores_buf = get_buffer(scores.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "scores buffer not found".into(),
        })?;

        let params = AlibiParams {
            batch_size: batch_size as u32,
            num_heads: num_heads as u32,
            seq_len_q: seq_len_q as u32,
            seq_len_k: seq_len_k as u32,
        };

        let params_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("alibi_params"),
            size: std::mem::size_of::<AlibiParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.wgpu_queue()
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let cache = self.pipeline_cache();
        let module = cache.get_or_create_module("alibi_add_bias_f32", SHADER_SOURCE);

        let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
            num_storage_buffers: 1,
            num_uniform_buffers: 1,
            num_readonly_storage: 0,
        });
        let pipeline = cache.get_or_create_pipeline(
            "alibi_add_bias_f32",
            "alibi_add_bias_f32",
            &module,
            &layout,
        );

        let bind_group = cache.create_bind_group(&layout, &[&scores_buf, &params_buf]);

        let total_elems = (batch_size * num_heads * seq_len_q * seq_len_k) as u32;
        let workgroups = (total_elems + 255) / 256;

        let mut encoder =
            self.wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("alibi_add_bias"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("alibi_add_bias"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.wgpu_queue().submit(std::iter::once(encoder.finish()));

        Ok(())
    }
}
