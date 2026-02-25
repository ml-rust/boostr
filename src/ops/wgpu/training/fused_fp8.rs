//! WebGPU implementation of FusedFp8TrainingOps
//!
//! F32 only (WebGPU is 32-bit by design).
//! Uses a two-pass approach: pass 1 computes unscaled values + norm²,
//! pass 2 applies clipping if needed (after readback of norm²).

use crate::error::{Error, Result};
use crate::ops::impl_generic::training::dynamic_loss_scale_update_impl;
use crate::ops::traits::FusedFp8TrainingOps;
use numr::dtype::DType;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::BufferUsages;

const SHADER_SOURCE: &str = include_str!("../shaders/training/fused_grad_unscale_clip.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UnscaleClipParams {
    inv_scale: f32,
    max_norm: f32,
    n: u32,
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

impl FusedFp8TrainingOps<WgpuRuntime> for WgpuClient {
    fn fused_grad_unscale_clip(
        &self,
        grad: &Tensor<WgpuRuntime>,
        max_norm: f64,
        loss_scale: f64,
    ) -> Result<(Tensor<WgpuRuntime>, f64, bool)> {
        validate_f32(grad, "fused_grad_unscale_clip")?;
        let n: usize = grad.shape().iter().product();

        // Allocate a fresh output tensor — can't alias grad (read vs read_write conflict)
        let out = Tensor::<WgpuRuntime>::zeros(grad.shape(), DType::F32, grad.device());

        let out_buf = get_buffer(out.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "out buffer not found".into(),
        })?;
        let grad_buf = get_buffer(grad.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "grad buffer not found".into(),
        })?;

        // Create result buffer for atomics: [found_inf: u32, norm_sq: f32]
        let result_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("unscale_clip_result"),
            size: 8, // 2 x u32/f32
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Zero-initialize
        self.wgpu_queue().write_buffer(&result_buf, 0, &[0u8; 8]);

        let params = UnscaleClipParams {
            inv_scale: (1.0 / loss_scale) as f32,
            max_norm: max_norm as f32,
            n: n as u32,
            _pad: 0,
        };
        let params_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("unscale_clip_params"),
            size: std::mem::size_of::<UnscaleClipParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.wgpu_queue()
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let cache = self.pipeline_cache();
        let module = cache.get_or_create_module("fused_grad_unscale_clip_f32", SHADER_SOURCE);

        let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
            num_storage_buffers: 3, // grad(read), out(rw), result(rw)
            num_uniform_buffers: 1,
            num_readonly_storage: 1,
        });
        let pipeline = cache.get_or_create_pipeline(
            "fused_grad_unscale_clip_f32",
            "fused_grad_unscale_clip_f32",
            &module,
            &layout,
        );

        let bind_group =
            cache.create_bind_group(&layout, &[&grad_buf, &out_buf, &result_buf, &params_buf]);

        let workgroups = (n as u32).div_ceil(256);
        let mut encoder =
            self.wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("unscale_clip"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("unscale_clip"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Readback buffer
        let readback_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("unscale_clip_readback"),
            size: 8,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&result_buf, 0, &readback_buf, 0, 8);

        self.wgpu_queue().submit(std::iter::once(encoder.finish()));

        // Map and read results
        let slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        let _ = self.wgpu_device().poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(60)),
        });
        rx.recv()
            .map_err(|e| Error::KernelError {
                reason: format!("readback channel error: {:?}", e),
            })?
            .map_err(|e| Error::KernelError {
                reason: format!("readback map error: {:?}", e),
            })?;

        let data = slice.get_mapped_range();
        let results: &[u32] = bytemuck::cast_slice(&data);
        let found_inf = results[0] != 0;
        let norm_sq = f32::from_bits(results[1]);
        drop(data);
        readback_buf.unmap();

        let norm = (norm_sq as f64).sqrt();

        // If clipping needed and no inf, apply clip_coef via a second dispatch
        if !found_inf && norm > max_norm && max_norm > 0.0 {
            let clip_coef = (max_norm / (norm + 1e-6)) as f32;
            // Simple scale: multiply all elements by clip_coef
            // We reuse the shader's clip pass entry point
            let clip_params = UnscaleClipParams {
                inv_scale: clip_coef,
                max_norm: 0.0, // unused in clip pass
                n: n as u32,
                _pad: 0,
            };
            self.wgpu_queue()
                .write_buffer(&params_buf, 0, bytemuck::bytes_of(&clip_params));

            // Zero result buf again
            self.wgpu_queue().write_buffer(&result_buf, 0, &[0u8; 8]);

            let clip_pipeline =
                cache.get_or_create_pipeline("clip_scale_f32", "clip_scale_f32", &module, &layout);
            // Binding 0 is read-only, can't alias with binding 1 (read_write out_buf)
            let dummy = Tensor::<WgpuRuntime>::zeros(&[1], DType::F32, grad.device());
            let dummy_buf =
                get_buffer(dummy.storage().ptr()).ok_or_else(|| Error::KernelError {
                    reason: "dummy buffer not found".into(),
                })?;
            let clip_bind_group =
                cache.create_bind_group(&layout, &[&dummy_buf, &out_buf, &result_buf, &params_buf]);

            let mut encoder2 =
                self.wgpu_device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("clip_scale"),
                    });
            {
                let mut pass = encoder2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("clip_scale"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&clip_pipeline);
                pass.set_bind_group(0, Some(&clip_bind_group), &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            self.wgpu_queue().submit(std::iter::once(encoder2.finish()));
        }

        Ok((out, norm, found_inf))
    }

    fn dynamic_loss_scale_update(
        &self,
        found_inf: bool,
        loss_scale: f64,
        growth_tracker: i32,
        growth_interval: i32,
        backoff_factor: f64,
    ) -> Result<(f64, i32)> {
        dynamic_loss_scale_update_impl(
            found_inf,
            loss_scale,
            growth_tracker,
            growth_interval,
            backoff_factor,
        )
    }
}
