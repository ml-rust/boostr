//! WebGPU implementation of SpeculativeOps
//!
//! verify_speculative_tokens: delegates to impl_generic (uses numr philox_uniform,
//!   results are Vec<VerificationResult> — CPU-side — so serial loop on CPU is correct).
//! compute_acceptance_probs: WGSL shader (element-wise, no RNG, F32 only).
//! compute_expected_tokens: WGSL shader (one thread per batch element, F32 only).

use crate::error::{Error, Result};
use crate::ops::impl_generic::inference::speculative::verify_speculative_tokens_impl;
use crate::ops::traits::inference::speculative::{SpeculativeOps, VerificationResult};
use numr::dtype::DType;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::BufferUsages;

const SPECULATIVE_SHADER: &str = include_str!("../shaders/inference/speculative_verify.wgsl");

fn validate_f32(t: &Tensor<WgpuRuntime>, op: &str) -> Result<()> {
    if t.dtype() != DType::F32 {
        return Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!(
                "{}: WebGPU speculative requires F32, got {:?}",
                op,
                t.dtype()
            ),
        });
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AcceptParams {
    total_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ExpectedParams {
    batch_size: u32,
    max_spec_tokens: u32,
    _pad0: u32,
    _pad1: u32,
}

impl SpeculativeOps<WgpuRuntime> for WgpuClient {
    fn verify_speculative_tokens(
        &self,
        draft_probs: &Tensor<WgpuRuntime>,
        target_probs: &Tensor<WgpuRuntime>,
        draft_tokens: &Tensor<WgpuRuntime>,
        seed: u64,
    ) -> Result<Vec<VerificationResult>> {
        // Delegate to impl_generic: generates uniform randoms via philox_uniform
        // (same algorithm as CPU/CUDA), then runs the serial accept/reject loop on CPU.
        // Results are Vec<VerificationResult> (CPU-side) regardless of backend.
        verify_speculative_tokens_impl(self, draft_probs, target_probs, draft_tokens, seed)
    }

    fn compute_acceptance_probs(
        &self,
        draft_probs: &Tensor<WgpuRuntime>,
        target_probs: &Tensor<WgpuRuntime>,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        validate_f32(draft_probs, "compute_acceptance_probs")?;
        validate_f32(target_probs, "compute_acceptance_probs")?;

        let dp_shape = draft_probs.shape();
        let tp_shape = target_probs.shape();

        if dp_shape != tp_shape {
            return Err(Error::InvalidArgument {
                arg: "target_probs",
                reason: format!("shape mismatch: {:?} vs {:?}", dp_shape, tp_shape),
            });
        }

        let total: usize = dp_shape.iter().product();
        let device = draft_probs.device();

        let acceptance = Tensor::<WgpuRuntime>::empty(dp_shape, DType::F32, device);
        let residual = Tensor::<WgpuRuntime>::empty(dp_shape, DType::F32, device);

        let dp_buf = get_buffer(draft_probs.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "draft buffer not found".into(),
        })?;
        let tp_buf =
            get_buffer(target_probs.storage().ptr()).ok_or_else(|| Error::KernelError {
                reason: "target buffer not found".into(),
            })?;
        let acc_buf = get_buffer(acceptance.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "acceptance buffer not found".into(),
        })?;
        let res_buf = get_buffer(residual.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "residual buffer not found".into(),
        })?;

        let params = AcceptParams {
            total_elements: total as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("accept_params"),
            size: std::mem::size_of::<AcceptParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.wgpu_queue()
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let cache = self.pipeline_cache();
        let module = cache.get_or_create_module("speculative_accept", SPECULATIVE_SHADER);
        let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
            num_storage_buffers: 4,
            num_uniform_buffers: 1,
            num_readonly_storage: 2,
        });
        let pipeline = cache.get_or_create_pipeline(
            "compute_acceptance_probs_f32",
            "compute_acceptance_probs_f32",
            &module,
            &layout,
        );

        let bind_group = cache.create_bind_group(
            &layout,
            &[&dp_buf, &tp_buf, &acc_buf, &res_buf, &params_buf],
        );

        let workgroups = (total as u32).div_ceil(256);

        let mut encoder =
            self.wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("acceptance_probs"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("acceptance_probs"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.wgpu_queue().submit(std::iter::once(encoder.finish()));

        Ok((acceptance, residual))
    }

    fn compute_expected_tokens(
        &self,
        acceptance_rates: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_f32(acceptance_rates, "compute_expected_tokens")?;

        let shape = acceptance_rates.shape();
        if shape.len() != 2 {
            return Err(Error::InvalidArgument {
                arg: "acceptance_rates",
                reason: format!("expected 2D [batch, K], got {}D", shape.len()),
            });
        }

        let batch_size = shape[0];
        let k = shape[1];
        let device = acceptance_rates.device();

        let output = Tensor::<WgpuRuntime>::empty(&[batch_size], DType::F32, device);

        let rates_buf =
            get_buffer(acceptance_rates.storage().ptr()).ok_or_else(|| Error::KernelError {
                reason: "rates buffer not found".into(),
            })?;
        let out_buf = get_buffer(output.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "output buffer not found".into(),
        })?;

        let params = ExpectedParams {
            batch_size: batch_size as u32,
            max_spec_tokens: k as u32,
            _pad0: 0,
            _pad1: 0,
        };

        let params_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("expected_params"),
            size: std::mem::size_of::<ExpectedParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.wgpu_queue()
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let cache = self.pipeline_cache();
        let module = cache.get_or_create_module("speculative_expected", SPECULATIVE_SHADER);
        let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
            num_storage_buffers: 2,
            num_uniform_buffers: 1,
            num_readonly_storage: 1,
        });
        let pipeline = cache.get_or_create_pipeline(
            "compute_expected_tokens_f32",
            "compute_expected_tokens_f32",
            &module,
            &layout,
        );

        let bind_group = cache.create_bind_group(&layout, &[&rates_buf, &out_buf, &params_buf]);

        let workgroups = (batch_size as u32).div_ceil(256);

        let mut encoder =
            self.wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("expected_tokens"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("expected_tokens"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.wgpu_queue().submit(std::iter::once(encoder.finish()));

        Ok(output)
    }
}
