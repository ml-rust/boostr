//! WebGPU implementation of DequantOps

use crate::error::{Error, Result};
use crate::quant::traits::DequantOps;
use crate::quant::{QuantFormat, QuantTensor};
use numr::dtype::DType;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::BufferUsages;

use super::shaders::dequant as shader_gen;

/// Params struct matching WGSL DequantParams
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DequantParams {
    num_blocks: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

impl DequantOps<WgpuRuntime> for WgpuClient {
    fn dequantize(
        &self,
        qt: &QuantTensor<WgpuRuntime>,
        target_dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(target_dtype, DType::F32) {
            return Err(Error::QuantError {
                reason: format!(
                    "WebGPU dequantize target must be F32, got {:?}",
                    target_dtype
                ),
            });
        }

        let (shader_source, entry_point) = match qt.format() {
            QuantFormat::Q4_0 => (shader_gen::generate_dequant_q4_0_shader(), "dequant_q4_0"),
            QuantFormat::Q8_0 => (shader_gen::generate_dequant_q8_0_shader(), "dequant_q8_0"),
            QuantFormat::Q4K => (shader_gen::generate_dequant_q4_k_shader(), "dequant_q4_k"),
            QuantFormat::Q6K => (shader_gen::generate_dequant_q6_k_shader(), "dequant_q6_k"),
            other => {
                return Err(Error::UnsupportedQuantFormat {
                    format: format!("{} (WebGPU dequant not implemented)", other),
                });
            }
        };

        let num_blocks = qt.num_blocks();

        // Get input buffer from numr's buffer registry
        let input_buf = get_buffer(qt.storage().ptr()).ok_or_else(|| Error::QuantError {
            reason: "input buffer not found in WebGPU registry".into(),
        })?;

        // Allocate f32 output tensor
        let f32_out = Tensor::<WgpuRuntime>::empty(qt.shape(), DType::F32, qt.device());
        let output_buf = get_buffer(f32_out.storage().ptr()).ok_or_else(|| Error::QuantError {
            reason: "output buffer not found in WebGPU registry".into(),
        })?;

        // Create params buffer
        let params = DequantParams {
            num_blocks: num_blocks as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("dequant_params"),
            size: std::mem::size_of::<DequantParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.wgpu_queue()
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        // Create pipeline
        let cache = self.pipeline_cache();
        let module = cache.get_or_create_module(entry_point, &shader_source);
        let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
            num_storage_buffers: 2,
            num_uniform_buffers: 1,
            num_readonly_storage: 0,
        });
        let pipeline = cache.get_or_create_pipeline(entry_point, entry_point, &module, &layout);

        let bind_group = cache.create_bind_group(&layout, &[&input_buf, &output_buf, &params_buf]);

        // Dispatch
        let mut encoder =
            self.wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dequant"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dequant"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let workgroups = (num_blocks as u32).div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.wgpu_queue().submit(std::iter::once(encoder.finish()));

        Ok(f32_out)
    }
}
