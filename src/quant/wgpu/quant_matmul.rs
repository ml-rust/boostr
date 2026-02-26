//! WebGPU implementation of QuantMatmulOps

use crate::error::{Error, Result};
use crate::quant::traits::QuantMatmulOps;
use crate::quant::{QuantFormat, QuantTensor};
use numr::dtype::DType;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::BufferUsages;

use super::int4_gemm as int4_dispatch;
use super::shaders::quant_matmul as shader_gen;

/// Params struct matching WGSL MatmulParams
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulParams {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

impl QuantMatmulOps<WgpuRuntime> for WgpuClient {
    fn int4_gemm(
        &self,
        input: &Tensor<WgpuRuntime>,
        qweight: &Tensor<WgpuRuntime>,
        scales: &Tensor<WgpuRuntime>,
        zeros: &Tensor<WgpuRuntime>,
        group_size: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        if input.dtype() != DType::F32 {
            return Err(Error::QuantError {
                reason: format!("int4_gemm input must be F32, got {:?}", input.dtype()),
            });
        }
        let in_shape = input.shape();
        let k = in_shape[in_shape.len() - 1];
        let m: usize = in_shape.iter().product::<usize>() / k;
        let n = qweight.shape()[1] * 8;
        let act_contig = input.contiguous();

        let mut out_shape = in_shape[..in_shape.len() - 1].to_vec();
        out_shape.push(n);
        let output = Tensor::<WgpuRuntime>::empty(&out_shape, DType::F32, input.device());
        int4_dispatch::dispatch_int4_gemm(
            self,
            &act_contig,
            qweight,
            scales,
            zeros,
            &output,
            m as u32,
            k as u32,
            n as u32,
            group_size as u32,
        )?;
        Ok(output)
    }

    fn int4_gemm_gptq(
        &self,
        input: &Tensor<WgpuRuntime>,
        qweight: &Tensor<WgpuRuntime>,
        qzeros: &Tensor<WgpuRuntime>,
        scales: &Tensor<WgpuRuntime>,
        g_idx: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        if input.dtype() != DType::F32 {
            return Err(Error::QuantError {
                reason: format!("int4_gemm_gptq input must be F32, got {:?}", input.dtype()),
            });
        }
        let in_shape = input.shape();
        let k = in_shape[in_shape.len() - 1];
        let m: usize = in_shape.iter().product::<usize>() / k;
        let n = qweight.shape()[1];
        let act_contig = input.contiguous();

        let mut out_shape = in_shape[..in_shape.len() - 1].to_vec();
        out_shape.push(n);
        let output = Tensor::<WgpuRuntime>::empty(&out_shape, DType::F32, input.device());
        int4_dispatch::dispatch_int4_gemm_gptq(
            self,
            &act_contig,
            qweight,
            qzeros,
            scales,
            g_idx,
            &output,
            m as u32,
            k as u32,
            n as u32,
        )?;
        Ok(output)
    }

    fn marlin_gemm(
        &self,
        input: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        scales: &Tensor<WgpuRuntime>,
        zeros: &Tensor<WgpuRuntime>,
        group_size: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        if input.dtype() != DType::F32 {
            return Err(Error::QuantError {
                reason: format!("marlin_gemm input must be F32, got {:?}", input.dtype()),
            });
        }
        let in_shape = input.shape();
        let k = in_shape[in_shape.len() - 1];
        let m: usize = in_shape.iter().product::<usize>() / k;
        let n = weight.shape()[1];
        let act_contig = input.contiguous();

        let mut out_shape = in_shape[..in_shape.len() - 1].to_vec();
        out_shape.push(n);
        let output = Tensor::<WgpuRuntime>::empty(&out_shape, DType::F32, input.device());
        int4_dispatch::dispatch_marlin_gemm(
            self,
            &act_contig,
            weight,
            scales,
            zeros,
            &output,
            m as u32,
            k as u32,
            n as u32,
            group_size as u32,
        )?;
        Ok(output)
    }

    fn quant_matmul(
        &self,
        activation: &Tensor<WgpuRuntime>,
        weight: &QuantTensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        // Validate activation dtype
        if activation.dtype() != DType::F32 {
            return Err(Error::QuantError {
                reason: format!(
                    "quant_matmul activation must be F32, got {:?}",
                    activation.dtype()
                ),
            });
        }

        // Validate weight is 2D: [N, K]
        let w_shape = weight.shape();
        if w_shape.len() != 2 {
            return Err(Error::QuantError {
                reason: format!("quant_matmul weight must be 2D [N, K], got {:?}", w_shape),
            });
        }
        let n = w_shape[0];
        let k = w_shape[1];

        // Validate activation shape: [..., K]
        let a_shape = activation.shape();
        if a_shape.is_empty() {
            return Err(Error::QuantError {
                reason: "quant_matmul activation must be at least 1D".into(),
            });
        }
        let a_k = a_shape[a_shape.len() - 1];
        if a_k != k {
            return Err(Error::QuantError {
                reason: format!(
                    "quant_matmul dimension mismatch: activation K={}, weight K={}",
                    a_k, k
                ),
            });
        }

        let (shader_source, entry_point) = match weight.format() {
            QuantFormat::Q4_0 => (
                shader_gen::generate_quant_matmul_q4_0_shader(),
                "quant_matmul_q4_0",
            ),
            QuantFormat::Q8_0 => (
                shader_gen::generate_quant_matmul_q8_0_shader(),
                "quant_matmul_q8_0",
            ),
            QuantFormat::Q4K => (
                shader_gen::generate_quant_matmul_q4_k_shader(),
                "quant_matmul_q4_k",
            ),
            QuantFormat::Q6K => (
                shader_gen::generate_quant_matmul_q6_k_shader(),
                "quant_matmul_q6_k",
            ),
            other => {
                return Err(Error::UnsupportedQuantFormat {
                    format: format!("{} (WebGPU quant_matmul not implemented)", other),
                });
            }
        };

        // Compute M from activation shape
        let total_elements: usize = a_shape.iter().product();
        let m = total_elements / k;

        // Ensure activation is contiguous
        let act_contig = activation.contiguous();

        // Get buffers
        let act_buf = get_buffer(act_contig.storage().ptr()).ok_or_else(|| Error::QuantError {
            reason: "activation buffer not found in WebGPU registry".into(),
        })?;
        let weight_buf = get_buffer(weight.storage().ptr()).ok_or_else(|| Error::QuantError {
            reason: "weight buffer not found in WebGPU registry".into(),
        })?;

        // Allocate output: [..., N]
        let mut out_shape = a_shape[..a_shape.len() - 1].to_vec();
        out_shape.push(n);
        let output = Tensor::<WgpuRuntime>::empty(&out_shape, DType::F32, activation.device());
        let output_buf = get_buffer(output.storage().ptr()).ok_or_else(|| Error::QuantError {
            reason: "output buffer not found in WebGPU registry".into(),
        })?;

        // Create params buffer
        let params = MatmulParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            _pad: 0,
        };
        let params_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("quant_matmul_params"),
            size: std::mem::size_of::<MatmulParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.wgpu_queue()
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        // Create pipeline
        let cache = self.pipeline_cache();
        let module = cache.get_or_create_module(entry_point, &shader_source);
        let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
            num_storage_buffers: 3,
            num_uniform_buffers: 1,
            num_readonly_storage: 0,
        });
        let pipeline = cache.get_or_create_pipeline(entry_point, entry_point, &module, &layout);

        let bind_group =
            cache.create_bind_group(&layout, &[&act_buf, &weight_buf, &output_buf, &params_buf]);

        // Dispatch 2D: ceil(N/16) x ceil(M/16)
        let mut encoder =
            self.wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("quant_matmul"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quant_matmul"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let wg_x = (n as u32).div_ceil(16);
            let wg_y = (m as u32).div_ceil(16);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        self.wgpu_queue().submit(std::iter::once(encoder.finish()));

        Ok(output)
    }
}
