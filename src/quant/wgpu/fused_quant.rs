//! WebGPU implementation of FusedQuantOps

use crate::error::{Error, Result};
use crate::quant::traits::FusedQuantOps;
use numr::dtype::DType;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::BufferUsages;

use super::int4_gemm::dispatch_int4_gemm;
use super::shaders::fused_int4_swiglu as swiglu_shader;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SwigluParams {
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
}

impl FusedQuantOps<WgpuRuntime> for WgpuClient {
    fn fused_int4_swiglu(
        &self,
        input: &Tensor<WgpuRuntime>,
        gate_qweight: &Tensor<WgpuRuntime>,
        gate_scales: &Tensor<WgpuRuntime>,
        gate_zeros: &Tensor<WgpuRuntime>,
        up_qweight: &Tensor<WgpuRuntime>,
        up_scales: &Tensor<WgpuRuntime>,
        up_zeros: &Tensor<WgpuRuntime>,
        group_size: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        if input.dtype() != DType::F32 {
            return Err(Error::QuantError {
                reason: format!(
                    "fused_int4_swiglu input must be F32, got {:?}",
                    input.dtype()
                ),
            });
        }

        let in_shape = input.shape();
        let k = in_shape[in_shape.len() - 1];
        let m: usize = in_shape.iter().product::<usize>() / k;
        let n = gate_qweight.shape()[1] * 8;

        let act_contig = input.contiguous();

        let mut out_shape = in_shape[..in_shape.len() - 1].to_vec();
        out_shape.push(n);
        let output = Tensor::<WgpuRuntime>::empty(&out_shape, DType::F32, input.device());

        let shader_source = swiglu_shader::generate_fused_int4_swiglu_shader();
        let entry_point = "fused_int4_swiglu";

        let bufs = [
            get_buffer(act_contig.storage().ptr()),
            get_buffer(gate_qweight.storage().ptr()),
            get_buffer(gate_scales.storage().ptr()),
            get_buffer(gate_zeros.storage().ptr()),
            get_buffer(up_qweight.storage().ptr()),
            get_buffer(up_scales.storage().ptr()),
            get_buffer(up_zeros.storage().ptr()),
            get_buffer(output.storage().ptr()),
        ];
        for (i, b) in bufs.iter().enumerate() {
            if b.is_none() {
                return Err(Error::QuantError {
                    reason: format!("fused_int4_swiglu buffer {} not found", i),
                });
            }
        }
        let bufs = bufs.map(|b| b.unwrap());

        let params = SwigluParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            group_size: group_size as u32,
        };
        let params_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("swiglu_params"),
            size: std::mem::size_of::<SwigluParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.wgpu_queue()
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let cache = self.pipeline_cache();
        let module = cache.get_or_create_module(entry_point, &shader_source);
        let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
            num_storage_buffers: 8,
            num_uniform_buffers: 1,
            num_readonly_storage: 0,
        });
        let pipeline = cache.get_or_create_pipeline(entry_point, entry_point, &module, &layout);
        let bind_group = cache.create_bind_group(
            &layout,
            &[
                &bufs[0],
                &bufs[1],
                &bufs[2],
                &bufs[3],
                &bufs[4],
                &bufs[5],
                &bufs[6],
                &bufs[7],
                &params_buf,
            ],
        );

        let mut encoder =
            self.wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fused_int4_swiglu"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fused_int4_swiglu"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(16), (m as u32).div_ceil(16), 1);
        }
        self.wgpu_queue().submit(std::iter::once(encoder.finish()));

        Ok(output)
    }

    fn fused_int4_qkv(
        &self,
        input: &Tensor<WgpuRuntime>,
        qweight_q: &Tensor<WgpuRuntime>,
        scales_q: &Tensor<WgpuRuntime>,
        zeros_q: &Tensor<WgpuRuntime>,
        qweight_k: &Tensor<WgpuRuntime>,
        scales_k: &Tensor<WgpuRuntime>,
        zeros_k: &Tensor<WgpuRuntime>,
        qweight_v: &Tensor<WgpuRuntime>,
        scales_v: &Tensor<WgpuRuntime>,
        zeros_v: &Tensor<WgpuRuntime>,
        group_size: usize,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        if input.dtype() != DType::F32 {
            return Err(Error::QuantError {
                reason: format!("fused_int4_qkv input must be F32, got {:?}", input.dtype()),
            });
        }

        let in_shape = input.shape();
        let k = in_shape[in_shape.len() - 1];
        let m: usize = in_shape.iter().product::<usize>() / k;
        let nq = qweight_q.shape()[1] * 8;
        let nkv = qweight_k.shape()[1] * 8;

        let act_contig = input.contiguous();

        let batch_dims = &in_shape[..in_shape.len() - 1];
        let mut q_shape = batch_dims.to_vec();
        q_shape.push(nq);
        let mut kv_shape = batch_dims.to_vec();
        kv_shape.push(nkv);

        let out_q = Tensor::<WgpuRuntime>::empty(&q_shape, DType::F32, input.device());
        let out_k = Tensor::<WgpuRuntime>::empty(&kv_shape, DType::F32, input.device());
        let out_v = Tensor::<WgpuRuntime>::empty(&kv_shape, DType::F32, input.device());

        // WebGPU has a limit of 8 storage buffers per shader stage, so we can't bind
        // all 13 buffers in a single dispatch. Instead, dispatch 3 separate int4_gemm calls.
        let gs = group_size as u32;
        let m_u32 = m as u32;
        let k_u32 = k as u32;
        dispatch_int4_gemm(
            self,
            &act_contig,
            qweight_q,
            scales_q,
            zeros_q,
            &out_q,
            m_u32,
            k_u32,
            nq as u32,
            gs,
        )?;
        dispatch_int4_gemm(
            self,
            &act_contig,
            qweight_k,
            scales_k,
            zeros_k,
            &out_k,
            m_u32,
            k_u32,
            nkv as u32,
            gs,
        )?;
        dispatch_int4_gemm(
            self,
            &act_contig,
            qweight_v,
            scales_v,
            zeros_v,
            &out_v,
            m_u32,
            k_u32,
            nkv as u32,
            gs,
        )?;

        Ok((out_q, out_k, out_v))
    }
}
