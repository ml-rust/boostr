//! WebGPU implementation of SsmKernelOps
//!
//! state_passing uses a fused WGSL shader (single dispatch vs O(nchunks) dispatches).
//! Other ops delegate to impl_generic (dominated by matmul which is already native).

use crate::error::{Error, Result};
use crate::ops::impl_generic::architecture::ssm_kernels::{
    ssd_chunk_cumsum_impl, ssd_chunk_scan_impl, ssd_chunk_state_impl,
};
use crate::ops::traits::architecture::ssm_kernels::SsmKernelOps;
use numr::dtype::DType;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::BufferUsages;

const SSD_STATE_PASSING_SHADER: &str =
    include_str!("../shaders/architecture/ssd_state_passing.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsdStatePassingParams {
    batch: u32,
    nchunks: u32,
    nheads: u32,
    headdim: u32,
    dstate: u32,
    chunk_size: u32,
    _pad0: u32,
    _pad1: u32,
}

fn validate_f32(t: &Tensor<WgpuRuntime>, op: &str) -> Result<()> {
    if t.dtype() != DType::F32 {
        return Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!("{}: WebGPU SSM requires F32, got {:?}", op, t.dtype()),
        });
    }
    Ok(())
}

#[allow(non_snake_case)]
impl SsmKernelOps<WgpuRuntime> for WgpuClient {
    fn ssd_chunk_cumsum(
        &self,
        dt: &Tensor<WgpuRuntime>,
        a: &Tensor<WgpuRuntime>,
        dt_bias: Option<&Tensor<WgpuRuntime>>,
        chunk_size: usize,
        dt_softplus: bool,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        ssd_chunk_cumsum_impl(self, dt, a, dt_bias, chunk_size, dt_softplus)
    }

    fn ssd_chunk_state(
        &self,
        x: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        dt: &Tensor<WgpuRuntime>,
        dA_cumsum: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        ssd_chunk_state_impl(self, x, b, dt, dA_cumsum)
    }

    fn ssd_state_passing(
        &self,
        states: &Tensor<WgpuRuntime>,
        dA_cumsum: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_f32(states, "ssd_state_passing")?;

        let s_shape = states.shape();
        let da_shape = dA_cumsum.shape();

        if s_shape.len() != 5 {
            return Err(Error::InvalidArgument {
                arg: "states",
                reason: format!("expected 5D, got {}D", s_shape.len()),
            });
        }

        let batch = s_shape[0];
        let nchunks = s_shape[1];
        let nheads = s_shape[2];
        let headdim = s_shape[3];
        let dstate = s_shape[4];
        let chunk_size = da_shape[3];

        if nchunks <= 1 {
            return Ok(states.clone());
        }

        // Ensure contiguous
        let states_c = states.contiguous();
        let da_c = dA_cumsum.contiguous();

        // Allocate output
        let states_out = Tensor::<WgpuRuntime>::empty(s_shape, DType::F32, states.device());

        let states_buf =
            get_buffer(states_c.storage().ptr()).ok_or_else(|| Error::KernelError {
                reason: "states buffer not found".into(),
            })?;
        let da_buf = get_buffer(da_c.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "dA_cumsum buffer not found".into(),
        })?;
        let out_buf = get_buffer(states_out.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "output buffer not found".into(),
        })?;

        let params = SsdStatePassingParams {
            batch: batch as u32,
            nchunks: nchunks as u32,
            nheads: nheads as u32,
            headdim: headdim as u32,
            dstate: dstate as u32,
            chunk_size: chunk_size as u32,
            _pad0: 0,
            _pad1: 0,
        };

        let params_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssd_state_passing_params"),
            size: std::mem::size_of::<SsdStatePassingParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.wgpu_queue()
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let cache = self.pipeline_cache();
        let module = cache.get_or_create_module("ssd_state_passing_f32", SSD_STATE_PASSING_SHADER);
        let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
            num_storage_buffers: 3,
            num_uniform_buffers: 1,
            num_readonly_storage: 2,
        });
        let pipeline = cache.get_or_create_pipeline(
            "ssd_state_passing_f32",
            "ssd_state_passing_f32",
            &module,
            &layout,
        );

        let bind_group =
            cache.create_bind_group(&layout, &[&states_buf, &da_buf, &out_buf, &params_buf]);

        let total_threads = (batch * nheads * headdim * dstate) as u32;
        let workgroup_size = 256u32;
        let num_workgroups = total_threads.div_ceil(workgroup_size);

        let mut encoder =
            self.wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ssd_state_passing"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ssd_state_passing"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }
        self.wgpu_queue().submit(std::iter::once(encoder.finish()));

        Ok(states_out)
    }

    fn ssd_chunk_scan(
        &self,
        x: &Tensor<WgpuRuntime>,
        states: &Tensor<WgpuRuntime>,
        c: &Tensor<WgpuRuntime>,
        dA_cumsum: &Tensor<WgpuRuntime>,
        d: Option<&Tensor<WgpuRuntime>>,
    ) -> Result<Tensor<WgpuRuntime>> {
        ssd_chunk_scan_impl(self, x, states, c, dA_cumsum, d)
    }
}
