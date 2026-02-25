//! CUDA implementation of SsmKernelOps
//!
//! state_passing uses a fused CUDA kernel (single launch vs O(nchunks) launches).
//! Other ops delegate to impl_generic (dominated by matmul which is already native).

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, SSD_STATE_PASSING_MODULE};
use crate::ops::impl_generic::architecture::ssm_kernels::{
    ssd_chunk_cumsum_impl, ssd_chunk_scan_impl, ssd_chunk_state_impl,
};
use crate::ops::traits::architecture::ssm_kernels::SsmKernelOps;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

fn state_passing_kernel_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("ssd_state_passing_f32"),
        DType::F16 => Ok("ssd_state_passing_f16"),
        DType::BF16 => Ok("ssd_state_passing_bf16"),
        _ => Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!("SSD state passing: unsupported dtype {:?}", dtype),
        }),
    }
}

#[allow(non_snake_case)]
impl SsmKernelOps<CudaRuntime> for CudaClient {
    fn ssd_chunk_cumsum(
        &self,
        dt: &Tensor<CudaRuntime>,
        a: &Tensor<CudaRuntime>,
        dt_bias: Option<&Tensor<CudaRuntime>>,
        chunk_size: usize,
        dt_softplus: bool,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        ssd_chunk_cumsum_impl(self, dt, a, dt_bias, chunk_size, dt_softplus)
    }

    fn ssd_chunk_state(
        &self,
        x: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        dt: &Tensor<CudaRuntime>,
        dA_cumsum: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        ssd_chunk_state_impl(self, x, b, dt, dA_cumsum)
    }

    fn ssd_state_passing(
        &self,
        states: &Tensor<CudaRuntime>,
        dA_cumsum: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
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

        let dtype = states.dtype();
        let kernel_name = state_passing_kernel_name(dtype)?;
        let device = states.device();

        // Ensure contiguous
        let states_c = states.contiguous();
        let da_c = dA_cumsum.contiguous();

        // Allocate output
        let states_out = Tensor::<CudaRuntime>::empty(s_shape, dtype, device);

        let device_index = device.id();
        let module =
            kernels::get_or_load_module(self.context(), device_index, SSD_STATE_PASSING_MODULE)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        let total_threads = (batch * nheads * headdim * dstate) as u32;
        let block_dim = 256u32;
        let grid_dim = (total_threads + block_dim - 1) / block_dim;

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        let states_ptr = states_c.ptr();
        let da_ptr = da_c.ptr();
        let out_ptr = states_out.ptr();
        let batch_i = batch as i32;
        let nchunks_i = nchunks as i32;
        let nheads_i = nheads as i32;
        let headdim_i = headdim as i32;
        let dstate_i = dstate as i32;
        let chunk_size_i = chunk_size as i32;

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&states_ptr);
            builder.arg(&da_ptr);
            builder.arg(&out_ptr);
            builder.arg(&batch_i);
            builder.arg(&nchunks_i);
            builder.arg(&nheads_i);
            builder.arg(&headdim_i);
            builder.arg(&dstate_i);
            builder.arg(&chunk_size_i);
            builder.launch(cfg).map_err(|e| Error::KernelError {
                reason: format!("ssd_state_passing launch failed: {:?}", e),
            })?;
        }

        Ok(states_out)
    }

    fn ssd_chunk_scan(
        &self,
        x: &Tensor<CudaRuntime>,
        states: &Tensor<CudaRuntime>,
        c: &Tensor<CudaRuntime>,
        dA_cumsum: &Tensor<CudaRuntime>,
        d: Option<&Tensor<CudaRuntime>>,
    ) -> Result<Tensor<CudaRuntime>> {
        ssd_chunk_scan_impl(self, x, states, c, dA_cumsum, d)
    }
}
