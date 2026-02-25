//! CPU implementation of SsmKernelOps
//!
//! state_passing uses a direct loop (avoids O(nchunks) tensor narrow/mul/add/cat overhead).
//! Other ops delegate to impl_generic (dominated by matmul which is already SIMD-native).

use crate::error::{Error, Result};
use crate::ops::impl_generic::architecture::ssm_kernels::{
    ssd_chunk_cumsum_impl, ssd_chunk_scan_impl, ssd_chunk_state_impl,
};
use crate::ops::traits::architecture::ssm_kernels::SsmKernelOps;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

#[allow(non_snake_case)]
impl SsmKernelOps<CpuRuntime> for CpuClient {
    fn ssd_chunk_cumsum(
        &self,
        dt: &Tensor<CpuRuntime>,
        a: &Tensor<CpuRuntime>,
        dt_bias: Option<&Tensor<CpuRuntime>>,
        chunk_size: usize,
        dt_softplus: bool,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        ssd_chunk_cumsum_impl(self, dt, a, dt_bias, chunk_size, dt_softplus)
    }

    fn ssd_chunk_state(
        &self,
        x: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        dt: &Tensor<CpuRuntime>,
        dA_cumsum: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        ssd_chunk_state_impl(self, x, b, dt, dA_cumsum)
    }

    fn ssd_state_passing(
        &self,
        states: &Tensor<CpuRuntime>,
        dA_cumsum: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
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

        let states_c = states.contiguous();
        let da_c = dA_cumsum.contiguous();

        let states_data = states_c.to_vec::<f32>();
        let da_data = da_c.to_vec::<f32>();
        let mut out_data = states_data.clone();

        // states layout: [batch, nchunks, nheads, headdim, dstate]
        let s_chunk_stride = nheads * headdim * dstate;
        let s_batch_stride = nchunks * s_chunk_stride;

        // dA_cumsum layout: [batch, nheads, nchunks, chunk_size]
        let da_chunk_stride = chunk_size;
        let da_head_stride = nchunks * chunk_size;
        let da_batch_stride = nheads * da_head_stride;

        for b in 0..batch {
            for h in 0..nheads {
                for d in 0..headdim {
                    for n in 0..dstate {
                        let s_base = b * s_batch_stride + h * headdim * dstate + d * dstate + n;
                        let da_base = b * da_batch_stride + h * da_head_stride + (chunk_size - 1);

                        let mut prev = out_data[s_base];

                        for c in 1..nchunks {
                            let da_val = da_data[da_base + c * da_chunk_stride];
                            let scale = da_val.min(0.0).exp();

                            let s_offset = s_base + c * s_chunk_stride;
                            prev = out_data[s_offset] + scale * prev;
                            out_data[s_offset] = prev;
                        }
                    }
                }
            }
        }

        let device = states.device();
        Ok(Tensor::<CpuRuntime>::from_slice(&out_data, s_shape, device))
    }

    fn ssd_chunk_scan(
        &self,
        x: &Tensor<CpuRuntime>,
        states: &Tensor<CpuRuntime>,
        c: &Tensor<CpuRuntime>,
        dA_cumsum: &Tensor<CpuRuntime>,
        d: Option<&Tensor<CpuRuntime>>,
    ) -> Result<Tensor<CpuRuntime>> {
        ssd_chunk_scan_impl(self, x, states, c, dA_cumsum, d)
    }
}
