//! CUDA implementation of FusedQuantOps

use crate::error::{Error, Result};
use crate::quant::traits::FusedQuantOps;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::kernels::{self, FUSED_INT4_QKV_MODULE, FUSED_INT4_SWIGLU_MODULE};

#[allow(clippy::too_many_arguments)]
impl FusedQuantOps<CudaRuntime> for CudaClient {
    fn fused_int4_swiglu(
        &self,
        input: &Tensor<CudaRuntime>,
        gate_qweight: &Tensor<CudaRuntime>,
        gate_scales: &Tensor<CudaRuntime>,
        gate_zeros: &Tensor<CudaRuntime>,
        up_qweight: &Tensor<CudaRuntime>,
        up_scales: &Tensor<CudaRuntime>,
        up_zeros: &Tensor<CudaRuntime>,
        group_size: usize,
    ) -> Result<Tensor<CudaRuntime>> {
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
        let device_index = input.device().id();

        let act_contig = input.contiguous();

        let mut out_shape = in_shape[..in_shape.len() - 1].to_vec();
        out_shape.push(n);
        let output = Tensor::<CudaRuntime>::empty(&out_shape, DType::F32, input.device());

        let module =
            kernels::get_or_load_module(self.context(), device_index, FUSED_INT4_SWIGLU_MODULE)?;
        let func = kernels::get_kernel_function(&module, "fused_int4_swiglu_f32")?;

        let (m_u32, k_u32, n_u32, gs_u32) = (m as u32, k as u32, n as u32, group_size as u32);
        let cfg = LaunchConfig {
            grid_dim: (n_u32.div_ceil(16), m_u32.div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        let act_ptr = act_contig.ptr();
        let gqw_ptr = gate_qweight.ptr();
        let gsc_ptr = gate_scales.ptr();
        let gzr_ptr = gate_zeros.ptr();
        let uqw_ptr = up_qweight.ptr();
        let usc_ptr = up_scales.ptr();
        let uzr_ptr = up_zeros.ptr();
        let out_ptr = output.ptr();

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&act_ptr);
            builder.arg(&gqw_ptr);
            builder.arg(&gsc_ptr);
            builder.arg(&gzr_ptr);
            builder.arg(&uqw_ptr);
            builder.arg(&usc_ptr);
            builder.arg(&uzr_ptr);
            builder.arg(&out_ptr);
            builder.arg(&m_u32);
            builder.arg(&k_u32);
            builder.arg(&n_u32);
            builder.arg(&gs_u32);
            builder.launch(cfg).map_err(|e| Error::QuantError {
                reason: format!("CUDA fused_int4_swiglu launch failed: {:?}", e),
            })?;
        }

        Ok(output)
    }

    fn fused_int4_qkv(
        &self,
        input: &Tensor<CudaRuntime>,
        qweight_q: &Tensor<CudaRuntime>,
        scales_q: &Tensor<CudaRuntime>,
        zeros_q: &Tensor<CudaRuntime>,
        qweight_k: &Tensor<CudaRuntime>,
        scales_k: &Tensor<CudaRuntime>,
        zeros_k: &Tensor<CudaRuntime>,
        qweight_v: &Tensor<CudaRuntime>,
        scales_v: &Tensor<CudaRuntime>,
        zeros_v: &Tensor<CudaRuntime>,
        group_size: usize,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
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
        let device_index = input.device().id();

        let act_contig = input.contiguous();

        let batch_dims = &in_shape[..in_shape.len() - 1];
        let mut q_shape = batch_dims.to_vec();
        q_shape.push(nq);
        let mut kv_shape = batch_dims.to_vec();
        kv_shape.push(nkv);

        let out_q = Tensor::<CudaRuntime>::empty(&q_shape, DType::F32, input.device());
        let out_k = Tensor::<CudaRuntime>::empty(&kv_shape, DType::F32, input.device());
        let out_v = Tensor::<CudaRuntime>::empty(&kv_shape, DType::F32, input.device());

        let module =
            kernels::get_or_load_module(self.context(), device_index, FUSED_INT4_QKV_MODULE)?;
        let func = kernels::get_kernel_function(&module, "fused_int4_qkv_f32")?;

        let max_n = nq.max(nkv) as u32;
        let (m_u32, k_u32, nq_u32, nkv_u32, gs_u32) =
            (m as u32, k as u32, nq as u32, nkv as u32, group_size as u32);

        let cfg = LaunchConfig {
            grid_dim: (max_n.div_ceil(16), m_u32.div_ceil(16), 3),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        let act_ptr = act_contig.ptr();
        let qwq_ptr = qweight_q.ptr();
        let scq_ptr = scales_q.ptr();
        let zrq_ptr = zeros_q.ptr();
        let qwk_ptr = qweight_k.ptr();
        let sck_ptr = scales_k.ptr();
        let zrk_ptr = zeros_k.ptr();
        let qwv_ptr = qweight_v.ptr();
        let scv_ptr = scales_v.ptr();
        let zrv_ptr = zeros_v.ptr();
        let oq_ptr = out_q.ptr();
        let ok_ptr = out_k.ptr();
        let ov_ptr = out_v.ptr();

        unsafe {
            let mut builder = self.stream().launch_builder(&func);
            builder.arg(&act_ptr);
            builder.arg(&qwq_ptr);
            builder.arg(&scq_ptr);
            builder.arg(&zrq_ptr);
            builder.arg(&qwk_ptr);
            builder.arg(&sck_ptr);
            builder.arg(&zrk_ptr);
            builder.arg(&qwv_ptr);
            builder.arg(&scv_ptr);
            builder.arg(&zrv_ptr);
            builder.arg(&oq_ptr);
            builder.arg(&ok_ptr);
            builder.arg(&ov_ptr);
            builder.arg(&m_u32);
            builder.arg(&k_u32);
            builder.arg(&nq_u32);
            builder.arg(&nkv_u32);
            builder.arg(&gs_u32);
            builder.launch(cfg).map_err(|e| Error::QuantError {
                reason: format!("CUDA fused_int4_qkv launch failed: {:?}", e),
            })?;
        }

        Ok((out_q, out_k, out_v))
    }
}
