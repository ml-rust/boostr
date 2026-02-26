//! CPU implementation of FusedQuantOps

use crate::error::{Error, Result};
use crate::quant::traits::FusedQuantOps;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use super::kernels::{fused_int4_qkv, fused_int4_swiglu};

impl FusedQuantOps<CpuRuntime> for CpuClient {
    fn fused_int4_swiglu(
        &self,
        input: &Tensor<CpuRuntime>,
        gate_qweight: &Tensor<CpuRuntime>,
        gate_scales: &Tensor<CpuRuntime>,
        gate_zeros: &Tensor<CpuRuntime>,
        up_qweight: &Tensor<CpuRuntime>,
        up_scales: &Tensor<CpuRuntime>,
        up_zeros: &Tensor<CpuRuntime>,
        group_size: usize,
    ) -> Result<Tensor<CpuRuntime>> {
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

        // gate_qweight [K, N/8] → N = shape[1] * 8
        let n = gate_qweight.shape()[1] * 8;

        let inp = unsafe { input.storage().as_host_slice::<f32>() };
        let gqw = unsafe { gate_qweight.storage().as_host_slice::<u32>() };
        let gsc = unsafe { gate_scales.storage().as_host_slice::<f32>() };
        let gzr = unsafe { gate_zeros.storage().as_host_slice::<f32>() };
        let uqw = unsafe { up_qweight.storage().as_host_slice::<u32>() };
        let usc = unsafe { up_scales.storage().as_host_slice::<f32>() };
        let uzr = unsafe { up_zeros.storage().as_host_slice::<f32>() };

        let mut output = vec![0.0f32; m * n];
        fused_int4_swiglu::fused_int4_swiglu_f32(
            inp,
            gqw,
            gsc,
            gzr,
            uqw,
            usc,
            uzr,
            &mut output,
            m,
            k,
            n,
            group_size,
        );

        let mut out_shape = in_shape[..in_shape.len() - 1].to_vec();
        out_shape.push(n);
        Ok(Tensor::<CpuRuntime>::from_slice(
            &output,
            &out_shape,
            input.device(),
        ))
    }

    fn fused_int4_qkv(
        &self,
        input: &Tensor<CpuRuntime>,
        qweight_q: &Tensor<CpuRuntime>,
        scales_q: &Tensor<CpuRuntime>,
        zeros_q: &Tensor<CpuRuntime>,
        qweight_k: &Tensor<CpuRuntime>,
        scales_k: &Tensor<CpuRuntime>,
        zeros_k: &Tensor<CpuRuntime>,
        qweight_v: &Tensor<CpuRuntime>,
        scales_v: &Tensor<CpuRuntime>,
        zeros_v: &Tensor<CpuRuntime>,
        group_size: usize,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
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

        let inp = unsafe { input.storage().as_host_slice::<f32>() };
        let qwq = unsafe { qweight_q.storage().as_host_slice::<u32>() };
        let scq = unsafe { scales_q.storage().as_host_slice::<f32>() };
        let zrq = unsafe { zeros_q.storage().as_host_slice::<f32>() };
        let qwk = unsafe { qweight_k.storage().as_host_slice::<u32>() };
        let sck = unsafe { scales_k.storage().as_host_slice::<f32>() };
        let zrk = unsafe { zeros_k.storage().as_host_slice::<f32>() };
        let qwv = unsafe { qweight_v.storage().as_host_slice::<u32>() };
        let scv = unsafe { scales_v.storage().as_host_slice::<f32>() };
        let zrv = unsafe { zeros_v.storage().as_host_slice::<f32>() };

        let mut out_q = vec![0.0f32; m * nq];
        let mut out_k = vec![0.0f32; m * nkv];
        let mut out_v = vec![0.0f32; m * nkv];

        fused_int4_qkv::fused_int4_qkv_f32(
            inp, qwq, scq, zrq, qwk, sck, zrk, qwv, scv, zrv, &mut out_q, &mut out_k, &mut out_v,
            m, k, nq, nkv, group_size,
        );

        let batch_dims = &in_shape[..in_shape.len() - 1];
        let mut q_shape = batch_dims.to_vec();
        q_shape.push(nq);
        let mut kv_shape = batch_dims.to_vec();
        kv_shape.push(nkv);

        let dev = input.device();
        Ok((
            Tensor::<CpuRuntime>::from_slice(&out_q, &q_shape, dev),
            Tensor::<CpuRuntime>::from_slice(&out_k, &kv_shape, dev),
            Tensor::<CpuRuntime>::from_slice(&out_v, &kv_shape, dev),
        ))
    }
}
