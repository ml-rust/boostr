//! Fused SGD launcher (momentum, dampening, Nesterov, weight decay).

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, FUSED_SGD_MODULE};
use cudarc::driver::PushKernelArg;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::fused_optimizer_common::{kernel_suffix, launch_cfg};

/// Fused SGD step over one parameter. Returns `(param, momentum_buffer)`.
#[allow(clippy::too_many_arguments)]
pub(super) fn fused_sgd_step_impl(
    client: &CudaClient,
    param: &Tensor<CudaRuntime>,
    grad: &Tensor<CudaRuntime>,
    momentum_buf: Option<&Tensor<CudaRuntime>>,
    lr: f64,
    momentum: f64,
    dampening: f64,
    wd: f64,
    nesterov: bool,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let n: usize = param.shape().iter().product();
    let dtype = param.dtype();
    let suffix = kernel_suffix(dtype)?;
    let kernel_name = format!("fused_sgd_{}", suffix);

    let new_param = param.clone();
    let new_buf = match momentum_buf {
        Some(buf) => buf.clone(),
        None => Tensor::<CudaRuntime>::zeros(param.shape(), dtype, param.device())?,
    };
    let has_buf = momentum_buf.is_some();

    let device_index = param.device().id();
    let module = kernels::get_or_load_module(client.context(), device_index, FUSED_SGD_MODULE)?;
    let func = kernels::get_kernel_function(&module, &kernel_name)?;

    let cfg = launch_cfg(n);
    let p_ptr = new_param.ptr();
    let g_ptr = grad.ptr();
    let b_ptr = new_buf.ptr();
    let nesterov_i = if nesterov { 1i32 } else { 0i32 };
    let has_buf_i = if has_buf { 1i32 } else { 0i32 };
    let n_i32 = n as i32;
    let lr_f = lr as f32;
    let mom_f = momentum as f32;
    let damp_f = dampening as f32;
    let wd_f = wd as f32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&p_ptr);
        builder.arg(&g_ptr);
        builder.arg(&b_ptr);
        if dtype == DType::F64 {
            builder.arg(&lr);
            builder.arg(&momentum);
            builder.arg(&dampening);
            builder.arg(&wd);
        } else {
            builder.arg(&lr_f);
            builder.arg(&mom_f);
            builder.arg(&damp_f);
            builder.arg(&wd_f);
        }
        builder.arg(&nesterov_i);
        builder.arg(&has_buf_i);
        builder.arg(&n_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("fused_sgd launch failed: {:?}", e),
        })?;
    }

    Ok((new_param, new_buf))
}
