//! Fused Adagrad launcher.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, FUSED_ADAGRAD_MODULE};
use cudarc::driver::PushKernelArg;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::fused_optimizer_common::{kernel_suffix, launch_cfg};

/// Fused Adagrad step over one parameter. Returns `(param, accumulator)`.
pub(super) fn fused_adagrad_step_impl(
    client: &CudaClient,
    param: &Tensor<CudaRuntime>,
    grad: &Tensor<CudaRuntime>,
    accum: &Tensor<CudaRuntime>,
    lr: f64,
    eps: f64,
    wd: f64,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let n: usize = param.shape().iter().product();
    let dtype = param.dtype();
    let suffix = kernel_suffix(dtype)?;
    let kernel_name = format!("fused_adagrad_{}", suffix);

    let new_param = param.clone();
    let new_accum = accum.clone();

    let device_index = param.device().id();
    let module = kernels::get_or_load_module(client.context(), device_index, FUSED_ADAGRAD_MODULE)?;
    let func = kernels::get_kernel_function(&module, &kernel_name)?;

    let cfg = launch_cfg(n);
    let p_ptr = new_param.ptr();
    let g_ptr = grad.ptr();
    let a_ptr = new_accum.ptr();
    let n_i32 = n as i32;
    let lr_f = lr as f32;
    let eps_f = eps as f32;
    let wd_f = wd as f32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&p_ptr);
        builder.arg(&g_ptr);
        builder.arg(&a_ptr);
        if dtype == DType::F64 {
            builder.arg(&lr);
            builder.arg(&eps);
            builder.arg(&wd);
        } else {
            builder.arg(&lr_f);
            builder.arg(&eps_f);
            builder.arg(&wd_f);
        }
        builder.arg(&n_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("fused_adagrad launch failed: {:?}", e),
        })?;
    }

    Ok((new_param, new_accum))
}
