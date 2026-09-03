//! Fused LAMB launcher.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, FUSED_LAMB_MODULE};
use cudarc::driver::PushKernelArg;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::fused_optimizer_common::{kernel_suffix, launch_cfg};

/// Fused LAMB step over one parameter. Returns `(update, m, v)`.
#[allow(clippy::too_many_arguments)]
pub(super) fn fused_lamb_step_impl(
    client: &CudaClient,
    param: &Tensor<CudaRuntime>,
    grad: &Tensor<CudaRuntime>,
    m: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    beta1: f64,
    beta2: f64,
    eps: f64,
    wd: f64,
    bias_corr1: f64,
    bias_corr2: f64,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    let n: usize = param.shape().iter().product();
    let dtype = param.dtype();
    let suffix = kernel_suffix(dtype)?;
    let kernel_name = format!("fused_lamb_{}", suffix);

    let new_m = m.clone();
    let new_v = v.clone();
    let update = Tensor::<CudaRuntime>::zeros(param.shape(), dtype, param.device())?;

    let device_index = param.device().id();
    let module = kernels::get_or_load_module(client.context(), device_index, FUSED_LAMB_MODULE)?;
    let func = kernels::get_kernel_function(&module, &kernel_name)?;

    let cfg = launch_cfg(n);
    let p_ptr = param.ptr();
    let g_ptr = grad.ptr();
    let m_ptr = new_m.ptr();
    let v_ptr = new_v.ptr();
    let u_ptr = update.ptr();
    let n_i32 = n as i32;
    let b1_f = beta1 as f32;
    let b2_f = beta2 as f32;
    let eps_f = eps as f32;
    let wd_f = wd as f32;
    let bc1_f = bias_corr1 as f32;
    let bc2_f = bias_corr2 as f32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&p_ptr);
        builder.arg(&g_ptr);
        builder.arg(&m_ptr);
        builder.arg(&v_ptr);
        builder.arg(&u_ptr);
        if dtype == DType::F64 {
            builder.arg(&beta1);
            builder.arg(&beta2);
            builder.arg(&eps);
            builder.arg(&wd);
            builder.arg(&bias_corr1);
            builder.arg(&bias_corr2);
        } else {
            builder.arg(&b1_f);
            builder.arg(&b2_f);
            builder.arg(&eps_f);
            builder.arg(&wd_f);
            builder.arg(&bc1_f);
            builder.arg(&bc2_f);
        }
        builder.arg(&n_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("fused_lamb launch failed: {:?}", e),
        })?;
    }

    Ok((update, new_m, new_v))
}
