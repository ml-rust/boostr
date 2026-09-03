//! Fused AdamW launchers: the per-parameter step and the multi-tensor variant.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels::{self, FUSED_ADAMW_MODULE, FUSED_MULTI_TENSOR_MODULE};
use crate::ops::traits::FusedOptimizerOps;
use cudarc::driver::PushKernelArg;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::fused_optimizer_common::{kernel_suffix, launch_cfg};

/// Fused AdamW step over one parameter. Returns `(param, m, v)`.
#[allow(clippy::too_many_arguments)]
pub(super) fn fused_adamw_step_impl(
    client: &CudaClient,
    param: &Tensor<CudaRuntime>,
    grad: &Tensor<CudaRuntime>,
    m: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    wd: f64,
    step_size: f64,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    let n: usize = param.shape().iter().product();
    let dtype = param.dtype();
    let suffix = kernel_suffix(dtype)?;
    let kernel_name = format!("fused_adamw_{}", suffix);

    // Clone param, m, v for in-place update
    let new_param = param.clone();
    let new_m = m.clone();
    let new_v = v.clone();

    let device_index = param.device().id();
    let module = kernels::get_or_load_module(client.context(), device_index, FUSED_ADAMW_MODULE)?;
    let func = kernels::get_kernel_function(&module, &kernel_name)?;

    let cfg = launch_cfg(n);
    let p_ptr = new_param.ptr();
    let g_ptr = grad.ptr();
    let m_ptr = new_m.ptr();
    let v_ptr = new_v.ptr();
    let n_i32 = n as i32;
    let lr_f = lr as f32;
    let b1_f = beta1 as f32;
    let b2_f = beta2 as f32;
    let eps_f = eps as f32;
    let wd_f = wd as f32;
    let ss_f = step_size as f32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&p_ptr);
        builder.arg(&g_ptr);
        builder.arg(&m_ptr);
        builder.arg(&v_ptr);
        if dtype == DType::F64 {
            builder.arg(&lr);
            builder.arg(&beta1);
            builder.arg(&beta2);
            builder.arg(&eps);
            builder.arg(&wd);
            builder.arg(&step_size);
        } else {
            builder.arg(&lr_f);
            builder.arg(&b1_f);
            builder.arg(&b2_f);
            builder.arg(&eps_f);
            builder.arg(&wd_f);
            builder.arg(&ss_f);
        }
        builder.arg(&n_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("fused_adamw launch failed: {:?}", e),
        })?;
    }

    Ok((new_param, new_m, new_v))
}

/// Fused AdamW step over many parameters in one launch. Returns one
/// `(param, m, v)` triple per group, in input order.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn fused_multi_tensor_adamw_impl(
    client: &CudaClient,
    groups: &[(
        &Tensor<CudaRuntime>,
        &Tensor<CudaRuntime>,
        &Tensor<CudaRuntime>,
        &Tensor<CudaRuntime>,
    )],
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    wd: f64,
    step_size: f64,
) -> Result<
    Vec<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )>,
> {
    if groups.is_empty() {
        return Ok(Vec::new());
    }

    // For non-F32 or single group, fall back to per-param launches
    let dtype = groups[0].0.dtype();
    if dtype != DType::F32 || groups.len() == 1 {
        return groups
            .iter()
            .map(|(p, g, m, v)| {
                client.fused_adamw_step(p, g, m, v, lr, beta1, beta2, eps, wd, step_size)
            })
            .collect();
    }

    let num_groups = groups.len();

    // Clone tensors for in-place update
    let mut results: Vec<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> = Vec::with_capacity(num_groups);
    let mut ptrs_host: Vec<u64> = Vec::with_capacity(num_groups * 4);
    let mut cum_sizes_host: Vec<i32> = Vec::with_capacity(num_groups + 1);
    let mut total_n: usize = 0;

    cum_sizes_host.push(0);

    for (param, grad, m, v) in groups {
        let new_param = (*param).clone();
        let new_m = (*m).clone();
        let new_v = (*v).clone();

        ptrs_host.push(new_param.ptr());
        ptrs_host.push(grad.ptr());
        ptrs_host.push(new_m.ptr());
        ptrs_host.push(new_v.ptr());

        let n: usize = param.shape().iter().product();
        total_n += n;
        cum_sizes_host.push(total_n as i32);

        results.push((new_param, new_m, new_v));
    }

    // Upload metadata to device
    let stream = client.stream_arc();
    let mut ptrs_dev = unsafe {
        stream
            .alloc::<u64>(ptrs_host.len())
            .map_err(|e| Error::KernelError {
                reason: format!("alloc ptrs buffer: {:?}", e),
            })?
    };
    stream
        .memcpy_htod(&ptrs_host, &mut ptrs_dev)
        .map_err(|e| Error::KernelError {
            reason: format!("upload ptrs: {:?}", e),
        })?;

    let mut cum_dev = unsafe {
        stream
            .alloc::<i32>(cum_sizes_host.len())
            .map_err(|e| Error::KernelError {
                reason: format!("alloc cum_sizes buffer: {:?}", e),
            })?
    };
    stream
        .memcpy_htod(&cum_sizes_host, &mut cum_dev)
        .map_err(|e| Error::KernelError {
            reason: format!("upload cum_sizes: {:?}", e),
        })?;

    let device_index = groups[0].0.device().id();
    let module =
        kernels::get_or_load_module(client.context(), device_index, FUSED_MULTI_TENSOR_MODULE)?;
    let func = kernels::get_kernel_function(&module, "fused_multi_tensor_adamw_f32")?;

    let cfg = launch_cfg(total_n);
    let num_groups_i32 = num_groups as i32;
    let lr_f = lr as f32;
    let b1_f = beta1 as f32;
    let b2_f = beta2 as f32;
    let eps_f = eps as f32;
    let wd_f = wd as f32;
    let ss_f = step_size as f32;
    let total_n_i32 = total_n as i32;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&ptrs_dev);
        builder.arg(&cum_dev);
        builder.arg(&num_groups_i32);
        builder.arg(&lr_f);
        builder.arg(&b1_f);
        builder.arg(&b2_f);
        builder.arg(&eps_f);
        builder.arg(&wd_f);
        builder.arg(&ss_f);
        builder.arg(&total_n_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("fused_multi_tensor_adamw launch failed: {:?}", e),
        })?;
    }

    Ok(results)
}
