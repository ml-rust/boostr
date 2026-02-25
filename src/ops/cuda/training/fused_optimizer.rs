//! CUDA implementation of FusedOptimizerOps
//!
//! In-place kernel launches for fused optimizer steps.
//! Each kernel reads param, grad, and state → updates all in a single pass.

use crate::error::{Error, Result};
use crate::ops::traits::FusedOptimizerOps;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::ops::cuda::kernels::{
    self, FUSED_ADAGRAD_MODULE, FUSED_ADAMW_MODULE, FUSED_LAMB_MODULE, FUSED_SGD_MODULE,
};

fn launch_cfg(n: usize) -> LaunchConfig {
    let threads = 256u32;
    let blocks = ((n + 255) / 256) as u32;
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn kernel_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F64 => Ok("f64"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        _ => Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!("unsupported dtype {:?} for fused optimizer", dtype),
        }),
    }
}

impl FusedOptimizerOps<CudaRuntime> for CudaClient {
    fn fused_adamw_step(
        &self,
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
        let module = kernels::get_or_load_module(self.context(), device_index, FUSED_ADAMW_MODULE)?;
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
            let mut builder = self.stream().launch_builder(&func);
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

    fn fused_sgd_step(
        &self,
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
            None => Tensor::<CudaRuntime>::zeros(param.shape(), dtype, param.device()),
        };
        let has_buf = momentum_buf.is_some();

        let device_index = param.device().id();
        let module = kernels::get_or_load_module(self.context(), device_index, FUSED_SGD_MODULE)?;
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
            let mut builder = self.stream().launch_builder(&func);
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

    fn fused_adagrad_step(
        &self,
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
        let module =
            kernels::get_or_load_module(self.context(), device_index, FUSED_ADAGRAD_MODULE)?;
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
            let mut builder = self.stream().launch_builder(&func);
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

    fn fused_lamb_step(
        &self,
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
        let update = Tensor::<CudaRuntime>::zeros(param.shape(), dtype, param.device());

        let device_index = param.device().id();
        let module = kernels::get_or_load_module(self.context(), device_index, FUSED_LAMB_MODULE)?;
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
            let mut builder = self.stream().launch_builder(&func);
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
}
