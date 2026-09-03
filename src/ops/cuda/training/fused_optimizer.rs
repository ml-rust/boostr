//! CUDA implementation of FusedOptimizerOps
//!
//! In-place kernel launches for fused optimizer steps.
//! Each kernel reads param, grad, and state → updates all in a single pass.
//!
//! Trait wiring only. One launcher per optimizer lives in a sibling:
//! - `fused_optimizer_adamw.rs`: AdamW, single-parameter and multi-tensor
//! - `fused_optimizer_sgd.rs`: SGD
//! - `fused_optimizer_adagrad.rs`: Adagrad
//! - `fused_optimizer_lamb.rs`: LAMB
//! - `fused_optimizer_common.rs`: launch geometry and dtype suffix

use crate::error::Result;
use crate::ops::traits::FusedOptimizerOps;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::fused_optimizer_adagrad::fused_adagrad_step_impl;
use super::fused_optimizer_adamw::{fused_adamw_step_impl, fused_multi_tensor_adamw_impl};
use super::fused_optimizer_lamb::fused_lamb_step_impl;
use super::fused_optimizer_sgd::fused_sgd_step_impl;

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
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
        fused_adamw_step_impl(
            self, param, grad, m, v, lr, beta1, beta2, eps, wd, step_size,
        )
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
        fused_sgd_step_impl(
            self,
            param,
            grad,
            momentum_buf,
            lr,
            momentum,
            dampening,
            wd,
            nesterov,
        )
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
        fused_adagrad_step_impl(self, param, grad, accum, lr, eps, wd)
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
        fused_lamb_step_impl(
            self, param, grad, m, v, beta1, beta2, eps, wd, bias_corr1, bias_corr2,
        )
    }

    fn fused_multi_tensor_adamw(
        &self,
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
        fused_multi_tensor_adamw_impl(self, groups, lr, beta1, beta2, eps, wd, step_size)
    }
}
