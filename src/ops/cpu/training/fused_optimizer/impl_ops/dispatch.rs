//! `FusedOptimizerOps` dispatch: one match per dtype, delegating to the
//! per-optimizer kernel modules.

use crate::error::{Error, Result};
use crate::ops::traits::FusedOptimizerOps;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use super::super::{adagrad, adamw, lamb, sgd};

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
impl FusedOptimizerOps<CpuRuntime> for CpuClient {
    fn fused_adamw_step(
        &self,
        param: &Tensor<CpuRuntime>,
        grad: &Tensor<CpuRuntime>,
        m: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        wd: f64,
        step_size: f64,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        validate_shapes(param, grad, "fused_adamw_step")?;
        validate_shapes(param, m, "fused_adamw_step")?;
        validate_shapes(param, v, "fused_adamw_step")?;

        match param.dtype() {
            DType::F32 => {
                adamw::fused_adamw_f32(param, grad, m, v, lr, beta1, beta2, eps, wd, step_size)
            }
            DType::F64 => {
                adamw::fused_adamw_f64(param, grad, m, v, lr, beta1, beta2, eps, wd, step_size)
            }
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => {
                adamw::fused_adamw_narrow(param, grad, m, v, lr, beta1, beta2, eps, wd, step_size)
            }
            dt => Err(Error::InvalidArgument {
                arg: "dtype",
                reason: unsupported_dtype_reason("fused_adamw_step", dt),
            }),
        }
    }

    fn fused_sgd_step(
        &self,
        param: &Tensor<CpuRuntime>,
        grad: &Tensor<CpuRuntime>,
        momentum_buf: Option<&Tensor<CpuRuntime>>,
        lr: f64,
        momentum: f64,
        dampening: f64,
        wd: f64,
        nesterov: bool,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        validate_shapes(param, grad, "fused_sgd_step")?;
        if let Some(buf) = momentum_buf {
            validate_shapes(param, buf, "fused_sgd_step")?;
        }

        match param.dtype() {
            DType::F32 => sgd::fused_sgd_f32(
                param,
                grad,
                momentum_buf,
                lr,
                momentum,
                dampening,
                wd,
                nesterov,
            ),
            DType::F64 => sgd::fused_sgd_f64(
                param,
                grad,
                momentum_buf,
                lr,
                momentum,
                dampening,
                wd,
                nesterov,
            ),
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => sgd::fused_sgd_narrow(
                param,
                grad,
                momentum_buf,
                lr,
                momentum,
                dampening,
                wd,
                nesterov,
            ),
            dt => Err(Error::InvalidArgument {
                arg: "dtype",
                reason: unsupported_dtype_reason("fused_sgd_step", dt),
            }),
        }
    }

    fn fused_adagrad_step(
        &self,
        param: &Tensor<CpuRuntime>,
        grad: &Tensor<CpuRuntime>,
        accum: &Tensor<CpuRuntime>,
        lr: f64,
        eps: f64,
        wd: f64,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        validate_shapes(param, grad, "fused_adagrad_step")?;
        validate_shapes(param, accum, "fused_adagrad_step")?;

        match param.dtype() {
            DType::F32 => adagrad::fused_adagrad_f32(param, grad, accum, lr, eps, wd),
            DType::F64 => adagrad::fused_adagrad_f64(param, grad, accum, lr, eps, wd),
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => {
                adagrad::fused_adagrad_narrow(param, grad, accum, lr, eps, wd)
            }
            dt => Err(Error::InvalidArgument {
                arg: "dtype",
                reason: unsupported_dtype_reason("fused_adagrad_step", dt),
            }),
        }
    }

    fn fused_lamb_step(
        &self,
        param: &Tensor<CpuRuntime>,
        grad: &Tensor<CpuRuntime>,
        m: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
        beta1: f64,
        beta2: f64,
        eps: f64,
        wd: f64,
        bias_corr1: f64,
        bias_corr2: f64,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        validate_shapes(param, grad, "fused_lamb_step")?;
        validate_shapes(param, m, "fused_lamb_step")?;
        validate_shapes(param, v, "fused_lamb_step")?;

        match param.dtype() {
            DType::F32 => lamb::fused_lamb_f32(
                param, grad, m, v, beta1, beta2, eps, wd, bias_corr1, bias_corr2,
            ),
            DType::F64 => lamb::fused_lamb_f64(
                param, grad, m, v, beta1, beta2, eps, wd, bias_corr1, bias_corr2,
            ),
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => lamb::fused_lamb_narrow(
                param, grad, m, v, beta1, beta2, eps, wd, bias_corr1, bias_corr2,
            ),
            dt => Err(Error::InvalidArgument {
                arg: "dtype",
                reason: unsupported_dtype_reason("fused_lamb_step", dt),
            }),
        }
    }

    fn fused_multi_tensor_adamw(
        &self,
        groups: &[(
            &Tensor<CpuRuntime>,
            &Tensor<CpuRuntime>,
            &Tensor<CpuRuntime>,
            &Tensor<CpuRuntime>,
        )],
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        wd: f64,
        step_size: f64,
    ) -> Result<Vec<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)>> {
        groups
            .iter()
            .map(|(param, grad, m, v)| {
                self.fused_adamw_step(param, grad, m, v, lr, beta1, beta2, eps, wd, step_size)
            })
            .collect()
    }
}

/// Error text for a dtype no CPU fused-optimizer arm handles.
///
/// BF16 and F16 are handled, but only in a build with the `f16` feature: numr
/// cannot allocate or convert those dtypes without it. Say so, rather than
/// reporting them as plain "unsupported".
fn unsupported_dtype_reason(op: &str, dt: DType) -> String {
    match dt {
        DType::F16 | DType::BF16 => format!(
            "{}: dtype {:?} requires the `f16` feature (build boostr with --features f16)",
            op, dt
        ),
        _ => format!("{}: unsupported dtype {:?}", op, dt),
    }
}

fn validate_shapes(a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>, op: &str) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(Error::InvalidArgument {
            arg: "shape",
            reason: format!("{}: shape mismatch {:?} vs {:?}", op, a.shape(), b.shape()),
        });
    }
    Ok(())
}
