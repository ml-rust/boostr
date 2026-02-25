//! Fused optimizer operation traits
//!
//! Single-pass parameter updates: read grad + param + state → update all in one kernel.
//! Eliminates 4-8 intermediate tensor allocations per parameter per step.

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Fused optimizer operations — single-kernel parameter updates.
///
/// Each method takes parameter, gradient, and optimizer state tensors, updating
/// them all in a single pass. This reduces memory traffic by 4-8x compared to
/// composing individual numr ops (mul_scalar, add, sqrt, div, etc.).
///
/// All tensors must be the same shape and on the same device. State tensors
/// (m, v, accum, momentum_buf) are modified in-place.
pub trait FusedOptimizerOps<R: Runtime> {
    /// Fused AdamW step: update param, m, v in a single pass.
    ///
    /// Algorithm (per element):
    /// ```text
    /// m = beta1 * m + (1 - beta1) * grad
    /// v = beta2 * v + (1 - beta2) * grad^2
    /// update = step_size * m / (sqrt(v) + eps)
    /// param = param * (1 - lr * wd) - update
    /// ```
    ///
    /// where `step_size = lr * sqrt(1 - beta2^t) / (1 - beta1^t)`.
    ///
    /// # Arguments
    /// * `param` - Parameter tensor (modified in-place)
    /// * `grad` - Gradient tensor (read-only)
    /// * `m` - First moment estimate (modified in-place)
    /// * `v` - Second moment estimate (modified in-place)
    /// * `lr` - Learning rate
    /// * `beta1` - First moment decay
    /// * `beta2` - Second moment decay
    /// * `eps` - Numerical stability constant
    /// * `wd` - Weight decay coefficient
    /// * `step_size` - Pre-computed `lr * sqrt(1 - beta2^t) / (1 - beta1^t)`
    fn fused_adamw_step(
        &self,
        param: &Tensor<R>,
        grad: &Tensor<R>,
        m: &Tensor<R>,
        v: &Tensor<R>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        wd: f64,
        step_size: f64,
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// Fused SGD step with optional momentum.
    ///
    /// Algorithm (per element):
    /// ```text
    /// grad_wd = grad + wd * param          (if wd > 0)
    /// buf = momentum * buf + (1 - dampening) * grad_wd   (if has_buf)
    /// buf = grad_wd                                       (if first step)
    /// update = grad_wd + momentum * buf     (if nesterov)
    /// update = buf                          (if standard)
    /// param = param - lr * update
    /// ```
    ///
    /// If `momentum_buf` is `None`, this is the first step (buf = grad).
    fn fused_sgd_step(
        &self,
        param: &Tensor<R>,
        grad: &Tensor<R>,
        momentum_buf: Option<&Tensor<R>>,
        lr: f64,
        momentum: f64,
        dampening: f64,
        wd: f64,
        nesterov: bool,
    ) -> Result<(Tensor<R>, Tensor<R>)>;

    /// Fused AdaGrad step: update param and accumulator in a single pass.
    ///
    /// Algorithm (per element):
    /// ```text
    /// grad_wd = grad + wd * param          (if wd > 0)
    /// accum = accum + grad_wd^2
    /// param = param - lr * grad_wd / (sqrt(accum) + eps)
    /// ```
    fn fused_adagrad_step(
        &self,
        param: &Tensor<R>,
        grad: &Tensor<R>,
        accum: &Tensor<R>,
        lr: f64,
        eps: f64,
        wd: f64,
    ) -> Result<(Tensor<R>, Tensor<R>)>;

    /// Fused LAMB step: update param, m, v in a single pass.
    ///
    /// Computes Adam-style update then scales by trust ratio.
    /// Trust ratio = min(||param|| / ||update||, max_trust_ratio).
    ///
    /// NOTE: This returns the raw update and norms so the caller can compute
    /// the trust ratio (which requires a global reduction). The caller then
    /// applies `param = param - effective_lr * update`.
    fn fused_lamb_step(
        &self,
        param: &Tensor<R>,
        grad: &Tensor<R>,
        m: &Tensor<R>,
        v: &Tensor<R>,
        beta1: f64,
        beta2: f64,
        eps: f64,
        wd: f64,
        bias_corr1: f64,
        bias_corr2: f64,
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// Fused multi-tensor AdamW: update ALL parameter groups in a single kernel launch.
    ///
    /// Instead of launching one kernel per parameter, this batches all param groups
    /// into a single dispatch. On GPU this eliminates per-parameter kernel launch
    /// overhead (~5-10μs each), which adds up significantly for models with hundreds
    /// of parameters.
    ///
    /// Each entry in `groups` is `(param, grad, m, v)` — all same shape per group.
    /// Returns `Vec<(new_param, new_m, new_v)>` in the same order.
    ///
    /// Hyperparameters are shared across all groups (same lr, betas, eps, wd, step_size).
    fn fused_multi_tensor_adamw(
        &self,
        groups: &[(&Tensor<R>, &Tensor<R>, &Tensor<R>, &Tensor<R>)],
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        wd: f64,
        step_size: f64,
    ) -> Result<Vec<(Tensor<R>, Tensor<R>, Tensor<R>)>>;
}
