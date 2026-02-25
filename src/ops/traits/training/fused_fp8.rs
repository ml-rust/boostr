//! Fused FP8 training operation traits
//!
//! Gradient unscale + clip + inf/nan detect in a single kernel, plus dynamic
//! loss scale updates. Essential for mixed-precision training (FP16/BF16/FP8).

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Fused FP8 training operations — gradient unscale/clip and dynamic loss scaling.
///
/// These ops are the core of mixed-precision training: they handle the bookkeeping
/// of loss scaling (to prevent underflow in FP16/BF16 gradients) and gradient
/// clipping (to prevent exploding gradients) in fused kernels.
#[allow(clippy::too_many_arguments)]
pub trait FusedFp8TrainingOps<R: Runtime> {
    /// Fused gradient unscale + clip + inf/nan detect.
    ///
    /// Algorithm:
    /// ```text
    /// 1. Check inf/nan → if found, return early (found_inf=true)
    /// 2. Unscale: grad / loss_scale
    /// 3. Compute L2 norm of unscaled grad
    /// 4. Clip if norm > max_norm: grad *= max_norm / norm
    /// ```
    ///
    /// # Arguments
    /// * `grad` - Gradient tensor (read-only)
    /// * `max_norm` - Maximum allowed L2 norm
    /// * `loss_scale` - Current loss scale factor
    ///
    /// # Returns
    /// `(clipped_grad, grad_norm, found_inf)` — the processed gradient,
    /// its L2 norm before clipping, and whether inf/nan was detected.
    fn fused_grad_unscale_clip(
        &self,
        grad: &Tensor<R>,
        max_norm: f64,
        loss_scale: f64,
    ) -> Result<(Tensor<R>, f64, bool)>;

    /// Update dynamic loss scale based on inf/nan history.
    ///
    /// Algorithm:
    /// ```text
    /// If found_inf:
    ///     scale *= backoff_factor
    ///     growth_tracker = 0
    /// Else:
    ///     growth_tracker += 1
    ///     If growth_tracker >= growth_interval:
    ///         scale *= 2
    ///         growth_tracker = 0
    /// ```
    ///
    /// # Arguments
    /// * `found_inf` - Whether inf/nan was found in this step
    /// * `loss_scale` - Current loss scale
    /// * `growth_tracker` - Steps since last scale increase
    /// * `growth_interval` - Steps between scale increases
    /// * `backoff_factor` - Scale multiplier on inf (typically 0.5)
    ///
    /// # Returns
    /// `(new_scale, new_growth_tracker)`
    fn dynamic_loss_scale_update(
        &self,
        found_inf: bool,
        loss_scale: f64,
        growth_tracker: i32,
        growth_interval: i32,
        backoff_factor: f64,
    ) -> Result<(f64, i32)>;
}
