//! Shared implementation of dynamic loss scale update.
//!
//! Pure scalar logic — identical for all backends. No tensors involved.

use crate::error::Result;

/// Update dynamic loss scale based on inf/nan history.
///
/// If `found_inf`: scale down by `backoff_factor`, reset tracker.
/// Otherwise: increment tracker; if it reaches `growth_interval`, double the scale.
pub fn dynamic_loss_scale_update_impl(
    found_inf: bool,
    loss_scale: f64,
    growth_tracker: i32,
    growth_interval: i32,
    backoff_factor: f64,
) -> Result<(f64, i32)> {
    if found_inf {
        Ok((loss_scale * backoff_factor, 0))
    } else {
        let new_tracker = growth_tracker + 1;
        if new_tracker >= growth_interval {
            Ok((loss_scale * 2.0, 0))
        } else {
            Ok((loss_scale, new_tracker))
        }
    }
}
