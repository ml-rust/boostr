//! Optimizer arithmetic precision: master weights for narrow parameter dtypes.
//!
//! AdamW's update is normalized, so `|delta_w| ~= lr` regardless of gradient
//! magnitude. At a fine-tuning `lr` of 2e-5 against a pretrained weight around
//! 0.02, the relative step is 1e-3 — below BF16's 2^-8 (~0.4%) relative
//! resolution. Rounding `w + delta_w` back to BF16 therefore returns exactly
//! `w`, every step, for every parameter: the run reports a healthy loss and
//! never trains. The second moment `v` sums SQUARED gradients, which flushes
//! small gradients toward zero in BF16 independently of that.
//!
//! The fix is the standard one: the optimizer keeps an F32 master copy of every
//! parameter narrower than F32, updates the master, and writes a cast of it
//! back into the parameter the model computes with.

use numr::dtype::DType;

/// Dtype the optimizer keeps its state (and master weights) in for a parameter
/// of `param_dtype`.
///
/// Float dtypes narrower than F32 widen to F32. F32, F64, and every other dtype
/// are returned unchanged, so their arithmetic is bit-identical to a build
/// without master weights and they allocate nothing extra.
pub fn optimizer_state_dtype(param_dtype: DType) -> DType {
    match param_dtype {
        DType::F16 | DType::BF16 | DType::FP8E4M3 | DType::FP8E5M2 => DType::F32,
        other => other,
    }
}

/// True when `param_dtype` needs an F32 master copy held by the optimizer.
pub fn needs_master_weight(param_dtype: DType) -> bool {
    optimizer_state_dtype(param_dtype) != param_dtype
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_and_f64_are_untouched() {
        assert_eq!(optimizer_state_dtype(DType::F32), DType::F32);
        assert_eq!(optimizer_state_dtype(DType::F64), DType::F64);
        assert!(!needs_master_weight(DType::F32));
        assert!(!needs_master_weight(DType::F64));
    }

    #[test]
    fn test_narrow_floats_widen_to_f32() {
        for dt in [DType::BF16, DType::F16, DType::FP8E4M3, DType::FP8E5M2] {
            assert_eq!(optimizer_state_dtype(dt), DType::F32, "{dt:?}");
            assert!(needs_master_weight(dt), "{dt:?}");
        }
    }
}
