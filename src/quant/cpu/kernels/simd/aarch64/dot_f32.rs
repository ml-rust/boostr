//! NEON f32 dot product with FMA for quantized matmul accumulation

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

const F32_LANES: usize = 4;

/// Horizontal sum of 4 f32 values in a NEON register.
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn hsum_f32_neon(v: float32x4_t) -> f32 {
    let pair = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    vget_lane_f32::<0>(vpadd_f32(pair, pair))
}

/// NEON f32 dot product of two slices using FMA.
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a` and `b` must be valid for reads of `len` f32 elements each
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot_f32_neon(a: *const f32, b: *const f32, len: usize) -> f32 {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let vb = vld1q_f32(b.add(offset));
        acc = vfmaq_f32(acc, va, vb);
    }

    let mut result = hsum_f32_neon(acc);

    for i in 0..remainder {
        let offset = chunks * F32_LANES + i;
        result += *a.add(offset) * *b.add(offset);
    }

    result
}
