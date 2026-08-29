//! SIMD f32 dot product with FMA for quantized matmul accumulation

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Horizontal sum of 8 f32 values in an AVX2 register.
///
/// # Safety
/// - CPU must support AVX2 (enforced by `#[target_feature]` at call site)
/// - `v` must be a valid `__m256` value produced by AVX2 intrinsics
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn hsum_f32_avx2(v: __m256) -> f32 {
    // All intrinsics here are pure register operations (no pointer dereference).
    // AVX2 is guaranteed by `#[target_feature]`; SSE2 ops (_mm_*) are always safe on x86_64.
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 0b_00_00_00_01);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}

/// SIMD f32 dot product of two slices using FMA.
///
/// # Safety
/// - CPU must support AVX2 + FMA (enforced by `#[target_feature]` at call site)
/// - `a` and `b` must be valid for reads of `len` f32 elements each
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_f32_avx2_fma(a: *const f32, b: *const f32, len: usize) -> f32 {
    // SAFETY: AVX2 + FMA guaranteed by `#[target_feature]` on this function.
    // _mm256_loadu_ps accepts unaligned pointers; loop offsets stay within [0, chunks*LANES) ⊆ [0, len).
    // Scalar tail reads are in [chunks*LANES, len); pointer arithmetic is bounded by `len` (caller contract).
    unsafe {
        const LANES: usize = 8;
        let chunks = len / LANES;
        let remainder = len % LANES;

        let mut acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * LANES;
            let va = _mm256_loadu_ps(a.add(offset));
            let vb = _mm256_loadu_ps(b.add(offset));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }

        // `acc` is a valid __m256 (zeroed or accumulated from valid loads above).
        let mut result = hsum_f32_avx2(acc);

        for i in 0..remainder {
            let offset = chunks * LANES + i;
            result += *a.add(offset) * *b.add(offset);
        }

        result
    }
}

/// An f32 dot product over two equal-length windows, resolved once and then
/// called per tile.
///
/// A fused matmul calls the dot once per tile per activation row, so probing
/// CPU features inside it would repeat an atomic load every 64 elements.
/// [`select_dot_f32`] pays that once per matmul instead.
pub type DotF32 = fn(&[f32], &[f32]) -> f32;

/// The fastest f32 dot this CPU supports.
///
/// Every returned implementation accumulates in vector lanes, so the sum is
/// NOT in the scalar left-to-right order and NOT bit-identical to it. A
/// caller that needs bit-identity must not use this.
pub fn select_dot_f32() -> DotF32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return dot_f32_avx2_fma_slices;
        }
        dot_f32_scalar
    }

    #[cfg(target_arch = "aarch64")]
    {
        dot_f32_neon_slices
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        dot_f32_scalar
    }
}

/// [`dot_f32_avx2_fma`] over slices, so the pointer contract is discharged
/// by the slice lengths.
#[cfg(target_arch = "x86_64")]
fn dot_f32_avx2_fma_slices(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    // SAFETY: AVX2 and FMA were checked by `select_dot_f32`, which is the
    // only producer of this function pointer. `len` is bounded by both
    // slice lengths, so both are valid for `len` f32 reads.
    unsafe { dot_f32_avx2_fma(a.as_ptr(), b.as_ptr(), len) }
}

/// [`dot_f32_neon`] over slices, under the same length rule.
#[cfg(target_arch = "aarch64")]
fn dot_f32_neon_slices(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    // SAFETY: NEON is architectural on AArch64. `len` is bounded by both
    // slice lengths, so both are valid for `len` f32 reads.
    unsafe { super::aarch64::dot_f32::dot_f32_neon(a.as_ptr(), b.as_ptr(), len) }
}

/// Left-to-right f32 dot product. Correct on every machine.
pub fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The selected dot tracks the scalar one within f32 accumulation
    /// error. It is NOT asserted bit-identical: lane accumulation reorders
    /// the sum, which is exactly why this tolerance exists.
    #[test]
    fn the_selected_dot_tracks_the_scalar_dot() {
        for len in [0usize, 1, 7, 8, 63, 64, 257] {
            let a: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.017).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| ((i as f32) * 0.031).cos() * 1.5).collect();
            let got = select_dot_f32()(&a, &b);
            let want = dot_f32_scalar(&a, &b);
            assert!(
                (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                "len {len}: selected {got}, scalar {want}"
            );
        }
    }

    /// A shorter operand bounds the dot, so no read runs past either slice.
    #[test]
    fn the_shorter_operand_bounds_the_dot() {
        let a = vec![1.0f32; 40];
        let b = vec![2.0f32; 12];
        assert!((select_dot_f32()(&a, &b) - 24.0).abs() < 1e-5);
        assert!((select_dot_f32()(&b, &a) - 24.0).abs() < 1e-5);
    }
}
