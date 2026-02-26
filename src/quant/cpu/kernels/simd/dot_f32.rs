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
