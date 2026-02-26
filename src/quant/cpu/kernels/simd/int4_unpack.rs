//! SIMD INT4 unpacking helpers
//!
//! Unpack 8 INT4 values from a u32 into 8 f32 values using AVX2 or scalar fallback.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::int4_gemm::AWQ_SHIFTS;

/// Unpack 8 INT4 values from a u32 (AWQ bit layout) into 8 f32s.
/// Also loads scales and zeros, applies dequant: out[i] = (q[i] - zero[i]) * scale[i]
///
/// Returns the 8 dequantized f32 weight values.
///
/// # Safety
/// - CPU must support AVX2 (enforced by `#[target_feature]` at call site)
/// - `scales` and `zeros` must be valid for at least `base_col + 8` f32 elements
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn unpack_dequant_awq_avx2(
    packed: u32,
    scales: *const f32,
    zeros: *const f32,
    base_col: usize,
) -> __m256 {
    // SAFETY: AVX2 is guaranteed by `#[target_feature]` on this function.
    // AWQ_SHIFTS contains valid right-shift counts (0, 4, 8, …, 28) for 32-bit integers.
    // _mm256_loadu_ps accepts unaligned pointers; `scales` and `zeros` validity is a caller contract.
    unsafe {
        let packed_vec = _mm256_set1_epi32(packed as i32);
        let shifts = _mm256_set_epi32(
            AWQ_SHIFTS[7] as i32,
            AWQ_SHIFTS[6] as i32,
            AWQ_SHIFTS[5] as i32,
            AWQ_SHIFTS[4] as i32,
            AWQ_SHIFTS[3] as i32,
            AWQ_SHIFTS[2] as i32,
            AWQ_SHIFTS[1] as i32,
            AWQ_SHIFTS[0] as i32,
        );
        let shifted = _mm256_srlv_epi32(packed_vec, shifts);
        let mask = _mm256_set1_epi32(0xF);
        let q_i32 = _mm256_and_si256(shifted, mask);
        let q_f32 = _mm256_cvtepi32_ps(q_i32);
        let scale_vec = _mm256_loadu_ps(scales.add(base_col));
        let zero_vec = _mm256_loadu_ps(zeros.add(base_col));
        let diff = _mm256_sub_ps(q_f32, zero_vec);
        _mm256_mul_ps(diff, scale_vec)
    }
}

/// Unpack 8 INT4 values from a u32 (sequential bit layout) into 8 f32s.
/// Dequant formula: w = (q - 8) * scale + zero (Marlin format)
///
/// # Safety
/// - CPU must support AVX2 (enforced by `#[target_feature]` at call site)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn unpack_dequant_seq_avx2(packed: u32, scale: f32, zero: f32) -> __m256 {
    // All intrinsics here are pure register operations (no pointer dereference).
    // AVX2 is guaranteed by `#[target_feature]`; shift counts (28, 24, …, 0) are valid for u32.
    // `scale` and `zero` are plain f32 scalars broadcast with _mm256_set1_ps.
    let packed_vec = _mm256_set1_epi32(packed as i32);
    let shifts = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
    let shifted = _mm256_srlv_epi32(packed_vec, shifts);
    let mask = _mm256_set1_epi32(0xF);
    let q_i32 = _mm256_and_si256(shifted, mask);
    let q_f32 = _mm256_cvtepi32_ps(q_i32);
    let eight = _mm256_set1_ps(8.0);
    let diff = _mm256_sub_ps(q_f32, eight);
    let scale_vec = _mm256_set1_ps(scale);
    let zero_vec = _mm256_set1_ps(zero);
    _mm256_add_ps(_mm256_mul_ps(diff, scale_vec), zero_vec)
}
