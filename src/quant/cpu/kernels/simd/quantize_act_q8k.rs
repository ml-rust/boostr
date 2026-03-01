//! AVX2 f32→Q8_K activation quantizer
//!
//! Q8_K format (292 bytes per 256 elements):
//!   [0..4]     d: f32 (scale)
//!   [4..260]   qs: [i8; 256] (quantized values)
//!   [260..292] bsums: [i16; 16] (block sums, 16 sub-blocks of 16 elements)
//!
//! Each 256-element block: find absmax, scale = absmax/127, quantize to i8.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Size of one Q8_K block in bytes
pub const Q8K_BLOCK_BYTES: usize = 292;

/// Quantize f32 activation to Q8_K format using AVX2.
///
/// `input`: f32 values, length must be multiple of 256
/// `output`: pre-allocated buffer, must have room for (input.len()/256) * 292 bytes
///
/// # Safety
/// Requires AVX2. Caller ensures buffer sizes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn quantize_f32_to_q8k_avx2(input: &[f32], output: &mut [u8]) {
    unsafe {
        let num_blocks = input.len() / 256;
        debug_assert!(output.len() >= num_blocks * Q8K_BLOCK_BYTES);

        for b in 0..num_blocks {
            let src = &input[b * 256..];
            let dst_base = output.as_mut_ptr().add(b * Q8K_BLOCK_BYTES);

            // Find absmax across 256 elements using AVX2
            let mut amax = _mm256_setzero_ps();
            let sign_mask = _mm256_set1_ps(-0.0f32);
            for i in (0..256).step_by(8) {
                let v = _mm256_loadu_ps(src.as_ptr().add(i));
                let abs_v = _mm256_andnot_ps(sign_mask, v);
                amax = _mm256_max_ps(amax, abs_v);
            }

            // Horizontal max
            let hi128 = _mm256_extractf128_ps(amax, 1);
            let lo128 = _mm256_castps256_ps128(amax);
            let max128 = _mm_max_ps(lo128, hi128);
            let max64 = _mm_movehl_ps(max128, max128);
            let max128 = _mm_max_ps(max128, max64);
            let max32 = _mm_shuffle_ps(max128, max128, 1);
            let max_val = _mm_max_ss(max128, max32);
            let amax_f32 = _mm_cvtss_f32(max_val);

            let d = amax_f32 / 127.0f32;
            let id = if amax_f32 != 0.0 {
                127.0f32 / amax_f32
            } else {
                0.0f32
            };

            // Store d as f32 at offset 0
            std::ptr::copy_nonoverlapping(d.to_le_bytes().as_ptr(), dst_base, 4);

            // Quantize 256 elements and compute bsums (16 sub-blocks of 16 elements)
            let id_vec = _mm256_set1_ps(id);
            for sub in 0..16 {
                let base = sub * 16;
                let mut bsum: i16 = 0;

                for g in 0..2 {
                    let offset = base + g * 8;
                    let v = _mm256_loadu_ps(src.as_ptr().add(offset));
                    let scaled = _mm256_mul_ps(v, id_vec);
                    let rounded =
                        _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    let clamped = _mm256_max_ps(
                        _mm256_set1_ps(-128.0),
                        _mm256_min_ps(_mm256_set1_ps(127.0), rounded),
                    );
                    let i32s = _mm256_cvtps_epi32(clamped);

                    let mut vals = [0i32; 8];
                    _mm256_storeu_si256(vals.as_mut_ptr() as *mut __m256i, i32s);
                    for j in 0..8 {
                        let q = vals[j] as i8;
                        *dst_base.add(4 + offset + j) = q as u8;
                        bsum += q as i16;
                    }
                }

                // Store bsum for this sub-block at offset 260
                let bsum_bytes = bsum.to_le_bytes();
                *dst_base.add(260 + sub * 2) = bsum_bytes[0];
                *dst_base.add(260 + sub * 2 + 1) = bsum_bytes[1];
            }
        }
    }
}

/// Dispatch wrapper
pub fn quantize_f32_to_q8k(input: &[f32], output: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { quantize_f32_to_q8k_avx2(input, output) };
        }
    }

    // Scalar fallback
    quantize_f32_to_q8k_scalar(input, output);
}

/// Scalar fallback for non-AVX2 platforms
fn quantize_f32_to_q8k_scalar(input: &[f32], output: &mut [u8]) {
    let num_blocks = input.len() / 256;

    for b in 0..num_blocks {
        let src = &input[b * 256..];
        let dst = &mut output[b * Q8K_BLOCK_BYTES..];

        let amax = src[..256].iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let d = amax / 127.0f32;
        let id = if amax != 0.0 { 127.0f32 / amax } else { 0.0f32 };

        dst[0..4].copy_from_slice(&d.to_le_bytes());

        for sub in 0..16 {
            let mut bsum: i16 = 0;
            for j in 0..16 {
                let idx = sub * 16 + j;
                let q = (src[idx] * id).round().clamp(-128.0, 127.0) as i8;
                dst[4 + idx] = q as u8;
                bsum += q as i16;
            }
            dst[260 + sub * 2..262 + sub * 2].copy_from_slice(&bsum.to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_q8k_roundtrip() {
        let k = 256;
        let input: Vec<f32> = (0..k).map(|i| (i as f32 - 128.0) * 0.1).collect();
        let mut output = vec![0u8; Q8K_BLOCK_BYTES];

        quantize_f32_to_q8k(&input, &mut output);

        let d = f32::from_le_bytes(output[0..4].try_into().unwrap());
        assert!(d > 0.0, "d should be positive, got {d}");

        // Verify bsums consistency
        for sub in 0..16 {
            let bsum = i16::from_le_bytes(output[260 + sub * 2..262 + sub * 2].try_into().unwrap());
            let manual_sum: i16 = (0..16).map(|j| output[4 + sub * 16 + j] as i8 as i16).sum();
            assert_eq!(bsum, manual_sum, "bsum mismatch at sub-block {sub}");
        }
    }

    #[test]
    fn test_quantize_q8k_zeros() {
        let input = vec![0.0f32; 256];
        let mut output = vec![0u8; Q8K_BLOCK_BYTES];
        quantize_f32_to_q8k(&input, &mut output);

        let d = f32::from_le_bytes(output[0..4].try_into().unwrap());
        assert_eq!(d, 0.0);
        assert!(output[4..260].iter().all(|&v| v == 0));
    }
}
