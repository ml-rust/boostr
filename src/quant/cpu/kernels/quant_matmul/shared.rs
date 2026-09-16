//! Dot-product primitives and per-format dispatch shared by the single and
//! batched matmul kernels.

use crate::quant::QuantFormat;
use crate::quant::cpu::kernels::{dequant, dequant_k_quants};

/// f32 dot product with SIMD acceleration when available.
pub(super) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        let len = a.len();
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe {
                crate::quant::cpu::kernels::simd::dot_f32::dot_f32_avx2_fma(
                    a.as_ptr(),
                    b.as_ptr(),
                    len,
                )
            };
        }
        a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        crate::quant::cpu::kernels::simd::aarch64::dot_f32::dot_f32_neon(
            a.as_ptr(),
            b.as_ptr(),
            a.len(),
        )
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut sum = 0.0f32;
        for (&ai, &bi) in a.iter().zip(b.iter()) {
            sum += ai * bi;
        }
        sum
    }
}

/// Fused dot product dispatch for formats with SIMD fused kernels
pub(super) fn fused_dot_dispatch(
    act_row: &[f32],
    row_data: &[u8],
    k: usize,
    format: QuantFormat,
) -> f32 {
    use crate::quant::cpu::kernels::simd;
    match format {
        QuantFormat::Q2K => simd::fused_q2k_dot::fused_dot_q2k(act_row, row_data, k),
        QuantFormat::Q3K => simd::fused_q3k_dot::fused_dot_q3k(act_row, row_data, k),
        QuantFormat::Q4K => simd::fused_q4k_dot::fused_dot_q4k(act_row, row_data, k),
        QuantFormat::Q5K => simd::fused_q5k_dot::fused_dot_q5k(act_row, row_data, k),
        QuantFormat::Q6K => simd::fused_q6k_dot::fused_dot_q6k(act_row, row_data, k),
        _ => unreachable!(),
    }
}

/// Q8_K integer dot product dispatch (maddubs path)
pub(super) fn fused_dot_q8k_dispatch(
    act_q8k: &[u8],
    row_data: &[u8],
    k: usize,
    format: QuantFormat,
) -> f32 {
    use crate::quant::cpu::kernels::simd;
    match format {
        QuantFormat::Q2K => simd::fused_q2k_q8k_dot::fused_dot_q2k_q8k(act_q8k, row_data, k),
        QuantFormat::Q3K => simd::fused_q3k_q8k_dot::fused_dot_q3k_q8k(act_q8k, row_data, k),
        QuantFormat::Q4K => simd::fused_q4k_q8k_dot::fused_dot_q4k_q8k(act_q8k, row_data, k),
        QuantFormat::Q5K => simd::fused_q5k_q8k_dot::fused_dot_q5k_q8k(act_q8k, row_data, k),
        QuantFormat::Q6K => simd::fused_q6k_q8k_dot::fused_dot_q6k_q8k(act_q8k, row_data, k),
        _ => unreachable!(),
    }
}

/// Dequantize a single row of quantized blocks into f32
pub fn dequant_row_f32(row_bytes: &[u8], output: &mut [f32], format: QuantFormat) {
    match format {
        // Simple quants
        QuantFormat::Q4_0 => dequant::dequant_q4_0(row_bytes, output),
        QuantFormat::Q4_1 => dequant::dequant_q4_1(row_bytes, output),
        QuantFormat::Q5_0 => dequant::dequant_q5_0(row_bytes, output),
        QuantFormat::Q5_1 => dequant::dequant_q5_1(row_bytes, output),
        QuantFormat::Q8_0 => dequant::dequant_q8_0(row_bytes, output),
        QuantFormat::Q8_1 => dequant::dequant_q8_1(row_bytes, output),
        // K-quants
        QuantFormat::Q2K => dequant_k_quants::dequant_q2k(row_bytes, output),
        QuantFormat::Q3K => dequant_k_quants::dequant_q3k(row_bytes, output),
        QuantFormat::Q4K => dequant::dequant_q4k(row_bytes, output),
        QuantFormat::Q5K => dequant_k_quants::dequant_q5k(row_bytes, output),
        QuantFormat::Q6K => dequant::dequant_q6k(row_bytes, output),
        QuantFormat::Q8K => dequant_k_quants::dequant_q8k(row_bytes, output),
        // IQ/TQ formats
        QuantFormat::IQ4NL => dequant::dequant_iq4_nl(row_bytes, output),
        QuantFormat::IQ4XS => dequant::dequant_iq4_xs(row_bytes, output),
        QuantFormat::IQ2XXS => dequant::dequant_iq2_xxs(row_bytes, output),
        QuantFormat::IQ2XS => dequant::dequant_iq2_xs(row_bytes, output),
        QuantFormat::IQ2S => dequant::dequant_iq2_s(row_bytes, output),
        QuantFormat::IQ3XXS => dequant::dequant_iq3_xxs(row_bytes, output),
        QuantFormat::IQ3S => dequant::dequant_iq3_s(row_bytes, output),
        QuantFormat::IQ1S => dequant::dequant_iq1_s(row_bytes, output),
        QuantFormat::IQ1M => dequant::dequant_iq1_m(row_bytes, output),
        QuantFormat::TQ1_0 => dequant::dequant_tq1_0(row_bytes, output),
        QuantFormat::TQ2_0 => dequant::dequant_tq2_0(row_bytes, output),
    }
}
