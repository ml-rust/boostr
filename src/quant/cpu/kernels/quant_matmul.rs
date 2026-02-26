//! CPU quantized matmul kernels
//!
//! Dequantize-and-accumulate per block row for cache efficiency.
//! Computes: activation [M, K] × weight^T → output [M, N]
//!
//! Weight is stored as [N, K] (N output rows, K input cols each), matching
//! the packing axis contract: quantization blocks run along the last (K) axis.
//! We iterate over weight rows (output columns), dequantize one row at a time,
//! and accumulate the dot product contribution.

use super::dequant;
use crate::quant::QuantFormat;

/// f32 dot product with SIMD acceleration when available.
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { super::simd::dot_f32::dot_f32_avx2_fma(a.as_ptr(), b.as_ptr(), len) };
        }
    }

    let mut sum = 0.0f32;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Quantized matmul: activation \[M, K\] × weight\[N, K\]^T → output \[M, N\]
///
/// `act`: \[M * K\] f32 values (row-major)
/// `weight_bytes`: raw quantized bytes for \[N, K\] weight matrix (blocks along K)
/// `output`: \[M * N\] f32 values (row-major)
/// `m`, `k`, `n`: matrix dimensions
pub fn quant_matmul_f32(
    act: &[f32],
    weight_bytes: &[u8],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    format: QuantFormat,
) {
    debug_assert_eq!(act.len(), m * k);
    debug_assert_eq!(output.len(), m * n);

    let block_size = format.block_size();
    let block_bytes = format.block_bytes();
    let blocks_per_row = k / block_size;
    let row_bytes = blocks_per_row * block_bytes;

    debug_assert_eq!(weight_bytes.len(), n * row_bytes);

    // Temp buffer for one dequantized weight row
    let mut dequant_row = vec![0.0f32; k];

    // For each weight row (output column)
    for j in 0..n {
        let row_start = j * row_bytes;
        let row_data = &weight_bytes[row_start..row_start + row_bytes];

        // Dequantize this row
        dequant_row_f32(row_data, &mut dequant_row, format);

        // Dot product with each activation row
        for i in 0..m {
            let act_row = &act[i * k..(i + 1) * k];
            output[i * n + j] = dot_f32(act_row, &dequant_row);
        }
    }
}

/// Dequantize a single row of quantized blocks into f32
fn dequant_row_f32(row_bytes: &[u8], output: &mut [f32], format: QuantFormat) {
    match format {
        QuantFormat::Q4_0 => dequant::dequant_q4_0(row_bytes, output),
        QuantFormat::Q8_0 => dequant::dequant_q8_0(row_bytes, output),
        QuantFormat::Q4K => dequant::dequant_q4k(row_bytes, output),
        QuantFormat::Q6K => dequant::dequant_q6k(row_bytes, output),
        _ => {
            // Zero out for unsupported formats (caller should validate before)
            output.iter_mut().for_each(|v| *v = 0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn test_quant_matmul_q4_0_identity_like() {
        // 1×32 activation × 1×32 weight (single output element)
        // activation = all 1.0, weight dequantizes to all 2.0
        // result = 32 * 1.0 * 2.0 = 64.0
        let m = 1;
        let k = 32;
        let n = 1;

        let act = vec![1.0f32; m * k];

        // Q4_0 block: scale=2.0, nibbles=0x99 → (9-8)*2.0 = 2.0
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&f16::from_f32(2.0).to_le_bytes());
        block[2..18].fill(0x99);

        let mut output = vec![0.0f32; m * n];
        quant_matmul_f32(&act, &block, &mut output, m, k, n, QuantFormat::Q4_0);

        assert!(
            (output[0] - 64.0).abs() < 0.5,
            "expected ~64.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_quant_matmul_q8_0_2x1() {
        // 2×32 activation × 1×32 weight → 2×1 output
        let m = 2;
        let k = 32;
        let n = 1;

        // Row 0: all 1.0, Row 1: all 0.5
        let mut act = vec![0.0f32; m * k];
        act[..k].fill(1.0);
        act[k..].fill(0.5);

        // Q8_0 block: scale=0.5, qs=4 → value = 4 * 0.5 = 2.0
        let mut block = [0u8; 34];
        block[0..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        block[2..34].fill(4);

        let mut output = vec![0.0f32; m * n];
        quant_matmul_f32(&act, &block, &mut output, m, k, n, QuantFormat::Q8_0);

        // Row 0: 32 * 1.0 * 2.0 = 64.0
        // Row 1: 32 * 0.5 * 2.0 = 32.0
        assert!(
            (output[0] - 64.0).abs() < 0.5,
            "expected ~64.0, got {}",
            output[0]
        );
        assert!(
            (output[1] - 32.0).abs() < 0.5,
            "expected ~32.0, got {}",
            output[1]
        );
    }

    #[test]
    fn test_quant_matmul_multiple_output_cols() {
        // 1×32 activation × 2×32 weight → 1×2 output
        let m = 1;
        let k = 32;
        let n = 2;

        let act = vec![1.0f32; m * k];

        // Weight row 0: scale=1.0, nibbles=0x99 → value=1.0, dot=32.0
        let mut block0 = [0u8; 18];
        block0[0..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block0[2..18].fill(0x99);

        // Weight row 1: scale=3.0, nibbles=0x99 → value=3.0, dot=96.0
        let mut block1 = [0u8; 18];
        block1[0..2].copy_from_slice(&f16::from_f32(3.0).to_le_bytes());
        block1[2..18].fill(0x99);

        let mut weight_bytes = Vec::new();
        weight_bytes.extend_from_slice(&block0);
        weight_bytes.extend_from_slice(&block1);

        let mut output = vec![0.0f32; m * n];
        quant_matmul_f32(&act, &weight_bytes, &mut output, m, k, n, QuantFormat::Q4_0);

        assert!(
            (output[0] - 32.0).abs() < 0.5,
            "expected ~32.0, got {}",
            output[0]
        );
        assert!(
            (output[1] - 96.0).abs() < 0.5,
            "expected ~96.0, got {}",
            output[1]
        );
    }
}
