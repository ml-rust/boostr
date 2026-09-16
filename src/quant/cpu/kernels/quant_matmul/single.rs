//! Single quantized-weight matmul: activation `[M, K]` × weight `[N, K]`^T.

use rayon::prelude::*;

use super::shared::{dequant_row_f32, dot_f32, fused_dot_dispatch, fused_dot_q8k_dispatch};
use crate::quant::QuantFormat;
use crate::quant::cpu::kernels::simd;

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

    // Parallel over chunks of output columns (N dimension).
    // Each chunk processes a range of weight rows independently with its own
    // dequant buffer, avoiding false sharing by writing to disjoint output regions.
    //
    // For decode (M=1), the output is [1, N] so we split N across threads.
    // Each thread dequantizes its weight rows and computes dot products.

    // Choose chunk size based on M:
    // - M=1 (decode GEMV): one chunk per thread to minimize Rayon scheduling overhead
    // - M>1 (prefill): more chunks for better load balancing
    let num_threads = rayon::current_num_threads();
    let target_chunks = if m == 1 { num_threads } else { num_threads * 4 };
    let chunk_size = n.div_ceil(target_chunks);
    let chunk_size = chunk_size.max(16); // minimum 16 rows per chunk

    // We need to scatter results: output[i * n + j] for each (i, j).
    // To avoid synchronization, we process column ranges in parallel.
    // Each thread writes to output[i * n + j_start..j_end] for all i.
    //
    // We use a flat output slice and index into it.

    // Collect column ranges and process in parallel
    let col_ranges: Vec<(usize, usize)> = (0..n)
        .step_by(chunk_size)
        .map(|start| (start, (start + chunk_size).min(n)))
        .collect();

    // Process column chunks in parallel using index-based approach.
    // Each iteration of par_bridge processes one column range with its own dequant buffer.
    // Output is written via unsafe pointer arithmetic to disjoint column ranges.
    let output_ptr = output.as_mut_ptr() as usize; // usize is Send+Sync

    // K-quants with dedicated fused dequant+dot kernels
    let use_fused = matches!(
        format,
        QuantFormat::Q2K
            | QuantFormat::Q3K
            | QuantFormat::Q4K
            | QuantFormat::Q5K
            | QuantFormat::Q6K
    );

    // Q8_K integer dot product path — quantize activations to Q8_K then integer arithmetic.
    // Matches llama.cpp's approach for K-quant formats.
    // NOTE: Q8_0 intentionally NOT routed here (per-32 scales vs Q8_K's per-256 scales).
    //
    // This is GGUF's de facto activation contract, not a shape inference. A GGUF
    // file carries no contract; llama.cpp runs every K-quant matmul through
    // Q8_K-quantized activations, and every other GGUF reader inherits that.
    // The `k % 256` check is Q8_K's block width — a valid K-quant tensor always
    // satisfies it, so it guards a malformed shape, never chooses semantics. A
    // consumer needing exact-F32 activations must use TCF, which declares it.
    let use_q8k = use_fused && k.is_multiple_of(256);

    // Pre-quantize activation rows to Q8_K (one per M row) if using integer path
    let q8k_block_bytes = simd::quantize_act_q8k::Q8K_BLOCK_BYTES;
    let q8k_blocks_per_row = k / 256;
    let q8k_row_size = q8k_blocks_per_row * q8k_block_bytes;
    let act_q8k: Vec<u8> = if use_q8k {
        let mut buf = vec![0u8; m * q8k_row_size];
        for i in 0..m {
            let act_row = &act[i * k..(i + 1) * k];
            let q8k_row = &mut buf[i * q8k_row_size..(i + 1) * q8k_row_size];
            simd::quantize_act_q8k::quantize_f32_to_q8k(act_row, q8k_row);
        }
        buf
    } else {
        Vec::new()
    };
    let act_q8k_ptr = act_q8k.as_ptr() as usize;

    col_ranges.par_iter().for_each(|&(j_start, j_end)| {
        let out = output_ptr as *mut f32;

        if use_q8k {
            // Integer maddubs path: Q8_K activation × Q4_K/Q6_K weight
            for i in 0..m {
                let q8k_row = unsafe {
                    std::slice::from_raw_parts(
                        (act_q8k_ptr as *const u8).add(i * q8k_row_size),
                        q8k_row_size,
                    )
                };

                for j in j_start..j_end {
                    let row_data = &weight_bytes[j * row_bytes..(j + 1) * row_bytes];
                    let val = fused_dot_q8k_dispatch(q8k_row, row_data, k, format);
                    unsafe {
                        *out.add(i * n + j) = val;
                    }
                }
            }
        } else if use_fused {
            // f32 FMA fused path (fallback for non-256-aligned K)
            let cols = j_end - j_start;
            let pairs = cols / 2;
            let remainder = cols % 2;

            for i in 0..m {
                let act_row = &act[i * k..(i + 1) * k];

                // Process pairs of weight rows
                for p in 0..pairs {
                    let j0 = j_start + p * 2;
                    let j1 = j0 + 1;
                    let row_data0 = &weight_bytes[j0 * row_bytes..(j0 + 1) * row_bytes];
                    let row_data1 = &weight_bytes[j1 * row_bytes..(j1 + 1) * row_bytes];

                    let val0 = fused_dot_dispatch(act_row, row_data0, k, format);
                    let val1 = fused_dot_dispatch(act_row, row_data1, k, format);
                    unsafe {
                        *out.add(i * n + j0) = val0;
                        *out.add(i * n + j1) = val1;
                    }
                }

                // Handle odd remainder
                if remainder > 0 {
                    let j = j_end - 1;
                    let row_data = &weight_bytes[j * row_bytes..(j + 1) * row_bytes];
                    let val = fused_dot_dispatch(act_row, row_data, k, format);
                    unsafe {
                        *out.add(i * n + j) = val;
                    }
                }
            }
        } else {
            // Dequant to buffer then SIMD dot for other formats
            let mut dequant_row = vec![0.0f32; k];
            for j in j_start..j_end {
                let row_start = j * row_bytes;
                let row_data = &weight_bytes[row_start..row_start + row_bytes];

                dequant_row_f32(row_data, &mut dequant_row, format);

                for i in 0..m {
                    let act_row = &act[i * k..(i + 1) * k];
                    let val = dot_f32(act_row, &dequant_row);
                    unsafe {
                        *out.add(i * n + j) = val;
                    }
                }
            }
        }
    });
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
