//! Batched quantized matmul: one activation against several weight matrices,
//! keeping the activation resident in L2 cache across all of them.

use rayon::prelude::*;

use super::shared::{dequant_row_f32, dot_f32, fused_dot_dispatch, fused_dot_q8k_dispatch};
use crate::quant::QuantFormat;
use crate::quant::cpu::kernels::simd;

/// Batched quantized matmul: activation \[M, K\] × multiple weight\[Ni, K\]^T → multiple output\[M, Ni\]
///
/// Processes all weight matrices together so the activation stays in L2 cache.
/// For M=1 decode with QKV (3 projections) or gate+up (2 projections), this avoids
/// re-reading the activation vector 3-5x from L3/memory.
pub fn quant_matmul_batch_f32(
    act: &[f32],
    weight_list: &[(&[u8], usize)], // (weight_bytes, n) per matrix
    outputs: &mut [&mut [f32]],
    m: usize,
    k: usize,
    format: QuantFormat,
) {
    let block_size = format.block_size();
    let block_bytes = format.block_bytes();
    let blocks_per_row = k / block_size;
    let row_bytes = blocks_per_row * block_bytes;

    let use_fused = matches!(
        format,
        QuantFormat::Q2K
            | QuantFormat::Q3K
            | QuantFormat::Q4K
            | QuantFormat::Q5K
            | QuantFormat::Q6K
    );
    let use_q8k = use_fused && k.is_multiple_of(256);

    // Pre-quantize activation rows to Q8_K if using integer path
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

    // For each activation row, compute dot products against all weight matrices.
    // This keeps the activation in L2 cache while streaming through weight data.
    //
    // We parallelize over the N dimension of each weight matrix (same as single matmul),
    // but process all matrices for each column range before moving on.

    // Find the max N across all weight matrices for chunking
    let max_n: usize = weight_list.iter().map(|&(_, n)| n).max().unwrap_or(0);
    if max_n == 0 {
        return;
    }

    let num_threads = rayon::current_num_threads();
    let target_chunks = if m == 1 { num_threads } else { num_threads * 4 };
    let chunk_size = max_n.div_ceil(target_chunks);
    let chunk_size = chunk_size.max(16);

    // Collect output pointers as usize for Send+Sync
    let output_ptrs: Vec<(usize, usize)> = outputs
        .iter()
        .zip(weight_list.iter())
        .map(|(out, &(_, n))| (out.as_ptr() as usize, n))
        .collect();
    let weight_ptrs: Vec<(usize, usize)> = weight_list
        .iter()
        .map(|&(w, n)| (w.as_ptr() as usize, n))
        .collect();

    let col_ranges: Vec<(usize, usize)> = (0..max_n)
        .step_by(chunk_size)
        .map(|start| (start, (start + chunk_size).min(max_n)))
        .collect();

    col_ranges.par_iter().for_each(|&(j_start, j_end)| {
        // For each activation row
        for i in 0..m {
            // Process all weight matrices for this activation row and column range
            for (w_idx, &(w_ptr, n)) in weight_ptrs.iter().enumerate() {
                let (out_ptr, _) = output_ptrs[w_idx];
                let out = out_ptr as *mut f32;
                let w_base = w_ptr as *const u8;

                let j_end_clamped = j_end.min(n);
                if j_start >= n {
                    continue;
                }

                if use_q8k {
                    let q8k_row = unsafe {
                        std::slice::from_raw_parts(
                            (act_q8k_ptr as *const u8).add(i * q8k_row_size),
                            q8k_row_size,
                        )
                    };
                    for j in j_start..j_end_clamped {
                        let row_data = unsafe {
                            std::slice::from_raw_parts(w_base.add(j * row_bytes), row_bytes)
                        };
                        let val = fused_dot_q8k_dispatch(q8k_row, row_data, k, format);
                        unsafe {
                            *out.add(i * n + j) = val;
                        }
                    }
                } else if use_fused {
                    let act_row = &act[i * k..(i + 1) * k];
                    for j in j_start..j_end_clamped {
                        let row_data = unsafe {
                            std::slice::from_raw_parts(w_base.add(j * row_bytes), row_bytes)
                        };
                        let val = fused_dot_dispatch(act_row, row_data, k, format);
                        unsafe {
                            *out.add(i * n + j) = val;
                        }
                    }
                } else {
                    let act_row = &act[i * k..(i + 1) * k];
                    // Scalar path with dequant buffer
                    let mut dequant_row = vec![0.0f32; k];
                    for j in j_start..j_end_clamped {
                        let row_data = unsafe {
                            std::slice::from_raw_parts(w_base.add(j * row_bytes), row_bytes)
                        };
                        dequant_row_f32(row_data, &mut dequant_row, format);
                        let val = dot_f32(act_row, &dequant_row);
                        unsafe {
                            *out.add(i * n + j) = val;
                        }
                    }
                }
            }
        }
    });
}
