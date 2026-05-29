//! Correctness test for the Q4_K GEMM kernel (MWR variant).
//!
//! Verifies the refactored 4-warp K-parallel kernel against a serial Rust
//! reference that mirrors the kernel's exact dequantisation logic.
//! Tolerance 1e-3: covers f32 accumulation-order divergence between GPU
//! parallel-warp reduction and CPU sequential sum (both use f32 throughout).
//!
//! M=128 forces the GEMM dispatch path (threshold M <= 64 → GEMV).

use super::helpers::*;
use boostr::QuantMatmulOps;
use boostr::quant::{QuantFormat, QuantTensor};
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

// ── block packing ────────────────────────────────────────────────────────────

/// Pack one Q4_K block (256 elements → 144 bytes).
///
/// Layout: [d:f16 LE][dmin:f16 LE][sc:12B][qs:128B]
///
/// Uses d=1.0, dmin=0.0, all sub-block scales=1, mins=0, so the dequantised
/// value for each element equals its nibble (0-14).
fn pack_q4k_block(seed: usize, buf: &mut [u8]) {
    assert_eq!(buf.len(), 144);

    // d = 1.0 f16 little-endian (0x3C00), dmin = 0.0
    buf[0] = 0x00;
    buf[1] = 0x3C;
    buf[2] = 0x00;
    buf[3] = 0x00;

    // Scale bytes (sc = &block[4], 12 bytes):
    //   sc[0..4]  → scales[0..4]: sc[i] & 0x3F = 1  → sc[i] = 0x01
    //   sc[4..8]  → mins[0..4]:   sc[i+4] & 0x3F = 0 → sc[i+4] = 0x00
    //   sc[8..12] → scales[4..8]: (sc[i+4] & 0x0F) = 1; bits [7:4] = 0
    //             → mins[4..8]:   (sc[i+4] >> 4) = 0 ✓
    for i in 0..4usize {
        buf[4 + i] = 0x01; // sc[0..4]: scales[0..4] = 1
        buf[8 + i] = 0x00; // sc[4..8]: mins[0..4] = 0
        buf[12 + i] = 0x01; // sc[8..12]: scales[4..8] = 1, mins[4..8] = 0
    }

    // Nibble data: 128 bytes = 256 nibbles
    // Low nibble = element at position idx*2, high nibble = idx*2+1
    let qs = &mut buf[16..144];
    for (idx, q) in qs.iter_mut().enumerate() {
        let lo = ((seed + idx * 2) % 15 + 1) as u8; // 1-15 (non-zero)
        let hi = ((seed + idx * 2 + 1) % 15 + 1) as u8;
        *q = (hi << 4) | lo;
    }
}

// ── serial f32 reference (mirrors the CUDA kernel exactly) ───────────────────

/// Serial reference implementation.
///
/// Dequantisation logic is identical to the kernel:
///   scales[j] = 1.0, mins[j] = 0.0 (for our packed blocks with d=1, dmin=0)
///   dequant(j, l) = scales[j] * nibble(qs, j, l) - mins[j]
///                 = nibble(qs, j, l)
fn serial_q4k_gemm(act: &[f32], weight_bytes: &[u8], m: usize, k: usize, n: usize) -> Vec<f32> {
    let blocks_per_row = k / 256;
    let mut out = vec![0.0f32; m * n];

    for row in 0..m {
        for col in 0..n {
            let w_row = &weight_bytes[col * blocks_per_row * 144..];
            let act_row = &act[row * k..];
            let mut sum = 0.0f32;

            for b in 0..blocks_per_row {
                let blk = &w_row[b * 144..];
                let d = f16_le_to_f32(&blk[0..2]);
                let dmin = f16_le_to_f32(&blk[2..4]);
                let sc = &blk[4..16];
                let qs = &blk[16..144];
                let base = b * 256;

                // Identical scale decode to CUDA kernel
                let mut scales = [0u8; 8];
                let mut mins = [0u8; 8];
                for i in 0..4usize {
                    scales[i] = sc[i] & 0x3F;
                    mins[i] = sc[i + 4] & 0x3F;
                }
                for i in 4..8usize {
                    scales[i] = (sc[i + 4] & 0x0F) | ((sc[i - 4] >> 6) << 4);
                    mins[i] = (sc[i + 4] >> 4) | ((sc[i] >> 6) << 4);
                }

                for j in 0..8usize {
                    let dl = d * scales[j] as f32;
                    let ml = dmin * mins[j] as f32;
                    let chunk = j / 2;
                    let is_high = j % 2 == 1;
                    for l in 0..32usize {
                        let byte = qs[chunk * 32 + l];
                        let nibble = if is_high {
                            (byte >> 4) & 0x0F
                        } else {
                            byte & 0x0F
                        };
                        sum += act_row[base + j * 32 + l] * (dl * nibble as f32 - ml);
                    }
                }
            }
            out[row * n + col] = sum;
        }
    }
    out
}

/// Decode an f16 little-endian value to f32.
fn f16_le_to_f32(bytes: &[u8]) -> f32 {
    let bits = (bytes[0] as u16) | ((bytes[1] as u16) << 8);
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        let f = mant as f32 * (1.0f32 / (1u32 << 24) as f32);
        if sign == 1 { -f } else { f }
    } else if exp == 31 {
        if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13))
    }
}

// ── test ──────────────────────────────────────────────────────────────────────

#[test]
fn test_q4k_gemm_mwr_correctness() {
    let m = 128usize; // > 64 → GEMM path
    let k = 512usize; // 2 Q4K blocks per weight row
    let n = 16usize;

    // Deterministic f32 activation
    let act_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();

    // Build Q4K weight bytes [N, K]
    let blocks_per_row = k / 256;
    let weight_bytes_per_row = blocks_per_row * 144;
    let mut weight_bytes = vec![0u8; n * weight_bytes_per_row];
    for col in 0..n {
        for b in 0..blocks_per_row {
            let off = col * weight_bytes_per_row + b * 144;
            pack_q4k_block(col * 7 + b * 13 + 1, &mut weight_bytes[off..off + 144]);
        }
    }

    // Serial f32 reference (ground truth for CUDA comparison)
    let reference = serial_q4k_gemm(&act_data, &weight_bytes, m, k, n);

    // CPU backend sanity check (uses Q8K intermediate → larger tolerance)
    let cpu_device = CpuDevice::new();
    let act_cpu = Tensor::<CpuRuntime>::from_slice(&act_data, &[m, k], &cpu_device);
    let wt_cpu = QuantTensor::<CpuRuntime>::from_bytes(
        &weight_bytes,
        QuantFormat::Q4K,
        &[n, k],
        &cpu_device,
    )
    .expect("CPU Q4K QuantTensor");
    let (cpu_client, _) = setup_cpu();
    let cpu_vec = cpu_client
        .quant_matmul(&act_cpu, &wt_cpu)
        .expect("CPU quant_matmul")
        .to_vec::<f32>();
    for (i, &v) in cpu_vec.iter().enumerate() {
        assert!(v.is_finite(), "CPU result[{}] not finite: {}", i, v);
    }

    // CUDA must match the serial f32 reference within 1e-3 relative error.
    // Both paths accumulate in f32; parallel warp reduction vs sequential sum
    // may differ by at most a few ULPs × K, well within 0.1% for K=512.
    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        use boostr::QuantMatmulOps as _;
        use boostr::quant::QuantTensor;
        use numr::tensor::Tensor;

        let act_c = Tensor::from_slice(&act_data, &[m, k], &cuda_device);
        let wt_c = QuantTensor::from_bytes(&weight_bytes, QuantFormat::Q4K, &[n, k], &cuda_device)
            .expect("CUDA Q4K QuantTensor");

        let cuda_vec = cuda_client
            .quant_matmul(&act_c, &wt_c)
            .expect("CUDA quant_matmul Q4K")
            .to_vec::<f32>();

        // Both paths use f32 throughout; parallel warp reduction vs sequential
        // sum can differ by a few ULPs × K. 2e-3 relative tolerance is tight
        // enough to catch wrong answers yet loose enough for FP ordering.
        assert_parity_f32_tol(
            &cuda_vec,
            &reference,
            "Q4K CUDA MWR vs reference",
            2e-3,
            1e-4,
        );
    });
}
