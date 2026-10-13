//! CPU/CUDA parity for the Q8_0 quantized matmul, across the M values a
//! decode-then-prefill workload actually uses.
//!
//! Q8_0 had no parity test while Q4_K and Q6_K did, and a VoxCPM2 render on
//! CUDA looped one sentence until it hit the length cap where the same file on
//! CPU stopped correctly. A format whose GEMV path is only exercised by
//! end-to-end audio is a format whose numerical faults surface as bad speech.
//!
//! M is swept because Q8_0 dispatches THREE different kernels by M: the dp4a
//! MWR GEMV, the F32 GEMV, and the tiled GEMM. A test at one M leaves two
//! kernels unmeasured.

use super::helpers::*;
use boostr::QuantMatmulOps;
use boostr::quant::{QuantFormat, QuantTensor};
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// Elements per Q8_0 block.
const Q8_0_BLOCK: usize = 32;
/// Bytes per Q8_0 block: `d` as f16 LE, then 32 signed bytes.
const Q8_0_BYTES: usize = 34;

/// Pack one Q8_0 block: `d` = 1/16 so dequantised values stay small and exact
/// in f32, and the quants sweep the full signed range including negatives.
fn pack_q8_0_block(seed: usize, buf: &mut [u8]) {
    assert_eq!(buf.len(), Q8_0_BYTES);
    // 1/16 = 0x2C00 in binary16, exact.
    buf[0] = 0x00;
    buf[1] = 0x2C;
    for (i, q) in buf[2..].iter_mut().enumerate() {
        // -127..=127, deterministic, and never the same value down a row.
        let v = ((seed + i * 7) % 255) as i32 - 127;
        *q = v as i8 as u8;
    }
}

/// Serial f32 reference over the same bytes the kernels read.
fn serial_q8_0_matmul(act: &[f32], weight: &[u8], m: usize, k: usize, n: usize) -> Vec<f32> {
    let blocks_per_row = k / Q8_0_BLOCK;
    let row_bytes = blocks_per_row * Q8_0_BYTES;
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let w_row = &weight[col * row_bytes..];
            let act_row = &act[row * k..];
            let mut sum = 0.0f32;
            for b in 0..blocks_per_row {
                let blk = &w_row[b * Q8_0_BYTES..];
                let d = f16_le_to_f32(&blk[0..2]);
                for l in 0..Q8_0_BLOCK {
                    let q = blk[2 + l] as i8 as f32;
                    sum += act_row[b * Q8_0_BLOCK + l] * d * q;
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

/// Largest relative error between two equal-length slices, scaled by the
/// reference's own magnitude so a large dot product is not judged by absolute
/// difference.
fn max_relative_error(got: &[f32], want: &[f32]) -> (f32, usize) {
    let mut worst = 0.0f32;
    let mut at = 0usize;
    let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let e = (g - w).abs() / scale;
        if e > worst {
            worst = e;
            at = i;
        }
    }
    (worst, at)
}

/// Build a deterministic activation and weight pair for one shape.
fn fixture(m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<u8>) {
    let act: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.017).sin() * 0.5).collect();
    let row_bytes = (k / Q8_0_BLOCK) * Q8_0_BYTES;
    let mut weight = vec![0u8; n * row_bytes];
    for col in 0..n {
        for b in 0..k / Q8_0_BLOCK {
            let off = col * row_bytes + b * Q8_0_BYTES;
            pack_q8_0_block(col * 11 + b * 5 + 1, &mut weight[off..off + Q8_0_BYTES]);
        }
    }
    (act, weight)
}

/// The M values that select different kernels. 1 and 4 are decode, 32 is the
/// GEMV band, 128 crosses into the tiled GEMM.
const SWEEP_M: [usize; 4] = [1, 4, 32, 128];

/// `[N, K]` weight shapes, named after the VoxCPM2 projection they come from.
///
/// A kernel that is correct at a toy shape and wrong at a real one is the case
/// this table exists to catch: K and N here are the widths a render actually
/// runs, not round numbers chosen to keep a test fast.
const SHAPES: [(&str, usize, usize); 6] = [
    // Row length 64 = TWO Q8_0 blocks. The three VoxCPM2 `in_proj`/`cond_proj`
    // weights have exactly this shape, and they are quantized ONLY in the Q8_0
    // build: a K-quant super-block needs 256 elements per row, so Q6_K and Q4_K
    // leave them F32 and never reach this path at all.
    ("in_proj", 64, 64),
    ("toy", 32, 512),
    ("kv_proj", 256, 2048),
    ("q_proj", 2048, 2048),
    ("gate_up", 6144, 2048),
    ("down_proj", 2048, 6144),
];

#[test]
fn cpu_q8_0_matmul_matches_the_serial_reference() {
    let (label, n, k) = SHAPES[0];
    let device = CpuDevice::new();
    let (cpu_client, _) = setup_cpu();

    for &m in &SWEEP_M {
        let _ = label;
        let (act, weight) = fixture(m, k, n);
        let want = serial_q8_0_matmul(&act, &weight, m, k, n);

        let act_t = Tensor::<CpuRuntime>::from_slice(&act, &[m, k], &device).unwrap();
        let wt =
            QuantTensor::<CpuRuntime>::from_bytes(&weight, QuantFormat::Q8_0, &[n, k], &device)
                .expect("CPU Q8_0 QuantTensor");
        let got = cpu_client
            .quant_matmul(&act_t, &wt)
            .expect("CPU quant_matmul Q8_0")
            .to_vec::<f32>();

        assert_eq!(got.len(), want.len(), "m={m}: output length");
        let (err, at) = max_relative_error(&got, &want);
        // The CPU path quantises the activation to Q8_K, so it carries real
        // quantisation error against an f32 reference; 2% bounds that without
        // admitting a wrong kernel.
        assert!(
            err < 2e-2,
            "m={m}: CPU Q8_0 differs from the serial reference by {err} at index {at} \
             (got {}, want {})",
            got[at],
            want[at]
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_q8_0_matmul_matches_the_serial_reference_at_every_m() {
    with_cuda_backend(|cuda_client, cuda_device| {
        for &(label, n, k) in &SHAPES {
            for &m in &SWEEP_M {
                let (act, weight) = fixture(m, k, n);
                let want = serial_q8_0_matmul(&act, &weight, m, k, n);

                let act_t = Tensor::from_slice(&act, &[m, k], &cuda_device).unwrap();
                let wt = QuantTensor::from_bytes(&weight, QuantFormat::Q8_0, &[n, k], &cuda_device)
                    .expect("CUDA Q8_0 QuantTensor");
                let got = cuda_client
                    .quant_matmul(&act_t, &wt)
                    .expect("CUDA quant_matmul Q8_0")
                    .to_vec::<f32>();

                assert_eq!(got.len(), want.len(), "{label} m={m}: output length");
                for (i, &v) in got.iter().enumerate() {
                    assert!(v.is_finite(), "{label} m={m}: CUDA result[{i}] is {v}");
                }
                let (err, at) = max_relative_error(&got, &want);
                // The dp4a GEMV quantises the activation to Q8_1 before the dot,
                // which is a real and expected loss; 2% bounds it the same way the
                // CPU assertion does, and a wrong kernel misses by far more.
                assert!(
                    err < 2e-2,
                    "{label} [n={n} k={k}] m={m}: CUDA Q8_0 differs from the serial reference by \
                 {err} at index {at} (got {}, want {})",
                    got[at],
                    want[at]
                );
            }
        }
    });
}
