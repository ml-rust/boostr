//! The CUDA GGUF GEMM path must agree with the CUDA GEMV path.
//!
//! `quant_matmul` dispatches by M: small M takes a per-row GEMV, large M takes
//! a tiled GEMM. Those are DIFFERENT kernels computing the same thing, and only
//! the GEMV was covered for Q6_K — a tiled Q6_K GEMM could have been wrong in
//! every batched render without a single test noticing.
//!
//! Comparing the two paths against each other needs no serial decoder per
//! format: row `r` of a batched result must equal that row computed alone.
//!
//! The two paths quantize the activation differently — the GEMV and the Q8_0
//! MMQ go through Q8_1, the Q4_K/Q6_K tiled GEMM dequantizes the weight to f32
//! and keeps the activation exact — so they agree to activation-quantization
//! error, not to the bit. The bound below is far tighter than a wrong kernel.

use super::helpers::*;
use boostr::QuantMatmulOps;
use boostr::quant::{QuantFormat, QuantTensor, QuantizeOps};
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// Batch size that forces the GEMM path on every format.
const GEMM_M: usize = 128;
/// Rows compared individually. Four is enough to catch a tiling or indexing
/// fault; comparing all 128 would run 128 extra launches for no more signal.
const SAMPLED_ROWS: [usize; 4] = [0, 1, 63, 127];

/// Relative error allowed between the two paths, against the row's own scale.
const TOLERANCE: f32 = 1e-4;

fn formats() -> [QuantFormat; 3] {
    [QuantFormat::Q8_0, QuantFormat::Q6K, QuantFormat::Q4K]
}

/// Quantize a deterministic `[n, k]` weight with the CPU quantizer, and return
/// its packed bytes.
fn weight_bytes(format: QuantFormat, n: usize, k: usize) -> Vec<u8> {
    let device = CpuDevice::new();
    let (client, _) = setup_cpu();
    let values: Vec<f32> = (0..n * k)
        .map(|i| ((i % 811) as f32 * 0.023).sin())
        .collect();
    let input = Tensor::<CpuRuntime>::from_slice(&values, &[n, k], &device).expect("weight tensor");
    client
        .quantize(&input, format)
        .expect("quantize")
        .to_bytes()
        .expect("to_bytes")
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_gguf_gemm_rows_match_the_gemv_path() {
    let (n, k) = (128usize, 512usize);

    with_cuda_backend(|client, device| {
        for format in formats() {
            let bytes = weight_bytes(format, n, k);
            let weight = QuantTensor::from_bytes(&bytes, format, &[n, k], &device)
                .expect("CUDA QuantTensor");

            let act: Vec<f32> = (0..GEMM_M * k)
                .map(|i| ((i % 599) as f32 * 0.037).cos() * 0.5)
                .collect();
            let act_t = Tensor::from_slice(&act, &[GEMM_M, k], &device).expect("activation");
            let batched = client
                .quant_matmul(&act_t, &weight)
                .expect("batched quant_matmul")
                .to_vec::<f32>();

            for &row in &SAMPLED_ROWS {
                let one = Tensor::from_slice(&act[row * k..(row + 1) * k], &[1, k], &device)
                    .expect("single row");
                let single = client
                    .quant_matmul(&one, &weight)
                    .expect("single-row quant_matmul")
                    .to_vec::<f32>();

                let scale = single.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
                for (col, &want) in single.iter().enumerate() {
                    let got = batched[row * n + col];
                    assert!(
                        got.is_finite(),
                        "{} row {row} col {col}: batched result is {got}",
                        format.name()
                    );
                    let error = (got - want).abs() / scale;
                    assert!(
                        error < TOLERANCE,
                        "{} row {row} col {col}: GEMM {got} vs GEMV {want}, \
                         relative error {error}",
                        format.name()
                    );
                }
            }
        }
    });
}
