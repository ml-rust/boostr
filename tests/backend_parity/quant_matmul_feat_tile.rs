//! The narrow feature tile of the feature-major MMQ path must agree with the
//! GEMV path, and with itself across calls.
//!
//! `quant_matmul` picks the 64-feature tile when the 128-feature tiling
//! launches fewer blocks than the device has SMs. Whether a shape trips that
//! rule depends on the device, so the tile is FORCED here through the bench
//! hook and both tiles are compared against the per-row GEMV, which
//! quantizes the activation the same way. The bound is activation
//! quantization noise, far tighter than a wrong stage map or warp grid.
//!
//! The automatic path is checked too, at a shape starved on any device with
//! more than a handful of SMs, so the production rule reaches the kernel.
//!
//! Bit stability: the narrow tile's sum order is fixed per variant, so the
//! same inputs must give the same bytes call after call. Cross-tile bit
//! identity is NOT required; the two tiles sum in different orders.

use super::helpers::*;
use boostr::QuantMatmulOps;
use boostr::quant::cuda::quant_matmul::forced_tile::quant_matmul_forced_feat_tile;
use boostr::quant::cuda::quant_matmul::mmq_feat_major::FeatTile;
use boostr::quant::{QuantFormat, QuantTensor, QuantizeOps};
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// Relative error allowed between the two paths, against the row's own scale.
const TOLERANCE: f32 = 1e-4;

/// DiT projection shapes: a few tens of tokens against a small-N weight. The
/// first two are the shapes the tile rule was designed on; the third is
/// starved on any device with more than four SMs, for the automatic check.
const SHAPES: [(QuantFormat, usize, usize, usize); 3] = [
    (QuantFormat::Q4K, 22, 4096, 1024),
    (QuantFormat::Q6K, 44, 1024, 4096),
    (QuantFormat::Q5K, 22, 1024, 1024),
];

/// Quantize a deterministic `[n, k]` weight with the CPU quantizer.
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

fn activation(m: usize, k: usize) -> Vec<f32> {
    (0..m * k)
        .map(|i| ((i % 599) as f32 * 0.037).cos() * 0.5)
        .collect()
}

/// Every row of `got` against the GEMV run on that row alone.
fn assert_rows_match_gemv(
    client: &numr::runtime::cuda::CudaClient,
    device: &numr::runtime::cuda::CudaDevice,
    weight: &QuantTensor<numr::runtime::cuda::CudaRuntime>,
    act: &[f32],
    got: &[f32],
    (m, n, k): (usize, usize, usize),
    label: &str,
) {
    for row in 0..m {
        let one =
            Tensor::from_slice(&act[row * k..(row + 1) * k], &[1, k], device).expect("single row");
        let single = client
            .quant_matmul(&one, weight)
            .expect("single-row quant_matmul")
            .to_vec::<f32>();
        let scale = single.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
        for (col, &want) in single.iter().enumerate() {
            let value = got[row * n + col];
            assert!(
                value.is_finite(),
                "{label} row {row} col {col}: result is {value}"
            );
            let error = (value - want).abs() / scale;
            assert!(
                error < TOLERANCE,
                "{label} row {row} col {col}: MMQ {value} vs GEMV {want}, relative error {error}"
            );
        }
    }
}

#[test]
fn cuda_both_feature_tiles_match_the_gemv_path() {
    with_cuda_backend(|client, device| {
        for (format, m, n, k) in SHAPES {
            let bytes = weight_bytes(format, n, k);
            let weight = QuantTensor::from_bytes(&bytes, format, &[n, k], &device)
                .expect("CUDA QuantTensor");
            let act = activation(m, k);
            let act_t = Tensor::from_slice(&act, &[m, k], &device).expect("activation");

            for tile in [64u32, 128] {
                let label = format!("{} m={m} n={n} k={k} feat-tile={tile}", format.name());
                let out =
                    quant_matmul_forced_feat_tile(&client, &act_t, &weight, FeatTile::Force(tile))
                        .unwrap_or_else(|e| panic!("{label}: {e}"))
                        .to_vec::<f32>();
                assert_rows_match_gemv(&client, &device, &weight, &act, &out, (m, n, k), &label);
            }
        }
    });
}

#[test]
fn cuda_the_automatic_tile_matches_the_gemv_path_at_a_starved_shape() {
    with_cuda_backend(|client, device| {
        let (format, m, n, k) = SHAPES[2];
        let bytes = weight_bytes(format, n, k);
        let weight =
            QuantTensor::from_bytes(&bytes, format, &[n, k], &device).expect("CUDA QuantTensor");
        let act = activation(m, k);
        let act_t = Tensor::from_slice(&act, &[m, k], &device).expect("activation");
        let out = client
            .quant_matmul(&act_t, &weight)
            .expect("quant_matmul")
            .to_vec::<f32>();
        let label = format!("{} m={m} n={n} k={k} auto", format.name());
        assert_rows_match_gemv(&client, &device, &weight, &act, &out, (m, n, k), &label);
    });
}

#[test]
fn cuda_the_narrow_tile_is_bit_stable_across_calls() {
    with_cuda_backend(|client, device| {
        for (format, m, n, k) in SHAPES {
            let bytes = weight_bytes(format, n, k);
            let weight = QuantTensor::from_bytes(&bytes, format, &[n, k], &device)
                .expect("CUDA QuantTensor");
            let act = activation(m, k);
            let act_t = Tensor::from_slice(&act, &[m, k], &device).expect("activation");

            let first =
                quant_matmul_forced_feat_tile(&client, &act_t, &weight, FeatTile::Force(64))
                    .expect("first call")
                    .to_vec::<f32>();
            for call in 1..4 {
                let again =
                    quant_matmul_forced_feat_tile(&client, &act_t, &weight, FeatTile::Force(64))
                        .expect("repeat call")
                        .to_vec::<f32>();
                assert!(
                    first
                        .iter()
                        .zip(&again)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{} m={m} n={n} k={k}: narrow-tile call {call} differs from call 0",
                    format.name()
                );
            }
        }
    });
}
