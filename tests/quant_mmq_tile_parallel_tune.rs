//! The feature-major MMQ launch pick is a per-device measurement, and both
//! launches it picks between form the same bits.
//!
//! `prefers_tile_parallel` is probed once per (device, format) and cached,
//! so two reads agree. At the probe's own geometry the split-K pair and the
//! tile-parallel grid sum the same K-range partials in the same order, so
//! their outputs are bit-identical and so is the automatic pick, whichever
//! schedule the tuner chose. Formats cover the three probe paths: PQ2_0 and
//! Q8_0 at the default feature tile (two token tiles), Q4_K at the
//! single-warp tile (the decode batch).
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test quant_mmq_tile_parallel_tune
//!   cd boostr && NUMR_CUDA_TUNE=0 cargo test --features cuda --test quant_mmq_tile_parallel_tune

#![cfg(feature = "cuda")]

use boostr::quant::cuda::quant_matmul::forced_tile::{
    mmq_prefers_tile_parallel, mmq_tile_parallel_probe_shape, quant_matmul_forced_schedule,
};
use boostr::quant::cuda::quant_matmul::mmq_feat_major::Schedule;
use boostr::quant::traits::QuantMatmulOps;
use boostr::quant::{QuantFormat, QuantTensor, QuantizeOps};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

const FORMATS: [QuantFormat; 4] = [
    QuantFormat::PQ2_0,
    QuantFormat::PTQ1_0,
    QuantFormat::Q8_0,
    QuantFormat::Q4K,
];

fn cuda() -> Option<(CudaClient, CudaDevice)> {
    let device = CudaDevice::new(0);
    let client = CudaClient::new(device.clone()).ok()?;
    Some((client, device))
}

/// Packed weight bytes for `format` at `[n, k]`. PQ2_0 and PTQ1_0 have no
/// CPU quantize kernel, so their blocks are built directly: an f16 `d` (at
/// byte 0 for PQ2_0, byte 26 for PTQ1_0) and a code run in which every bit
/// pattern is a valid block.
fn packed_weight(format: QuantFormat, n: usize, k: usize) -> Vec<u8> {
    if matches!(format, QuantFormat::PQ2_0 | QuantFormat::PTQ1_0) {
        let block_bytes = format.block_bytes();
        let d_offset = if format == QuantFormat::PTQ1_0 { 26 } else { 0 };
        let blocks = n * k / format.block_size();
        let mut out = vec![0u8; blocks * block_bytes];
        for block in 0..blocks {
            let base = block * block_bytes;
            for pos in 0..block_bytes {
                out[base + pos] = ((block * 131 + pos * 17) % 251) as u8;
            }
            let d = half::f16::from_f32(0.01 + (block as f32 * 0.003) % 0.5);
            out[base + d_offset..base + d_offset + 2].copy_from_slice(&d.to_le_bytes());
        }
        return out;
    }
    let values: Vec<f32> = (0..n * k)
        .map(|i| ((i % 977) as f32 * 0.031).sin() + ((i / 977) as f32 * 0.17).cos() * 0.25)
        .collect();
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let input = Tensor::<CpuRuntime>::from_slice(&values, &[n, k], &device).expect("weight");
    client
        .quantize(&input, format)
        .expect("quantize")
        .to_bytes()
        .expect("weight bytes")
}

/// Activation rows with no two alike.
fn activation(device: &CudaDevice, m: usize, k: usize) -> Tensor<CudaRuntime> {
    let rows: Vec<f32> = (0..m * k)
        .map(|i| {
            let r = (i / k) as f32;
            ((i % k) as f32 * 0.013 + r * 0.7).sin() * 0.4 + (r * 0.31).cos() * 0.05
        })
        .collect();
    Tensor::<CudaRuntime>::from_slice(&rows, &[m, k], device).expect("act")
}

fn assert_same_bits(what: &str, format: QuantFormat, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{} {what}: length", format.name());
    if let Some(i) = (0..got.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
        panic!(
            "{} {what}: element {i} is {:e} ({:#010x}) vs {:e} ({:#010x})",
            format.name(),
            got[i],
            got[i].to_bits(),
            want[i],
            want[i].to_bits()
        );
    }
}

#[test]
fn the_pick_is_the_same_on_two_reads() {
    let Some((client, _device)) = cuda() else {
        eprintln!("CUDA not available, skipping");
        return;
    };
    for format in FORMATS {
        let first = mmq_prefers_tile_parallel(&client, format).expect("pick");
        let second = mmq_prefers_tile_parallel(&client, format).expect("pick");
        assert_eq!(first, second, "{} pick moved between reads", format.name());
        eprintln!("{}: prefers_tile_parallel = {first}", format.name());
    }
}

#[test]
fn both_schedules_form_the_same_bits_at_the_probe_geometry() {
    let Some((client, device)) = cuda() else {
        eprintln!("CUDA not available, skipping");
        return;
    };
    for format in FORMATS {
        let (m, n, k) = mmq_tile_parallel_probe_shape(&client, format).expect("probe shape");
        let weight =
            QuantTensor::from_bytes(&packed_weight(format, n, k), format, &[n, k], &device)
                .expect("weight");
        let act = activation(&device, m, k);

        let pair = quant_matmul_forced_schedule(&client, &act, &weight, Schedule::SplitK)
            .expect("split-K pair")
            .to_vec::<f32>();
        let grid = quant_matmul_forced_schedule(&client, &act, &weight, Schedule::TileParallel)
            .expect("tile-parallel grid")
            .to_vec::<f32>();
        let auto = client
            .quant_matmul(&act, &weight)
            .expect("quant_matmul")
            .to_vec::<f32>();

        assert_eq!(pair.len(), m * n);
        assert!(
            pair.iter().all(|v| v.is_finite()),
            "{} M={m} N={n} K={k}: non-finite output",
            format.name()
        );
        assert_same_bits("tile-parallel vs split-K", format, &grid, &pair);
        assert_same_bits("automatic pick vs split-K", format, &auto, &pair);
    }
}
