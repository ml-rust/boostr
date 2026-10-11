//! `DequantOps::dequantize` and `QuantMatmulOps::quant_matmul`, CPU vs
//! WebGPU, on identical GGUF block bytes.
//!
//! Covers the formats a Llama-family GGUF file uses on WebGPU: Q4_0, Q8_0,
//! Q4_K and Q6_K. A Q4_K_M file stores most weights as Q4_K and the rest as
//! Q6_K.
//!
//! # Fixtures
//!
//! Each block gets a distinct f16 scale and an index-varying payload, as in
//! `tests/gguf_dequant_cpu_cuda_parity.rs`. A uniform payload decodes the
//! same under a permuted element order and hides a layout defect.
//!
//! # Tolerance
//!
//! Dequantization runs the same integer decode and f32 scale arithmetic on
//! both backends: `1e-6 * (1 + |x|)` per element. The matmul is checked
//! against an f64 dot product over the CPU-dequantized weight: `1e-4` of
//! the largest reference magnitude.
//!
//! Run with:
//!   cd boostr && cargo test --features wgpu --test gguf_dequant_cpu_wgpu_parity

#![cfg(feature = "wgpu")]

use std::sync::{Mutex, OnceLock};

use boostr::quant::{DequantOps, QuantFormat, QuantMatmulOps, QuantTensor};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};
use numr::tensor::Tensor;

/// Per-block f16 scale bits: 1.0, 0.5, 2.0.
const D_BITS: [u16; 3] = [0x3C00, 0x3800, 0x4000];
/// Per-block f16 min bits: 0.25, 1.0, 0.5.
const M_BITS: [u16; 3] = [0x3400, 0x3C00, 0x3800];

static WGPU_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// The WebGPU client, or `None` after one skip line.
fn wgpu_setup() -> Option<(WgpuClient, WgpuDevice)> {
    let device = WgpuDevice::new(0);
    match WgpuClient::new(device.clone()) {
        Ok(client) => Some((client, device)),
        Err(e) => {
            println!("skip: no WebGPU adapter: {e}");
            None
        }
    }
}

/// Distinct payload byte per index, shifted per block.
fn payload(i: usize, b: usize) -> u8 {
    ((i as u32)
        .wrapping_mul(31)
        .wrapping_add(b as u32 * 97)
        .wrapping_add(11)
        & 0xFF) as u8
}

/// `blocks` blocks of `format`, each with its own scale and payload.
fn fixture(format: QuantFormat, blocks: usize) -> Vec<u8> {
    let size = format.block_bytes();
    let mut data = vec![0u8; blocks * size];
    for (b, blk) in data.chunks_exact_mut(size).enumerate() {
        let d = D_BITS[b % 3].to_le_bytes();
        let m = M_BITS[b % 3].to_le_bytes();
        match format {
            QuantFormat::Q4_0 => {
                blk[0..2].copy_from_slice(&d);
                for j in 0..16 {
                    let low = ((j + b) % 16) as u8;
                    let high = ((15 - j + b) % 16) as u8;
                    blk[2 + j] = low | (high << 4);
                }
            }
            QuantFormat::Q8_0 => {
                blk[0..2].copy_from_slice(&d);
                for i in 0..32 {
                    blk[2 + i] = (i as u8)
                        .wrapping_mul(37)
                        .wrapping_add((b as u8).wrapping_mul(13));
                }
            }
            QuantFormat::Q4K => {
                blk[0..2].copy_from_slice(&d);
                blk[2..4].copy_from_slice(&m);
                for i in 0..12 {
                    blk[4 + i] = payload(i, b);
                }
                for i in 0..128 {
                    blk[16 + i] = payload(i + 12, b);
                }
            }
            QuantFormat::Q6K => {
                for (i, slot) in blk.iter_mut().take(128).enumerate() {
                    *slot = payload(i, b);
                }
                for i in 0..64 {
                    blk[128 + i] = payload(i + 128, b);
                }
                for i in 0..16 {
                    // Small signed scales keep the products in a sane range.
                    blk[192 + i] = (payload(i + 192, b) % 64) as i8 as u8;
                }
                blk[208..210].copy_from_slice(&d);
            }
            other => panic!("no fixture for {other}"),
        }
    }
    data
}

/// Dequantize `bytes` on both backends and compare every element.
fn assert_dequant_parity(format: QuantFormat, blocks: usize) {
    let Some((wgpu_client, wgpu_device)) = wgpu_setup() else {
        return;
    };
    let _lock = WGPU_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let bytes = fixture(format, blocks);
    let numel = blocks * format.block_size();

    let cpu_device = CpuDevice::new();
    let cpu_client = CpuClient::new(cpu_device.clone());
    let cpu_qt = QuantTensor::<CpuRuntime>::from_bytes(&bytes, format, &[numel], &cpu_device)
        .expect("cpu QuantTensor");
    let cpu = cpu_client
        .dequantize(&cpu_qt, DType::F32)
        .expect("cpu dequantize")
        .to_vec::<f32>();

    let wgpu_qt = QuantTensor::<WgpuRuntime>::from_bytes(&bytes, format, &[numel], &wgpu_device)
        .expect("wgpu QuantTensor");
    let gpu = wgpu_client
        .dequantize(&wgpu_qt, DType::F32)
        .expect("wgpu dequantize")
        .to_vec::<f32>();

    assert_eq!(gpu.len(), cpu.len(), "{format}: element count");
    let mismatches: Vec<String> = cpu
        .iter()
        .zip(&gpu)
        .enumerate()
        .filter(|(_, (a, b))| (*a - *b).abs() > 1e-6 * (1.0 + a.abs().max(b.abs())))
        .take(8)
        .map(|(i, (a, b))| format!("idx {i}: cpu {a} wgpu {b}"))
        .collect();
    assert!(
        mismatches.is_empty(),
        "{format}: wgpu dequantize disagrees with the CPU:\n{}",
        mismatches.join("\n")
    );
}

/// `[m, k] x [n, k]^T` through `quant_matmul` on WebGPU, against an f64
/// dot product over the CPU-dequantized weight.
///
/// The CPU `quant_matmul` quantizes the activation for K-quant formats, so
/// it is no exact reference. WebGPU dots dequantized F32 weights with F32
/// activations, the same math as the f64 reference up to rounding. The
/// activation carries one large outlier, as LLM hidden states do.
fn assert_matmul_parity(format: QuantFormat, m: usize, n: usize, k: usize) {
    let Some((wgpu_client, wgpu_device)) = wgpu_setup() else {
        return;
    };
    let _lock = WGPU_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let blocks = n * k / format.block_size();
    let bytes = fixture(format, blocks);
    let mut act: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.173).sin() * 0.5)
        .collect();
    act[7] = 300.0;

    let cpu_device = CpuDevice::new();
    let cpu_client = CpuClient::new(cpu_device.clone());
    let cpu_w = QuantTensor::<CpuRuntime>::from_bytes(&bytes, format, &[n, k], &cpu_device)
        .expect("cpu weight");
    let weight = cpu_client
        .dequantize(&cpu_w, DType::F32)
        .expect("cpu dequantize")
        .to_vec::<f32>();
    let reference: Vec<f32> = (0..m * n)
        .map(|i| {
            let (row, col) = (i / n, i % n);
            (0..k)
                .map(|j| f64::from(act[row * k + j]) * f64::from(weight[col * k + j]))
                .sum::<f64>() as f32
        })
        .collect();

    let gpu_w = QuantTensor::<WgpuRuntime>::from_bytes(&bytes, format, &[n, k], &wgpu_device)
        .expect("wgpu weight");
    let gpu_x = Tensor::<WgpuRuntime>::from_slice(&act, &[m, k], &wgpu_device).expect("wgpu x");
    let gpu = wgpu_client
        .quant_matmul(&gpu_x, &gpu_w)
        .expect("wgpu quant_matmul")
        .to_vec::<f32>();

    assert_eq!(gpu.len(), m * n, "{format}: output count");
    let peak = reference.iter().fold(0f32, |p, v| p.max(v.abs())).max(1e-6);
    let worst = reference
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        worst <= 1e-4 * peak,
        "{format} m={m}: wgpu quant_matmul is {worst} from the f64 reference (peak {peak})"
    );
}

#[test]
fn q4_0_dequant_matches_cpu() {
    assert_dequant_parity(QuantFormat::Q4_0, 3);
}

#[test]
fn q8_0_dequant_matches_cpu() {
    assert_dequant_parity(QuantFormat::Q8_0, 3);
}

#[test]
fn q4k_dequant_matches_cpu() {
    assert_dequant_parity(QuantFormat::Q4K, 3);
}

#[test]
fn q6k_dequant_matches_cpu() {
    assert_dequant_parity(QuantFormat::Q6K, 3);
}

#[test]
fn q4_0_matmul_matches_cpu() {
    for m in [1, 5, 33] {
        assert_matmul_parity(QuantFormat::Q4_0, m, 24, 512);
    }
}

#[test]
fn q8_0_matmul_matches_cpu() {
    for m in [1, 5, 33] {
        assert_matmul_parity(QuantFormat::Q8_0, m, 24, 512);
    }
}

#[test]
fn q4k_matmul_matches_cpu() {
    for m in [1, 5, 33] {
        assert_matmul_parity(QuantFormat::Q4K, m, 24, 512);
    }
}

#[test]
fn q6k_matmul_matches_cpu() {
    for m in [1, 5, 33] {
        assert_matmul_parity(QuantFormat::Q6K, m, 24, 512);
    }
}
