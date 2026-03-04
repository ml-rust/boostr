//! Tests for CPU QuantMatmulOps implementation.

use crate::quant::QuantTensor;
use crate::quant::format::QuantFormat;
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use half::f16;
use numr::dtype::DType;
use numr::ops::MatmulOps;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

fn setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

#[test]
fn test_quant_matmul_q4_0_basic() {
    let (client, device) = setup();

    // activation [1, 32], weight [1, 32] → output [1, 1]
    let act = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 32], &[1, 32], &device);

    let mut block = [0u8; 18];
    block[0..2].copy_from_slice(&f16::from_f32(2.0).to_le_bytes());
    block[2..18].fill(0x99); // dequant value = 2.0

    let qt = QuantTensor::<CpuRuntime>::from_bytes(&block, QuantFormat::Q4_0, &[1, 32], &device)
        .unwrap();

    let result = client.quant_matmul(&act, &qt).unwrap();
    assert_eq!(result.shape(), &[1, 1]);

    let data = result.to_vec::<f32>();
    assert!(
        (data[0] - 64.0).abs() < 0.5,
        "expected ~64.0, got {}",
        data[0]
    );
}

#[test]
fn test_quant_matmul_matches_dequant_matmul() {
    let (client, device) = setup();

    // activation [2, 32]
    let act_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
    let act = Tensor::<CpuRuntime>::from_slice(&act_data, &[2, 32], &device);

    // weight [3, 32] as Q8_0 (3 rows, each 34 bytes)
    let mut weight_bytes = Vec::new();
    for row in 0..3 {
        let mut block = [0u8; 34];
        block[0..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        // Different qs per row for variety
        block[2..34].fill((row + 1) as u8);
        weight_bytes.extend_from_slice(&block);
    }

    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(&weight_bytes, QuantFormat::Q8_0, &[3, 32], &device)
            .unwrap();

    // Method 1: quant_matmul
    let result_qm = client.quant_matmul(&act, &qt).unwrap();

    // Method 2: dequant then matmul
    let dequant_w = client.dequantize(&qt, DType::F32).unwrap();
    // dequant gives [3, 32], matmul needs act [2, 32] × w^T [32, 3]
    // Our quant_matmul does act × w^T layout, so we need to transpose for standard matmul
    let dequant_w_t = dequant_w.transpose(0isize, 1isize).unwrap();
    let result_dm = MatmulOps::matmul(&client, &act, &dequant_w_t).unwrap();

    assert_eq!(result_qm.shape(), result_dm.shape());

    let qm_data = result_qm.to_vec::<f32>();
    let dm_data = result_dm.to_vec::<f32>();
    for (i, (&a, &b)) in qm_data.iter().zip(dm_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-2,
            "mismatch at index {}: quant_matmul={}, dequant+matmul={}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_quant_matmul_dim_mismatch() {
    let (client, device) = setup();

    let act = Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; 64], &[2, 32], &device);

    // Weight K=64 ≠ activation K=32
    let block = vec![0u8; 2 * 34]; // 2 blocks of Q8_0 = 64 elements
    let qt = QuantTensor::<CpuRuntime>::from_bytes(&block, QuantFormat::Q8_0, &[1, 64], &device)
        .unwrap();

    let result = client.quant_matmul(&act, &qt);
    assert!(result.is_err());
}

#[test]
fn test_quant_matmul_q2k_basic() {
    let (client, device) = setup();

    let act = Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; 256], &[1, 256], &device);

    // Q2K: 256 elements, 84 bytes/block — all zeros dequantizes to zeros
    let block = vec![0u8; 84];
    let qt = QuantTensor::<CpuRuntime>::from_bytes(&block, QuantFormat::Q2K, &[1, 256], &device)
        .unwrap();

    let result = client.quant_matmul(&act, &qt).unwrap();
    assert_eq!(result.shape(), &[1, 1]);
    let data = result.to_vec::<f32>();
    assert!(data[0].abs() < 1e-5, "expected ~0.0, got {}", data[0]);
}

#[test]
#[allow(clippy::needless_range_loop)]
fn test_quant_matmul_q2k_matches_dequant_matmul() {
    let (client, device) = setup();

    // K=512 (2 Q2K blocks), N=2, M=2
    let k = 512;
    let n = 2;
    let m = 2;

    let act_data: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
    let act = Tensor::<CpuRuntime>::from_slice(&act_data, &[m, k], &device);

    // Build Q2K blocks with non-trivial data
    let mut weight_bytes = Vec::new();
    for row in 0..n {
        for blk in 0..2 {
            let mut block = [0u8; 84];
            // scales: non-zero low nibble (sub-scale) and high nibble (sub-min)
            for i in 0..16 {
                let s = ((i + row + blk) % 15 + 1) as u8; // 1-15
                let m_val = ((i + row * 3 + blk) % 10) as u8; // 0-9
                block[i] = s | (m_val << 4);
            }
            // qs: non-trivial 2-bit values packed in bytes
            for i in 0..64 {
                block[16 + i] = ((i + row * 7 + blk * 3) % 256) as u8;
            }
            // d = 0.5, dmin = 0.1
            block[80..82].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
            block[82..84].copy_from_slice(&f16::from_f32(0.1).to_le_bytes());
            weight_bytes.extend_from_slice(&block);
        }
    }

    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(&weight_bytes, QuantFormat::Q2K, &[n, k], &device)
            .unwrap();

    // Method 1: quant_matmul (generic path)
    let result_qm = client.quant_matmul(&act, &qt).unwrap();

    // Method 2: dequant then matmul
    let dequant_w = client.dequantize(&qt, DType::F32).unwrap();
    let dequant_w_t = dequant_w.transpose(0isize, 1isize).unwrap();
    let result_dm = MatmulOps::matmul(&client, &act, &dequant_w_t).unwrap();

    assert_eq!(result_qm.shape(), result_dm.shape());

    let qm_data = result_qm.to_vec::<f32>();
    let dm_data = result_dm.to_vec::<f32>();
    for (i, (&a, &b)) in qm_data.iter().zip(dm_data.iter()).enumerate() {
        let tol = 0.05 * b.abs().max(1.0);
        assert!(
            (a - b).abs() < tol,
            "Q2K mismatch at index {}: quant_matmul={}, dequant+matmul={}, tol={}",
            i,
            a,
            b,
            tol
        );
    }
}

#[test]
#[allow(clippy::needless_range_loop)]
fn test_quant_matmul_q3k_matches_dequant_matmul() {
    let (client, device) = setup();

    // K=512 (2 Q3K blocks), N=3, M=2
    let k = 512;
    let n = 3;
    let m = 2;

    let act_data: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
    let act = Tensor::<CpuRuntime>::from_slice(&act_data, &[m, k], &device);

    // Build Q3K blocks with non-trivial data
    let mut weight_bytes = Vec::new();
    for row in 0..n {
        for blk in 0..2 {
            let mut block = [0u8; 110];
            // hmask[32]: non-trivial high bits
            for i in 0..32 {
                block[i] = ((i * 7 + row * 13 + blk * 5) % 256) as u8;
            }
            // qs[64]: non-trivial 2-bit values
            for i in 0..64 {
                block[32 + i] = ((i * 11 + row * 3 + blk * 7) % 256) as u8;
            }
            // scales[12]: non-trivial packed 6-bit scales
            for i in 0..12 {
                block[96 + i] = ((i * 5 + row * 9 + blk) % 256) as u8;
            }
            // d = 0.3
            block[108..110].copy_from_slice(&f16::from_f32(0.3).to_le_bytes());
            weight_bytes.extend_from_slice(&block);
        }
    }

    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(&weight_bytes, QuantFormat::Q3K, &[n, k], &device)
            .unwrap();

    // Method 1: quant_matmul (generic path)
    let result_qm = client.quant_matmul(&act, &qt).unwrap();

    // Method 2: dequant then matmul
    let dequant_w = client.dequantize(&qt, DType::F32).unwrap();
    let dequant_w_t = dequant_w.transpose(0isize, 1isize).unwrap();
    let result_dm = MatmulOps::matmul(&client, &act, &dequant_w_t).unwrap();

    assert_eq!(result_qm.shape(), result_dm.shape());

    let qm_data = result_qm.to_vec::<f32>();
    let dm_data = result_dm.to_vec::<f32>();
    for (i, (&a, &b)) in qm_data.iter().zip(dm_data.iter()).enumerate() {
        let tol = 0.05 * b.abs().max(1.0);
        assert!(
            (a - b).abs() < tol,
            "Q3K mismatch at index {}: quant_matmul={}, dequant+matmul={}, tol={}",
            i,
            a,
            b,
            tol
        );
    }
}
