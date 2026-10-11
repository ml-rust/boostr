//! `VarMap::from_gguf` on a llama.cpp-layout `llama` file, CPU vs WebGPU.
//!
//! The file stores query and key rows in llama.cpp's pair order. The reader
//! returns them to HuggingFace order on the host, before upload. This test
//! loads Q8_0, Q4_K, F16, BF16 and F32 query/key tensors on both runtimes and
//! requires identical results.
//!
//! Run with:
//!   cd boostr && cargo test --features wgpu --test gguf_rope_rows_wgpu

#![cfg(feature = "wgpu")]

use std::io::Write;

use boostr::format::gguf::{GgmlType, GgufValueType, RopeRowOrder, reorder_rope_rows};
use boostr::nn::VarMap;
use boostr::quant::{DequantOps, QuantFormat};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};

const N_HEAD: usize = 2;
const N_HEAD_KV: usize = 1;
const HEAD_DIM: usize = 4;
const Q_ROWS: usize = N_HEAD * HEAD_DIM;
const K_ROWS: usize = N_HEAD_KV * HEAD_DIM;
const Q8_COLS: usize = 64;
const Q4K_COLS: usize = 256;

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

/// A distinct payload byte per index and row.
fn payload(i: usize, row: usize) -> u8 {
    ((i as u32)
        .wrapping_mul(31)
        .wrapping_add(row as u32 * 97)
        .wrapping_add(11)
        & 0xFF) as u8
}

/// One Q8_0 row of `Q8_COLS` columns. The scale differs per row.
fn q8_row(row: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for block in 0..Q8_COLS / 32 {
        let scale = 0.25 + 0.125 * (row * 2 + block) as f32;
        out.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
        out.extend((0..32).map(|i| payload(i + block * 32, row)));
    }
    out
}

/// One Q4_K row of `Q4K_COLS` columns: one 144-byte super-block.
fn q4k_row(row: usize) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&half::f16::from_f32(0.5 + 0.25 * row as f32).to_le_bytes());
    out.extend_from_slice(&half::f16::from_f32(0.125).to_le_bytes());
    out.extend((0..140).map(|i| payload(i, row)));
    out
}

fn le_f32(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values.flat_map(f32::to_le_bytes).collect()
}

fn write_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// Rows in HF order, stored in llama.cpp order.
fn to_gguf(hf: &[u8], rows: usize, heads: usize) -> Vec<u8> {
    reorder_rope_rows(hf, rows, heads, RopeRowOrder::ToGguf).unwrap()
}

/// A tiny `llama` GGUF in llama.cpp row order.
fn llama_cpp_gguf() -> tempfile::NamedTempFile {
    let q_weight: Vec<u8> = (0..Q_ROWS).flat_map(q8_row).collect();
    let k_weight: Vec<u8> = (0..K_ROWS).flat_map(q4k_row).collect();
    let q_bias = le_f32((0..Q_ROWS).map(|r| r as f32 * 0.5 - 1.0));
    let k_bias: Vec<u8> = (0..K_ROWS)
        .flat_map(|r| half::f16::from_f32(r as f32 + 0.25).to_le_bytes())
        .collect();
    let v_weight: Vec<u8> = (0..K_ROWS * 4)
        .flat_map(|i| half::bf16::from_f32(i as f32 - 3.0).to_le_bytes())
        .collect();

    let tensors: [(&str, Vec<u64>, GgmlType, Vec<u8>); 5] = [
        (
            "blk.0.attn_q.weight",
            vec![Q8_COLS as u64, Q_ROWS as u64],
            GgmlType::Q8_0,
            to_gguf(&q_weight, Q_ROWS, N_HEAD),
        ),
        (
            "blk.0.attn_k.weight",
            vec![Q4K_COLS as u64, K_ROWS as u64],
            GgmlType::Q4K,
            to_gguf(&k_weight, K_ROWS, N_HEAD_KV),
        ),
        (
            "blk.0.attn_q.bias",
            vec![Q_ROWS as u64],
            GgmlType::F32,
            to_gguf(&q_bias, Q_ROWS, N_HEAD),
        ),
        (
            "blk.0.attn_k.bias",
            vec![K_ROWS as u64],
            GgmlType::F16,
            to_gguf(&k_bias, K_ROWS, N_HEAD_KV),
        ),
        (
            "blk.0.attn_v.weight",
            vec![4, K_ROWS as u64],
            GgmlType::BF16,
            v_weight,
        ),
    ];

    let mut buf = Vec::new();
    buf.extend_from_slice(&0x46554747u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    buf.extend_from_slice(&3u64.to_le_bytes());
    write_str(&mut buf, "general.architecture");
    buf.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
    write_str(&mut buf, "llama");
    for (key, value) in [
        ("llama.attention.head_count", N_HEAD as u32),
        ("llama.attention.head_count_kv", N_HEAD_KV as u32),
    ] {
        write_str(&mut buf, key);
        buf.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
        buf.extend_from_slice(&value.to_le_bytes());
    }
    let mut offset = 0u64;
    for (name, dims, ty, data) in &tensors {
        write_str(&mut buf, name);
        buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&(*ty as u32).to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
        offset += (data.len() as u64).div_ceil(32) * 32;
    }
    for (_, _, _, data) in &tensors {
        buf.resize(buf.len().div_ceil(32) * 32, 0);
        buf.extend_from_slice(data);
    }
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&buf).unwrap();
    file.flush().unwrap();
    file
}

#[test]
fn llama_cpp_rope_rows_load_on_wgpu_as_on_cpu() {
    let Some((wgpu_client, wgpu_device)) = wgpu_setup() else {
        return;
    };
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuClient::new(cpu_device.clone());
    let file = llama_cpp_gguf();

    let cpu = VarMap::<CpuRuntime>::from_gguf(file.path(), &cpu_device).unwrap();
    let gpu = VarMap::<WgpuRuntime>::from_gguf(file.path(), &wgpu_device)
        .expect("from_gguf on wgpu reorders query/key rows without a device copy");

    // The quantized rows return to HF order, byte for byte, on both runtimes.
    let q_name = "model.layers.0.self_attn.q_proj.weight";
    let k_name = "model.layers.0.self_attn.k_proj.weight";
    let hf_q: Vec<u8> = (0..Q_ROWS).flat_map(q8_row).collect();
    let hf_k: Vec<u8> = (0..K_ROWS).flat_map(q4k_row).collect();
    for (name, hf, format, rows, cols) in [
        (q_name, &hf_q, QuantFormat::Q8_0, Q_ROWS, Q8_COLS),
        (k_name, &hf_k, QuantFormat::Q4K, K_ROWS, Q4K_COLS),
    ] {
        let c = cpu.get(name).unwrap().as_quant_tensor().unwrap();
        let g = gpu.get(name).unwrap().as_quant_tensor().unwrap();
        assert_eq!(c.format(), format);
        assert_eq!(&c.to_bytes().unwrap(), hf, "{name}: CPU rows in HF order");
        assert_eq!(g.to_bytes().unwrap(), c.to_bytes().unwrap(), "{name} bytes");

        // Both runtimes decode the same blocks to the same values.
        let dc: Vec<f32> = cpu_client.dequantize(c, DType::F32).unwrap().to_vec();
        let dg: Vec<f32> = wgpu_client.dequantize(g, DType::F32).unwrap().to_vec();
        assert_eq!(dc.len(), rows * cols);
        for (i, (a, b)) in dc.iter().zip(&dg).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6 * (1.0 + a.abs()),
                "{name} dequant [{i}]: cpu {a} vs wgpu {b}"
            );
        }
    }

    // Dense tensors: F32 and F16 rows reorder, BF16 value rows stay.
    let q_bias: Vec<f32> = (0..Q_ROWS).map(|r| r as f32 * 0.5 - 1.0).collect();
    let k_bias: Vec<f32> = (0..K_ROWS).map(|r| r as f32 + 0.25).collect();
    let v_weight: Vec<f32> = (0..K_ROWS * 4).map(|i| i as f32 - 3.0).collect();
    for (name, want) in [
        ("model.layers.0.self_attn.q_proj.bias", q_bias),
        ("model.layers.0.self_attn.k_proj.bias", k_bias),
        ("model.layers.0.self_attn.v_proj.weight", v_weight),
    ] {
        let c: Vec<f32> = cpu.get(name).unwrap().as_tensor().unwrap().to_vec();
        let g: Vec<f32> = gpu.get(name).unwrap().as_tensor().unwrap().to_vec();
        assert_eq!(c, want, "{name}: CPU");
        assert_eq!(g, c, "{name}: wgpu");
    }
}
