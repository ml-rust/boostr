//! Tests for the GGUF reader. Included only under #[cfg(test)] from reader.rs.

use super::super::io::{GGUF_MAGIC, align_offset};
use super::super::types::{GgmlType, GgufValueType};
use super::*;
use crate::test_utils::cpu_setup;
use numr::runtime::cpu::CpuRuntime;
use std::io::Write;
use tempfile::NamedTempFile;

/// Helper: write a GGUF string (u64 length + bytes)
fn write_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// Build a minimal GGUF v3 byte buffer with one F32 tensor and one Q4_0 tensor.
fn create_test_gguf_bytes() -> Vec<u8> {
    let mut buf = Vec::new();

    // Magic
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    // Version 3
    buf.extend_from_slice(&3u32.to_le_bytes());
    // 2 tensors
    buf.extend_from_slice(&2u64.to_le_bytes());
    // 2 KV pairs
    buf.extend_from_slice(&2u64.to_le_bytes());

    // KV 1: general.architecture = "test"
    write_str(&mut buf, "general.architecture");
    buf.extend_from_slice(&(GgufValueType::String as u32).to_le_bytes());
    write_str(&mut buf, "test");

    // KV 2: test.block_count = 4
    write_str(&mut buf, "test.block_count");
    buf.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes());

    // Tensor 1: "weight_f32" F32 [4]
    write_str(&mut buf, "weight_f32");
    buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    buf.extend_from_slice(&4u64.to_le_bytes()); // dim[0]
    buf.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // offset

    // Tensor 2: "weight_q4" Q4_0 [32]
    write_str(&mut buf, "weight_q4");
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&32u64.to_le_bytes());
    buf.extend_from_slice(&(GgmlType::Q4_0 as u32).to_le_bytes());
    buf.extend_from_slice(&16u64.to_le_bytes()); // offset after 4 floats

    // Align to 32 bytes
    let aligned = buf.len().div_ceil(32) * 32;
    buf.resize(aligned, 0);

    // Data: weight_f32 = [1.0, 2.0, 3.0, 4.0]
    for f in [1.0f32, 2.0, 3.0, 4.0] {
        buf.extend_from_slice(&f.to_le_bytes());
    }

    // Data: weight_q4 - Q4_0 block (scale=1.0, all nibbles=8 -> dequant to 0)
    let scale_bits = half::f16::from_f32(1.0).to_bits();
    buf.push((scale_bits & 0xFF) as u8);
    buf.push(((scale_bits >> 8) & 0xFF) as u8);
    buf.extend(std::iter::repeat_n(0x88u8, 16));

    buf
}

/// Create a minimal GGUF v3 file with one F32 tensor and one Q4_0 tensor
fn create_test_gguf() -> NamedTempFile {
    let buf = create_test_gguf_bytes();
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&buf).unwrap();
    file.flush().unwrap();
    file
}

#[test]
fn test_open_gguf() {
    let f = create_test_gguf();
    let gguf = Gguf::open(f.path()).unwrap();
    assert_eq!(gguf.version(), 3);
    assert_eq!(gguf.len(), 2);
    assert_eq!(gguf.metadata().architecture(), Some("test"));
    assert_eq!(gguf.metadata().block_count(), Some(4));
}

#[test]
fn test_tensor_info_gguf() {
    let f = create_test_gguf();
    let gguf = Gguf::open(f.path()).unwrap();

    let f32_info = gguf.tensor_info("weight_f32").unwrap();
    assert_eq!(f32_info.shape, vec![4]);
    assert_eq!(f32_info.ggml_type, GgmlType::F32);
    assert_eq!(f32_info.size_bytes(), 16);

    let q4_info = gguf.tensor_info("weight_q4").unwrap();
    assert_eq!(q4_info.shape, vec![32]);
    assert_eq!(q4_info.ggml_type, GgmlType::Q4_0);
    assert_eq!(q4_info.size_bytes(), 18);
}

#[test]
fn test_load_f32_tensor() {
    let (_, device) = cpu_setup();
    let f = create_test_gguf();
    let mut gguf = Gguf::open(f.path()).unwrap();

    let tensor = gguf
        .load_tensor_f32::<CpuRuntime>("weight_f32", &device)
        .unwrap();
    // GGUF 1D tensor: shape reversed is still [4]
    assert_eq!(tensor.shape(), &[4]);
    let data = tensor.to_vec::<f32>();
    assert!((data[0] - 1.0).abs() < 1e-6);
    assert!((data[3] - 4.0).abs() < 1e-6);
}

#[test]
fn test_load_quantized_tensor() {
    let f = create_test_gguf();
    let mut gguf = Gguf::open(f.path()).unwrap();

    let device = numr::runtime::cpu::CpuDevice::new();
    let qt = gguf
        .load_tensor_quantized::<numr::runtime::cpu::CpuRuntime>("weight_q4", &device)
        .unwrap();
    assert_eq!(qt.shape(), &[32]);
    assert_eq!(qt.format(), crate::quant::QuantFormat::Q4_0);
}

#[test]
fn test_align_offset() {
    assert_eq!(align_offset(0, 32), 0);
    assert_eq!(align_offset(1, 32), 32);
    assert_eq!(align_offset(32, 32), 32);
    assert_eq!(align_offset(33, 32), 64);
}

#[test]
fn test_tensor_not_found() {
    let f = create_test_gguf();
    let gguf = Gguf::open(f.path()).unwrap();
    assert!(gguf.tensor_info("nonexistent").is_err());
}

#[test]
fn test_open_with_mmap() {
    let (_, device) = cpu_setup();
    let f = create_test_gguf();
    let mut gguf = Gguf::open_with_mmap(f.path(), true).unwrap();
    assert_eq!(gguf.version(), 3);
    let tensor = gguf
        .load_tensor_f32::<CpuRuntime>("weight_f32", &device)
        .unwrap();
    assert_eq!(tensor.shape(), &[4]);
    let data = tensor.to_vec::<f32>();
    assert!((data[0] - 1.0).abs() < 1e-6);
    assert!((data[3] - 4.0).abs() < 1e-6);
}

#[test]
fn test_open_without_mmap() {
    let (_, device) = cpu_setup();
    let f = create_test_gguf();
    // use_mmap=false should behave identically to open()
    let mut gguf = Gguf::open_with_mmap(f.path(), false).unwrap();
    let tensor = gguf
        .load_tensor_f32::<CpuRuntime>("weight_f32", &device)
        .unwrap();
    let data = tensor.to_vec::<f32>();
    assert!((data[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_from_bytes() {
    let buf = create_test_gguf_bytes();
    let gguf = Gguf::from_bytes(buf).unwrap();
    assert_eq!(gguf.version(), 3);
    assert_eq!(gguf.len(), 2);
    assert_eq!(gguf.metadata().architecture(), Some("test"));
}

#[test]
fn test_from_bytes_load_f32() {
    let (_, device) = cpu_setup();
    let buf = create_test_gguf_bytes();
    let mut gguf = Gguf::from_bytes(buf).unwrap();

    let tensor = gguf
        .load_tensor_f32::<CpuRuntime>("weight_f32", &device)
        .unwrap();
    assert_eq!(tensor.shape(), &[4]);
    let data = tensor.to_vec::<f32>();
    assert!((data[0] - 1.0).abs() < 1e-6);
    assert!((data[3] - 4.0).abs() < 1e-6);
}

#[test]
fn test_from_bytes_load_quantized() {
    let buf = create_test_gguf_bytes();
    let mut gguf = Gguf::from_bytes(buf).unwrap();

    let device = numr::runtime::cpu::CpuDevice::new();
    let qt = gguf
        .load_tensor_quantized::<CpuRuntime>("weight_q4", &device)
        .unwrap();
    assert_eq!(qt.shape(), &[32]);
    assert_eq!(qt.format(), crate::quant::QuantFormat::Q4_0);
}
