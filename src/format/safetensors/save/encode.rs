//! Host-byte encoding and header names for SafeTensors writing

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// Encode host values as little-endian bytes
///
/// `to_le` fixes the per-element width, so the byte length is always
/// `data.len() * N` — never a hard-wired constant.
fn encode_le<T, const N: usize>(data: Vec<T>, to_le: fn(T) -> [u8; N]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * N);
    for v in data {
        out.extend_from_slice(&to_le(v));
    }
    out
}

/// Copy a CPU tensor to host bytes at its OWN dtype width
///
/// Each arm reads with a host type whose width equals the tensor's dtype width.
/// Reading at a wider type (e.g. `to_vec::<f32>()` on a BF16 tensor) copies more
/// bytes than the storage holds, so it must never happen.
pub(super) fn tensor_to_le_bytes(tensor: &Tensor<CpuRuntime>) -> Result<Vec<u8>> {
    let bytes = match tensor.dtype() {
        DType::F32 => encode_le(tensor.to_vec::<f32>(), f32::to_le_bytes),
        DType::F64 => encode_le(tensor.to_vec::<f64>(), f64::to_le_bytes),
        DType::F16 => encode_le(tensor.to_vec::<half::f16>(), half::f16::to_le_bytes),
        DType::BF16 => encode_le(tensor.to_vec::<half::bf16>(), half::bf16::to_le_bytes),
        DType::I64 => encode_le(tensor.to_vec::<i64>(), i64::to_le_bytes),
        DType::I32 => encode_le(tensor.to_vec::<i32>(), i32::to_le_bytes),
        DType::I8 => encode_le(tensor.to_vec::<i8>(), i8::to_le_bytes),
        DType::U32 => encode_le(tensor.to_vec::<u32>(), u32::to_le_bytes),
        // U8 and Bool are stored one byte per element.
        DType::U8 | DType::Bool => encode_le(tensor.to_vec::<u8>(), u8::to_le_bytes),
        other => {
            return Err(Error::ModelError {
                reason: format!("unsupported SafeTensors save dtype: {other:?}"),
            });
        }
    };
    Ok(bytes)
}

/// SafeTensors header name for a dtype
///
/// Mirrors [`parse_dtype`](super::super::header::parse_dtype) exactly: boostr writes every
/// dtype it can read.
pub(super) fn dtype_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("F32"),
        DType::F16 => Ok("F16"),
        DType::BF16 => Ok("BF16"),
        DType::F64 => Ok("F64"),
        DType::I32 => Ok("I32"),
        DType::I64 => Ok("I64"),
        DType::U32 => Ok("U32"),
        DType::I8 => Ok("I8"),
        DType::U8 => Ok("U8"),
        DType::Bool => Ok("BOOL"),
        other => Err(Error::ModelError {
            reason: format!("unsupported SafeTensors save dtype: {other:?}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::safetensors::{SafeTensors, save_safetensors};
    use crate::test_utils::cpu_setup;
    use std::collections::HashMap;
    use tempfile::NamedTempFile;

    // ===== Save path: per-dtype writing =====

    /// Every dtype `parse_dtype` reads, `save_safetensors` must write.
    const SAVE_DTYPES: [DType; 10] = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::U32,
        DType::I8,
        DType::U8,
        DType::Bool,
    ];

    const SAMPLE_VALUES: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    /// Little-endian bytes for `SAMPLE_VALUES` encoded at `dtype`'s own width.
    fn sample_bytes(dtype: DType) -> Vec<u8> {
        let mut out = Vec::new();
        for (i, &v) in SAMPLE_VALUES.iter().enumerate() {
            match dtype {
                DType::F32 => out.extend_from_slice(&v.to_le_bytes()),
                DType::F64 => out.extend_from_slice(&f64::from(v).to_le_bytes()),
                DType::F16 => out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes()),
                DType::BF16 => out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes()),
                DType::I64 => out.extend_from_slice(&(v as i64).to_le_bytes()),
                DType::I32 => out.extend_from_slice(&(v as i32).to_le_bytes()),
                DType::I8 => out.extend_from_slice(&(v as i8).to_le_bytes()),
                DType::U8 => out.push(v as u8),
                DType::U32 => out.extend_from_slice(&(v as u32).to_le_bytes()),
                DType::Bool => out.push((i % 2) as u8),
                other => panic!("unsupported test dtype: {other:?}"),
            }
        }
        out
    }

    fn cpu_tensor_from_bytes(
        bytes: &[u8],
        dtype: DType,
        shape: &[usize],
        device: &numr::runtime::cpu::CpuDevice,
    ) -> Tensor<CpuRuntime> {
        let storage =
            numr::tensor::Storage::<CpuRuntime>::from_bytes(bytes, dtype, device).unwrap();
        Tensor::<CpuRuntime>::from_storage_contiguous(storage, shape)
    }

    /// Save a single 2x3 tensor of `dtype` built from `sample_bytes`.
    fn save_sample(
        dtype: DType,
        device: &numr::runtime::cpu::CpuDevice,
    ) -> (NamedTempFile, Vec<u8>) {
        let bytes = sample_bytes(dtype);
        let tensor = cpu_tensor_from_bytes(&bytes, dtype, &[2, 3], device);
        let mut tensors = HashMap::new();
        tensors.insert("w".to_string(), tensor);
        let tmp = NamedTempFile::new().unwrap();
        save_safetensors(tmp.path(), &tensors, None).unwrap();
        (tmp, bytes)
    }

    fn read_header_json(path: &std::path::Path) -> serde_json::Value {
        let raw = std::fs::read(path).unwrap();
        let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        serde_json::from_slice(&raw[8..8 + n]).unwrap()
    }

    /// On-disk bytes per element must equal the dtype's width.
    ///
    /// This is the assertion that catches a silent re-widening to F32: before the
    /// per-dtype write path, a BF16 tensor was saved as 4 bytes/element with an
    /// "F32" header, so `size_bytes()` here was 24 instead of 12 — and the host
    /// read itself was an out-of-bounds `to_vec::<f32>()` over 12 bytes of storage.
    #[test]
    fn test_save_writes_native_element_width() {
        let (_, device) = cpu_setup();
        for dtype in SAVE_DTYPES {
            let (tmp, _) = save_sample(dtype, &device);
            let st = SafeTensors::open(tmp.path()).unwrap();
            let info = st.tensor_info("w").unwrap();
            assert_eq!(info.dtype, dtype, "dtype mismatch for {dtype:?}");
            assert_eq!(
                info.size_bytes(),
                6 * dtype.size_in_bytes(),
                "byte width mismatch for {dtype:?}"
            );
            // The data section holds exactly one payload, at the native width.
            let raw = std::fs::read(tmp.path()).unwrap();
            let header_len = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
            assert_eq!(
                raw.len(),
                8 + header_len + 6 * dtype.size_in_bytes(),
                "file length mismatch for {dtype:?}"
            );
        }
    }

    /// The header's `dtype` string must name the tensor's own dtype.
    #[test]
    fn test_save_header_dtype_string() {
        let (_, device) = cpu_setup();
        let expected = [
            (DType::F32, "F32"),
            (DType::F16, "F16"),
            (DType::BF16, "BF16"),
            (DType::F64, "F64"),
            (DType::I32, "I32"),
            (DType::I64, "I64"),
            (DType::U32, "U32"),
            (DType::I8, "I8"),
            (DType::Bool, "BOOL"),
        ];
        for (dtype, name) in expected {
            let (tmp, _) = save_sample(dtype, &device);
            let header = read_header_json(tmp.path());
            assert_eq!(header["w"]["dtype"].as_str(), Some(name), "for {dtype:?}");
            assert_eq!(
                header["w"]["data_offsets"][1].as_u64(),
                Some((6 * dtype.size_in_bytes()) as u64),
                "data_offsets width for {dtype:?}"
            );
        }
    }

    /// Raw payload bytes survive the save unchanged, for every supported dtype.
    #[test]
    fn test_save_roundtrip_raw_bytes() {
        let (_, device) = cpu_setup();
        for dtype in SAVE_DTYPES {
            let (tmp, original) = save_sample(dtype, &device);
            let mut st = SafeTensors::open(tmp.path()).unwrap();
            assert_eq!(
                st.read_tensor_bytes("w").unwrap(),
                original,
                "payload mismatch for {dtype:?}"
            );
        }
    }

    /// Values and dtype both survive a save/reload through `load_tensor`.
    #[test]
    fn test_save_roundtrip_preserves_float_dtype() {
        let (_, device) = cpu_setup();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let (tmp, _) = save_sample(dtype, &device);
            let mut st = SafeTensors::open(tmp.path()).unwrap();
            let t = st.load_tensor::<CpuRuntime>("w", &device).unwrap();
            assert_eq!(t.shape(), &[2, 3]);
            assert_eq!(t.dtype(), dtype, "reloaded dtype changed for {dtype:?}");

            let got: Vec<f32> = match dtype {
                DType::F32 => t.to_vec::<f32>(),
                DType::F16 => t.to_vec::<half::f16>().iter().map(|v| v.to_f32()).collect(),
                _ => t
                    .to_vec::<half::bf16>()
                    .iter()
                    .map(|v| v.to_f32())
                    .collect(),
            };
            for (g, v) in got.iter().zip(SAMPLE_VALUES) {
                assert!((g - v).abs() < 1e-2, "value mismatch for {dtype:?}");
            }
        }
    }
}
