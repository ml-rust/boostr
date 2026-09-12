//! Decoding SafeTensors payloads into `Tensor<R>` at their native dtype.

use super::header::SafeTensors;
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::collections::HashMap;

impl SafeTensors {
    /// Load a tensor in its native dtype on the given device
    ///
    /// Preserves the original dtype from the SafeTensors file (F32, F16, BF16, etc.)
    /// without converting to F32. This halves memory for BF16/F16 models.
    pub fn load_tensor<R: Runtime<DType = DType>>(
        &mut self,
        name: &str,
        device: &R::Device,
    ) -> Result<Tensor<R>> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::ModelError {
                reason: format!("tensor not found: {name}"),
            })?
            .clone();

        let bytes = self.read_tensor_bytes(name)?;

        match info.dtype {
            DType::F32 | DType::F16 | DType::BF16 => {
                // Load raw bytes directly in native dtype
                let storage = numr::tensor::Storage::<R>::from_bytes(&bytes, info.dtype, device)
                    .map_err(Error::Numr)?;
                Ok(Tensor::<R>::from_storage_contiguous(storage, &info.shape))
            }
            DType::F64 => {
                // Downcast F64 to F32 (F64 weights are rare and wasteful)
                let data: Vec<f32> = bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| {
                        f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
                    })
                    .collect();
                Tensor::<R>::from_slice(&data, &info.shape, device).map_err(Error::Numr)
            }
            DType::I64 => {
                // SafeTensors stores integers little-endian; decode explicitly
                // rather than reinterpreting raw bytes, so this is correct on
                // both little- and big-endian hosts.
                let data: Vec<i64> = bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| i64::from_le_bytes(*b))
                    .collect();
                Tensor::<R>::from_slice(&data, &info.shape, device).map_err(Error::Numr)
            }
            DType::I32 => {
                // Little-endian decode, see the I64 arm above.
                let data: Vec<i32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| i32::from_le_bytes(*b))
                    .collect();
                Tensor::<R>::from_slice(&data, &info.shape, device).map_err(Error::Numr)
            }
            DType::U32 => {
                // Little-endian decode, see the I64 arm above.
                let data: Vec<u32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect();
                Tensor::<R>::from_slice(&data, &info.shape, device).map_err(Error::Numr)
            }
            DType::I8 | DType::Bool => {
                // Single-byte types have no endianness to decode; load the
                // raw bytes directly, same as the float arm above.
                let storage = numr::tensor::Storage::<R>::from_bytes(&bytes, info.dtype, device)
                    .map_err(Error::Numr)?;
                Ok(Tensor::<R>::from_storage_contiguous(storage, &info.shape))
            }
            // Reached only if `parse_dtype` is ever extended to accept a
            // SafeTensors dtype string this arm doesn't yet decode (e.g.
            // I16/U16/U64/U8, which numr's DType supports but this loader
            // does not yet handle) or DType gains a variant `parse_dtype`
            // can never produce (Complex64/128, FP8). Names the dtype so
            // the caller sees what to add rather than silently coercing it.
            other => Err(Error::ModelError {
                reason: format!("unsupported SafeTensors dtype: {other:?}"),
            }),
        }
    }

    /// Load all tensors to the given device
    pub fn load_all<R: Runtime<DType = DType>>(
        &mut self,
        device: &R::Device,
    ) -> Result<HashMap<String, Tensor<R>>> {
        let names: Vec<String> = self.tensors.keys().cloned().collect();
        let mut result = HashMap::with_capacity(names.len());
        for name in names {
            let tensor = self.load_tensor::<R>(&name, device)?;
            result.insert(name, tensor);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::super::header::tests::create_test_file;
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_tensor_f32() {
        let (_, device) = cpu_setup();
        let f = create_test_file();
        let mut st = SafeTensors::open(f.path()).unwrap();
        let tensor = st.load_tensor::<CpuRuntime>("weight", &device).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        let data = tensor.to_vec::<f32>();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[5] - 6.0).abs() < 1e-6);
    }

    fn create_test_file_bf16() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();

        let header = serde_json::json!({
            "__metadata__": { "format": "pt" },
            "weight": {
                "dtype": "BF16",
                "shape": [2, 3],
                "data_offsets": [0, 12]
            }
        });
        let header_str = header.to_string();
        let header_bytes = header_str.as_bytes();

        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header_bytes).unwrap();

        for f in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            file.write_all(&half::bf16::from_f32(f).to_le_bytes())
                .unwrap();
        }
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_load_tensor_bf16() {
        let (_, device) = cpu_setup();
        let f = create_test_file_bf16();
        let mut st = SafeTensors::open(f.path()).unwrap();
        let tensor = st.load_tensor::<CpuRuntime>("weight", &device).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.dtype(), DType::BF16);
        let data: Vec<half::bf16> = tensor.to_vec();
        assert!((data[0].to_f32() - 1.0).abs() < 1e-2);
        assert!((data[5].to_f32() - 6.0).abs() < 1e-2);
    }

    // ===== load_tensor: integer dtypes =====

    /// Writes a single-tensor SafeTensors file with a raw little-endian payload,
    /// bypassing `save_safetensors` so the test exercises `load_tensor`'s decode
    /// in isolation from the writer.
    fn write_int_tensor_file(
        dtype_str: &str,
        byte_payload: &[u8],
        shape: &[usize],
    ) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        let header = serde_json::json!({
            "w": {
                "dtype": dtype_str,
                "shape": shape,
                "data_offsets": [0, byte_payload.len()]
            }
        });
        let header_str = header.to_string();
        let header_bytes = header_str.as_bytes();
        file.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header_bytes).unwrap();
        file.write_all(byte_payload).unwrap();
        file.flush().unwrap();
        file
    }

    /// I64 round-trip covering a negative value and a magnitude above 2^32 — a
    /// truncation to 32 bits during decode would corrupt the latter silently.
    #[test]
    fn test_load_tensor_i64_roundtrip() {
        let (_, device) = cpu_setup();
        let values: [i64; 5] = [i64::MIN, i64::MAX, -1, 1i64 << 40, 0];
        let mut bytes = Vec::new();
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let f = write_int_tensor_file("I64", &bytes, &[5]);
        let mut st = SafeTensors::open(f.path()).unwrap();
        let tensor = st.load_tensor::<CpuRuntime>("w", &device).unwrap();
        assert_eq!(tensor.dtype(), DType::I64);
        assert_eq!(tensor.shape(), &[5]);
        assert_eq!(tensor.to_vec::<i64>(), values);
    }

    /// I32 round-trip (the narrower integer type) covering a negative value and
    /// both extremes of the 32-bit range.
    #[test]
    fn test_load_tensor_i32_roundtrip() {
        let (_, device) = cpu_setup();
        let values: [i32; 3] = [i32::MIN, i32::MAX, -12345];
        let mut bytes = Vec::new();
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let f = write_int_tensor_file("I32", &bytes, &[3]);
        let mut st = SafeTensors::open(f.path()).unwrap();
        let tensor = st.load_tensor::<CpuRuntime>("w", &device).unwrap();
        assert_eq!(tensor.dtype(), DType::I32);
        assert_eq!(tensor.to_vec::<i32>(), values);
    }

    #[test]
    fn test_load_tensor_u32_roundtrip() {
        let (_, device) = cpu_setup();
        let values: [u32; 2] = [0, u32::MAX];
        let mut bytes = Vec::new();
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let f = write_int_tensor_file("U32", &bytes, &[2]);
        let mut st = SafeTensors::open(f.path()).unwrap();
        let tensor = st.load_tensor::<CpuRuntime>("w", &device).unwrap();
        assert_eq!(tensor.dtype(), DType::U32);
        assert_eq!(tensor.to_vec::<u32>(), values);
    }

    #[test]
    fn test_load_tensor_i8_roundtrip() {
        let (_, device) = cpu_setup();
        let values: [i8; 3] = [i8::MIN, i8::MAX, -1];
        let bytes: Vec<u8> = values.iter().map(|&v| v as u8).collect();
        let f = write_int_tensor_file("I8", &bytes, &[3]);
        let mut st = SafeTensors::open(f.path()).unwrap();
        let tensor = st.load_tensor::<CpuRuntime>("w", &device).unwrap();
        assert_eq!(tensor.dtype(), DType::I8);
        assert_eq!(tensor.to_vec::<i8>(), values);
    }

    #[test]
    fn test_load_tensor_bool_roundtrip() {
        let (_, device) = cpu_setup();
        let bytes: Vec<u8> = vec![1, 0, 1, 1];
        let f = write_int_tensor_file("BOOL", &bytes, &[4]);
        let mut st = SafeTensors::open(f.path()).unwrap();
        let tensor = st.load_tensor::<CpuRuntime>("w", &device).unwrap();
        assert_eq!(tensor.dtype(), DType::Bool);
        assert_eq!(tensor.to_vec::<u8>(), bytes);
    }
}
