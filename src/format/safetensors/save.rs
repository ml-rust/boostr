//! SafeTensors file format writer

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

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
fn tensor_to_le_bytes(tensor: &Tensor<CpuRuntime>) -> Result<Vec<u8>> {
    let bytes = match tensor.dtype() {
        DType::F32 => encode_le(tensor.to_vec::<f32>(), f32::to_le_bytes),
        DType::F64 => encode_le(tensor.to_vec::<f64>(), f64::to_le_bytes),
        DType::F16 => encode_le(tensor.to_vec::<half::f16>(), half::f16::to_le_bytes),
        DType::BF16 => encode_le(tensor.to_vec::<half::bf16>(), half::bf16::to_le_bytes),
        DType::I64 => encode_le(tensor.to_vec::<i64>(), i64::to_le_bytes),
        DType::I32 => encode_le(tensor.to_vec::<i32>(), i32::to_le_bytes),
        DType::I8 => encode_le(tensor.to_vec::<i8>(), i8::to_le_bytes),
        DType::U32 => encode_le(tensor.to_vec::<u32>(), u32::to_le_bytes),
        // Bool is stored one byte per element, same as u8.
        DType::Bool => encode_le(tensor.to_vec::<u8>(), u8::to_le_bytes),
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
/// Mirrors [`parse_dtype`](super::parse_dtype) exactly: boostr writes every
/// dtype it can read.
fn dtype_name(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("F32"),
        DType::F16 => Ok("F16"),
        DType::BF16 => Ok("BF16"),
        DType::F64 => Ok("F64"),
        DType::I32 => Ok("I32"),
        DType::I64 => Ok("I64"),
        DType::U32 => Ok("U32"),
        DType::I8 => Ok("I8"),
        DType::Bool => Ok("BOOL"),
        other => Err(Error::ModelError {
            reason: format!("unsupported SafeTensors save dtype: {other:?}"),
        }),
    }
}

/// Save tensors to SafeTensors format
///
/// Each tensor is written at its own dtype — a BF16 tensor produces a BF16
/// entry at 2 bytes per element, not an F32 one at 4.
///
/// Only accepts CPU tensors. Move GPU tensors to CPU before saving.
pub fn save_safetensors<P: AsRef<Path>>(
    path: P,
    tensors: &HashMap<String, Tensor<CpuRuntime>>,
    metadata: Option<&HashMap<String, String>>,
) -> Result<()> {
    use std::io::Write;

    let mut file = File::create(path).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;

    // Collect tensor data and build header
    let mut tensor_entries: Vec<(String, Vec<u8>, Vec<usize>, DType)> = Vec::new();
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();

    for name in &names {
        let tensor = &tensors[*name];
        let dtype = tensor.dtype();
        let data = tensor_to_le_bytes(tensor)?;
        let shape = tensor.shape().to_vec();
        tensor_entries.push(((*name).clone(), data, shape, dtype));
    }

    // Build header JSON
    let mut header = serde_json::Map::new();

    if let Some(meta) = metadata {
        let mut meta_obj = serde_json::Map::new();
        for (k, v) in meta {
            meta_obj.insert(k.clone(), serde_json::Value::String(v.clone()));
        }
        header.insert("__metadata__".into(), serde_json::Value::Object(meta_obj));
    }

    let mut current_offset: usize = 0;
    for (name, data, shape, dtype) in &tensor_entries {
        let byte_len = data.len();
        let mut info = serde_json::Map::new();
        info.insert(
            "dtype".into(),
            serde_json::Value::String(dtype_name(*dtype)?.into()),
        );
        info.insert(
            "shape".into(),
            serde_json::Value::Array(
                shape
                    .iter()
                    .map(|&s| serde_json::Value::Number(s.into()))
                    .collect(),
            ),
        );
        info.insert(
            "data_offsets".into(),
            serde_json::Value::Array(vec![
                serde_json::Value::Number(current_offset.into()),
                serde_json::Value::Number((current_offset + byte_len).into()),
            ]),
        );
        header.insert(name.clone(), serde_json::Value::Object(info));
        current_offset += byte_len;
    }

    let header_str = serde_json::to_string(&serde_json::Value::Object(header)).map_err(|e| {
        Error::ModelError {
            reason: format!("JSON serialize error: {e}"),
        }
    })?;
    let header_bytes = header_str.as_bytes();

    // Write header size + header + data
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .map_err(|e| Error::ModelError {
            reason: format!("IO error: {e}"),
        })?;
    file.write_all(header_bytes)
        .map_err(|e| Error::ModelError {
            reason: format!("IO error: {e}"),
        })?;
    for (_, data, _, _) in &tensor_entries {
        file.write_all(data).map_err(|e| Error::ModelError {
            reason: format!("IO error: {e}"),
        })?;
    }
    file.flush().map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;

    Ok(())
}
