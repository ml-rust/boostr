//! Low-level binary reading helpers for the GGUF format.
//!
//! All functions in this module read from any `std::io::Read` source in
//! little-endian byte order, matching the GGUF binary specification.

use super::tensor_info::GgufTensorInfo;
use super::types::{GgmlType, GgufValueType};
use super::value::GgufValue;
use crate::error::{Error, Result};
use std::io::Read;

pub(super) const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
pub(super) const GGUF_DEFAULT_ALIGNMENT: usize = 32;

// ── Scalar primitives ──────────────────────────────────────────────

pub(super) fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;
    Ok(buf[0])
}

pub(super) fn read_u16<R: Read>(r: &mut R) -> Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;
    Ok(u16::from_le_bytes(buf))
}

pub(super) fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;
    Ok(u32::from_le_bytes(buf))
}

pub(super) fn read_i32<R: Read>(r: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;
    Ok(i32::from_le_bytes(buf))
}

pub(super) fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;
    Ok(u64::from_le_bytes(buf))
}

pub(super) fn read_i64<R: Read>(r: &mut R) -> Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;
    Ok(i64::from_le_bytes(buf))
}

pub(super) fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;
    Ok(f32::from_le_bytes(buf))
}

pub(super) fn read_f64<R: Read>(r: &mut R) -> Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;
    Ok(f64::from_le_bytes(buf))
}

// ── String / value / KV / tensor-info parsing ──────────────────────

pub(super) fn read_string<R: Read>(r: &mut R, _version: u32) -> Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(|e| Error::ModelError {
        reason: format!("IO error: {e}"),
    })?;
    String::from_utf8(buf).map_err(|e| Error::ModelError {
        reason: format!("invalid UTF-8: {e}"),
    })
}

pub(super) fn read_value<R: Read>(r: &mut R, vt: GgufValueType, version: u32) -> Result<GgufValue> {
    match vt {
        GgufValueType::Uint8 => Ok(GgufValue::Uint8(read_u8(r)?)),
        GgufValueType::Int8 => Ok(GgufValue::Int8(read_u8(r)? as i8)),
        GgufValueType::Uint16 => Ok(GgufValue::Uint16(read_u16(r)?)),
        GgufValueType::Int16 => Ok(GgufValue::Int16(read_u16(r)? as i16)),
        GgufValueType::Uint32 => Ok(GgufValue::Uint32(read_u32(r)?)),
        GgufValueType::Int32 => Ok(GgufValue::Int32(read_i32(r)?)),
        GgufValueType::Float32 => Ok(GgufValue::Float32(read_f32(r)?)),
        GgufValueType::Bool => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        GgufValueType::String => Ok(GgufValue::String(read_string(r, version)?)),
        GgufValueType::Array => {
            let elem_type = read_u32(r)?;
            let elem_type =
                GgufValueType::from_u32(elem_type).ok_or_else(|| Error::ModelError {
                    reason: format!("invalid array element type: {elem_type}"),
                })?;
            let len = read_u64(r)? as usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_value(r, elem_type, version)?);
            }
            Ok(GgufValue::Array(arr))
        }
        GgufValueType::Uint64 => Ok(GgufValue::Uint64(read_u64(r)?)),
        GgufValueType::Int64 => Ok(GgufValue::Int64(read_i64(r)?)),
        GgufValueType::Float64 => Ok(GgufValue::Float64(read_f64(r)?)),
    }
}

pub(super) fn read_kv_pair<R: Read>(r: &mut R, version: u32) -> Result<(String, GgufValue)> {
    let key = read_string(r, version)?;
    let vt_raw = read_u32(r)?;
    let vt = GgufValueType::from_u32(vt_raw).ok_or_else(|| Error::ModelError {
        reason: format!("invalid value type: {vt_raw}"),
    })?;
    let value = read_value(r, vt, version)?;
    Ok((key, value))
}

pub(super) fn read_tensor_info<R: Read>(r: &mut R, version: u32) -> Result<GgufTensorInfo> {
    let name = read_string(r, version)?;
    let n_dims = read_u32(r)?;

    let mut shape = Vec::with_capacity(n_dims as usize);
    for _ in 0..n_dims {
        shape.push(read_u64(r)? as usize);
    }

    let ggml_type_raw = read_u32(r)?;
    let ggml_type = GgmlType::from_u32(ggml_type_raw).ok_or_else(|| Error::ModelError {
        reason: format!("unsupported GGML type: {ggml_type_raw}"),
    })?;

    let offset = read_u64(r)?;

    Ok(GgufTensorInfo {
        name,
        n_dims,
        shape,
        ggml_type,
        offset,
    })
}

// ── Alignment helper ───────────────────────────────────────────────

pub(super) fn align_offset(offset: u64, alignment: usize) -> u64 {
    let a = alignment as u64;
    offset.div_ceil(a) * a
}
