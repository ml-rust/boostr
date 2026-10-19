//! SafeTensors header encoding

use super::encode::dtype_name;
use crate::error::{Error, Result};
use numr::dtype::DType;
use serde_json::{Map, Value};
use std::collections::HashMap;

/// Name, shape, and dtype of one tensor, in file order
pub(super) type Spec<'a> = (&'a str, &'a [usize], DType);

/// Serialize the header JSON, prefixed by its little-endian `u64` length
///
/// Byte offsets come from shape and dtype, so no tensor data is read.
/// Metadata keys and all other object keys are sorted, so equal inputs
/// always yield equal bytes. An unsupported dtype returns an error.
pub(super) fn encode_header(
    specs: &[Spec<'_>],
    metadata: Option<&HashMap<String, String>>,
) -> Result<Vec<u8>> {
    let mut header = Map::new();

    if let Some(meta) = metadata {
        let mut keys: Vec<&String> = meta.keys().collect();
        keys.sort();
        let mut meta_obj = Map::new();
        for k in keys {
            meta_obj.insert(k.clone(), Value::String(meta[k].clone()));
        }
        header.insert("__metadata__".into(), Value::Object(meta_obj));
    }

    let mut current_offset: usize = 0;
    for (name, shape, dtype) in specs {
        let byte_len = shape.iter().product::<usize>() * dtype.size_in_bytes();
        let mut info = Map::new();
        info.insert("dtype".into(), Value::String(dtype_name(*dtype)?.into()));
        info.insert(
            "shape".into(),
            Value::Array(shape.iter().map(|&s| Value::Number(s.into())).collect()),
        );
        info.insert(
            "data_offsets".into(),
            Value::Array(vec![
                Value::Number(current_offset.into()),
                Value::Number((current_offset + byte_len).into()),
            ]),
        );
        header.insert((*name).to_string(), Value::Object(info));
        current_offset += byte_len;
    }

    // Sorted keys keep the header bytes fixed for any map order.
    let mut header = Value::Object(header);
    header.sort_all_objects();
    let header_str = serde_json::to_string(&header).map_err(|e| Error::ModelError {
        reason: format!("JSON serialize error: {e}"),
    })?;
    let mut out = Vec::with_capacity(8 + header_str.len());
    out.extend_from_slice(&(header_str.len() as u64).to_le_bytes());
    out.extend_from_slice(header_str.as_bytes());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_insertion_order_does_not_change_bytes() {
        let keys = ["zeta", "alpha", "mid", "beta", "omega", "format"];
        let build = |order: &[&str]| {
            let mut m = HashMap::with_capacity(order.len() * 7 + 1);
            for k in order {
                m.insert((*k).to_string(), format!("v-{k}"));
            }
            m
        };
        let reversed: Vec<&str> = keys.iter().rev().copied().collect();
        let specs: [Spec<'_>; 1] = [("w", &[2, 3], DType::F32)];
        let a = encode_header(&specs, Some(&build(&keys))).unwrap();
        let b = encode_header(&specs, Some(&build(&reversed))).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_metadata_keys_are_sorted_in_output() {
        let mut m = HashMap::new();
        m.insert("b".to_string(), "2".to_string());
        m.insert("a".to_string(), "1".to_string());
        let out = encode_header(&[], Some(&m)).unwrap();
        let text = std::str::from_utf8(&out[8..]).unwrap();
        assert_eq!(text, r#"{"__metadata__":{"a":"1","b":"2"}}"#);
    }

    #[test]
    fn test_unsupported_dtype_is_an_error() {
        let specs: [Spec<'_>; 1] = [("x", &[1], DType::U64)];
        assert!(encode_header(&specs, None).is_err());
    }
}
