//! `block_size` validation: present, nonzero, power of two.

use super::config::{KEY_BLOCK_SIZE, missing_key, model_error};
use crate::error::Result;
use crate::format::gguf::metadata::GgufMetadata;

pub(super) fn parse_block_size(meta: &GgufMetadata) -> Result<usize> {
    let block_size = meta
        .get_u32(KEY_BLOCK_SIZE)
        .ok_or_else(|| missing_key(KEY_BLOCK_SIZE))?;
    if block_size == 0 || !block_size.is_power_of_two() {
        return Err(model_error(format!(
            "invalid {KEY_BLOCK_SIZE}: {block_size} (must be a nonzero power of two)"
        )));
    }
    Ok(block_size as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::gguf::GgufValue;

    fn meta_with(value: GgufValue) -> GgufMetadata {
        let mut m = GgufMetadata::default();
        m.kv.insert(KEY_BLOCK_SIZE.to_string(), value);
        m
    }

    #[test]
    fn missing_errors() {
        let m = GgufMetadata::default();
        let err = parse_block_size(&m).unwrap_err().to_string();
        assert!(err.contains(KEY_BLOCK_SIZE));
    }

    #[test]
    fn zero_errors() {
        let err = parse_block_size(&meta_with(GgufValue::Uint32(0)))
            .unwrap_err()
            .to_string();
        assert!(err.contains(KEY_BLOCK_SIZE));
    }

    #[test]
    fn not_power_of_two_errors() {
        let err = parse_block_size(&meta_with(GgufValue::Uint32(100)))
            .unwrap_err()
            .to_string();
        assert!(err.contains(KEY_BLOCK_SIZE));
    }

    #[test]
    fn power_of_two_accepted() {
        let got = parse_block_size(&meta_with(GgufValue::Uint32(1024))).unwrap();
        assert_eq!(got, 1024);
    }
}
