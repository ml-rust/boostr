//! GGUF metadata container

use super::value::GgufValue;
use std::collections::HashMap;

/// GGUF metadata (key-value pairs from the file header)
#[derive(Debug, Clone, Default)]
pub struct GgufMetadata {
    pub(crate) kv: HashMap<String, GgufValue>,
}

impl GgufMetadata {
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.kv.get(key).and_then(|v| v.as_string())
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.kv.get(key).and_then(|v| v.as_u32())
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.kv.get(key).and_then(|v| v.as_f32())
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.kv.get(key).and_then(|v| v.as_bool())
    }

    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.kv.get(key)
    }

    /// Every key-value pair, in no fixed order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &GgufValue)> {
        self.kv.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Number of key-value pairs.
    pub fn len(&self) -> usize {
        self.kv.len()
    }

    /// Whether the header holds no key-value pair.
    pub fn is_empty(&self) -> bool {
        self.kv.is_empty()
    }

    /// Get an array value by key.
    ///
    /// Returns the array elements if the value at `key` is a `GgufValue::Array`.
    pub fn get_array(&self, key: &str) -> Option<&[GgufValue]> {
        self.kv.get(key).and_then(|v| match v {
            GgufValue::Array(arr) => Some(arr.as_slice()),
            _ => None,
        })
    }

    /// Get a `UINT8` array by key as a contiguous byte buffer.
    ///
    /// Returns `None` when the key is absent, is not an array, or holds any
    /// element wider than a byte. The all-or-nothing rule is deliberate: the
    /// consumers of these blobs (SentencePiece's `precompiled_charsmap`, a
    /// darts-clone trie) are parsed as whole byte sequences, so a partially
    /// decoded buffer would be worse than no buffer at all.
    pub fn get_u8_array(&self, key: &str) -> Option<Vec<u8>> {
        self.get_array(key)?
            .iter()
            .map(GgufValue::as_u8)
            .collect::<Option<Vec<u8>>>()
    }

    /// Get a `STRING` array by key. All-or-nothing, like [`get_u8_array`](Self::get_u8_array).
    pub fn get_string_array(&self, key: &str) -> Option<Vec<String>> {
        self.get_array(key)?
            .iter()
            .map(|v| v.as_string().map(str::to_string))
            .collect::<Option<Vec<String>>>()
    }

    /// Get an integer array by key, widened to `i64`. All-or-nothing.
    ///
    /// Accepts any stored integer width (GGUF writers commonly use `INT32`
    /// for signed metadata arrays); see [`GgufValue::as_i64`].
    pub fn get_i64_array(&self, key: &str) -> Option<Vec<i64>> {
        self.get_array(key)?
            .iter()
            .map(GgufValue::as_i64)
            .collect::<Option<Vec<i64>>>()
    }

    /// Model architecture (e.g., "llama")
    pub fn architecture(&self) -> Option<&str> {
        self.get_string("general.architecture")
    }

    /// Number of transformer blocks
    pub fn block_count(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.get_u32(&format!("{arch}.block_count"))
    }

    /// Hidden/embedding dimension
    pub fn embedding_length(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.get_u32(&format!("{arch}.embedding_length"))
    }

    /// Context length
    pub fn context_length(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.get_u32(&format!("{arch}.context_length"))
    }
}
