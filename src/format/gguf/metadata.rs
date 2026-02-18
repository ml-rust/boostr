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

    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.kv.get(key)
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
