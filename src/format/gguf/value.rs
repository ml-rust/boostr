//! GGUF metadata value types

/// GGUF metadata value (type-erased)
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    pub fn as_string(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::Uint8(v) => Some(*v as u32),
            GgufValue::Uint16(v) => Some(*v as u32),
            GgufValue::Uint32(v) => Some(*v),
            GgufValue::Int32(v) => Some(*v as u32),
            _ => None,
        }
    }

    /// Narrow accessor for `UINT8` payloads.
    ///
    /// Separate from [`as_u32`](Self::as_u32) because the callers differ: `u32`
    /// is read for counts and ids, where a widening cast is harmless, while
    /// `u8` is read for opaque byte blobs (`tokenizer.ggml.precompiled_charsmap`)
    /// that must survive verbatim. Anything wider than a byte is refused rather
    /// than truncated, so a mistyped array surfaces as `None` instead of silently
    /// corrupting the blob.
    pub fn as_u8(&self) -> Option<u8> {
        match self {
            GgufValue::Uint8(v) => Some(*v),
            GgufValue::Int8(v) => Some(*v as u8),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::Uint32(v) => Some(*v as u64),
            GgufValue::Uint64(v) => Some(*v),
            GgufValue::Int64(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::Float32(v) => Some(*v),
            GgufValue::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array(arr) => Some(arr),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_accessors() {
        let s = GgufValue::String("test".into());
        assert_eq!(s.as_string(), Some("test"));
        assert_eq!(s.as_u32(), None);

        let n = GgufValue::Uint32(42);
        assert_eq!(n.as_u32(), Some(42));
        assert_eq!(n.as_u64(), Some(42));
    }
}
