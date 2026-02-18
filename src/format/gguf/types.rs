//! GGML and GGUF type definitions

use crate::quant::QuantFormat;

/// GGML tensor data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 30,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Number of elements per block
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
        }
    }

    /// Bytes per block
    pub fn block_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 40,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 276,
        }
    }

    pub fn is_quantized(&self) -> bool {
        !matches!(
            self,
            Self::F32
                | Self::F16
                | Self::BF16
                | Self::F64
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
        )
    }

    /// Convert to boostr QuantFormat (only valid for quantized types)
    pub fn to_quant_format(&self) -> Option<QuantFormat> {
        match self {
            Self::Q4_0 => Some(QuantFormat::Q4_0),
            Self::Q4_1 => Some(QuantFormat::Q4_1),
            Self::Q5_0 => Some(QuantFormat::Q5_0),
            Self::Q5_1 => Some(QuantFormat::Q5_1),
            Self::Q8_0 => Some(QuantFormat::Q8_0),
            Self::Q8_1 => Some(QuantFormat::Q8_1),
            Self::Q2K => Some(QuantFormat::Q2K),
            Self::Q3K => Some(QuantFormat::Q3K),
            Self::Q4K => Some(QuantFormat::Q4K),
            Self::Q5K => Some(QuantFormat::Q5K),
            Self::Q6K => Some(QuantFormat::Q6K),
            Self::Q8K => Some(QuantFormat::Q8K),
            _ => None,
        }
    }
}

/// GGUF metadata value types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_roundtrip() {
        assert_eq!(GgmlType::from_u32(0), Some(GgmlType::F32));
        assert_eq!(GgmlType::from_u32(2), Some(GgmlType::Q4_0));
        assert_eq!(GgmlType::from_u32(30), Some(GgmlType::BF16));
        assert_eq!(GgmlType::from_u32(999), None);
    }

    #[test]
    fn test_block_sizes() {
        assert_eq!(GgmlType::F32.block_size(), 1);
        assert_eq!(GgmlType::Q4_0.block_size(), 32);
        assert_eq!(GgmlType::Q4K.block_size(), 256);
    }

    #[test]
    fn test_is_quantized() {
        assert!(!GgmlType::F32.is_quantized());
        assert!(GgmlType::Q4_0.is_quantized());
        assert!(GgmlType::Q6K.is_quantized());
    }

    #[test]
    fn test_quant_format_mapping() {
        assert_eq!(GgmlType::Q4_0.to_quant_format(), Some(QuantFormat::Q4_0));
        assert_eq!(GgmlType::Q8_0.to_quant_format(), Some(QuantFormat::Q8_0));
        assert_eq!(GgmlType::F32.to_quant_format(), None);
    }
}
