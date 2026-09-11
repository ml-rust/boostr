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
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
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
            16 => Some(Self::IQ2XXS),
            17 => Some(Self::IQ2XS),
            18 => Some(Self::IQ3XXS),
            19 => Some(Self::IQ1S),
            20 => Some(Self::IQ4NL),
            21 => Some(Self::IQ3S),
            22 => Some(Self::IQ2S),
            23 => Some(Self::IQ4XS),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::IQ1M),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Number of elements per block.
    ///
    /// Quantized variants delegate to `QuantFormat` — the block layout lives
    /// there once, never restated here (see `QuantFormat::block_size`).
    pub fn block_size(&self) -> usize {
        match self.to_quant_format() {
            Some(fmt) => fmt.block_size(),
            None => 1, // F32/F16/BF16/F64/I8/I16/I32/I64: unblocked, 1 element each.
        }
    }

    /// Bytes per block.
    ///
    /// Quantized variants delegate to `QuantFormat::block_bytes` — see
    /// `block_size` above for why these numbers are never duplicated here.
    pub fn block_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Q8K
            | Self::IQ2XXS
            | Self::IQ2XS
            | Self::IQ3XXS
            | Self::IQ1S
            | Self::IQ4NL
            | Self::IQ3S
            | Self::IQ2S
            | Self::IQ4XS
            | Self::IQ1M => self
                .to_quant_format()
                .map(|fmt| fmt.block_bytes())
                .unwrap_or(0), // Unreachable: every arm here has a QuantFormat (see to_quant_format).
        }
    }

    pub fn is_quantized(&self) -> bool {
        self.to_quant_format().is_some()
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
            Self::IQ2XXS => Some(QuantFormat::IQ2XXS),
            Self::IQ2XS => Some(QuantFormat::IQ2XS),
            Self::IQ3XXS => Some(QuantFormat::IQ3XXS),
            Self::IQ1S => Some(QuantFormat::IQ1S),
            Self::IQ4NL => Some(QuantFormat::IQ4NL),
            Self::IQ3S => Some(QuantFormat::IQ3S),
            Self::IQ2S => Some(QuantFormat::IQ2S),
            Self::IQ4XS => Some(QuantFormat::IQ4XS),
            Self::IQ1M => Some(QuantFormat::IQ1M),
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::F64
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64 => None,
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

    /// Every `GgmlType` round-trips through its numeric id (defect 2 regression:
    /// the IQ variants previously had no ids at all, so `from_u32` returned `None`).
    #[test]
    fn test_all_ggml_types_roundtrip() {
        let types = [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q8_0,
            GgmlType::Q8_1,
            GgmlType::Q2K,
            GgmlType::Q3K,
            GgmlType::Q4K,
            GgmlType::Q5K,
            GgmlType::Q6K,
            GgmlType::Q8K,
            GgmlType::IQ2XXS,
            GgmlType::IQ2XS,
            GgmlType::IQ3XXS,
            GgmlType::IQ1S,
            GgmlType::IQ4NL,
            GgmlType::IQ3S,
            GgmlType::IQ2S,
            GgmlType::IQ4XS,
            GgmlType::I8,
            GgmlType::I16,
            GgmlType::I32,
            GgmlType::I64,
            GgmlType::F64,
            GgmlType::IQ1M,
            GgmlType::BF16,
        ];
        for ty in types {
            let id = ty as u32;
            assert_eq!(
                GgmlType::from_u32(id),
                Some(ty),
                "roundtrip failed for {ty:?} (id={id})"
            );
        }
    }

    /// Regression for defect 1: ggml.h id 24 is `I8`, never an IQ format.
    #[test]
    fn test_ggml_type_24_is_i8_not_iq1m() {
        assert_eq!(GgmlType::from_u32(24), Some(GgmlType::I8));
    }

    /// ggml.h: `GGML_TYPE_IQ1_M = 29`.
    #[test]
    fn test_ggml_type_29_is_iq1m() {
        assert_eq!(GgmlType::from_u32(29), Some(GgmlType::IQ1M));
    }

    /// Every quantized `GgmlType` maps to a `QuantFormat` and back, at the same id.
    #[test]
    fn test_quant_format_roundtrip_all_iq() {
        let pairs = [
            (GgmlType::IQ2XXS, QuantFormat::IQ2XXS),
            (GgmlType::IQ2XS, QuantFormat::IQ2XS),
            (GgmlType::IQ3XXS, QuantFormat::IQ3XXS),
            (GgmlType::IQ1S, QuantFormat::IQ1S),
            (GgmlType::IQ4NL, QuantFormat::IQ4NL),
            (GgmlType::IQ3S, QuantFormat::IQ3S),
            (GgmlType::IQ2S, QuantFormat::IQ2S),
            (GgmlType::IQ4XS, QuantFormat::IQ4XS),
            (GgmlType::IQ1M, QuantFormat::IQ1M),
        ];
        for (gt, qf) in pairs {
            assert_eq!(gt.to_quant_format(), Some(qf));
            assert_eq!(gt as u32, qf.ggml_type_id());
        }
    }

    /// Block sizes must agree between `GgmlType` and `QuantFormat` for every
    /// quantized format shared between them — the two must never drift apart.
    #[test]
    fn test_block_sizes_agree_with_quant_format() {
        let types = [
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q8_0,
            GgmlType::Q8_1,
            GgmlType::Q2K,
            GgmlType::Q3K,
            GgmlType::Q4K,
            GgmlType::Q5K,
            GgmlType::Q6K,
            GgmlType::Q8K,
            GgmlType::IQ2XXS,
            GgmlType::IQ2XS,
            GgmlType::IQ3XXS,
            GgmlType::IQ1S,
            GgmlType::IQ4NL,
            GgmlType::IQ3S,
            GgmlType::IQ2S,
            GgmlType::IQ4XS,
            GgmlType::IQ1M,
        ];
        for ty in types {
            let fmt = ty.to_quant_format().expect("quantized type must map");
            assert_eq!(
                ty.block_size(),
                fmt.block_size(),
                "block_size mismatch for {ty:?}"
            );
            assert_eq!(
                ty.block_bytes(),
                fmt.block_bytes(),
                "block_bytes mismatch for {ty:?}"
            );
        }
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
