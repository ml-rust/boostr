//! `QuantFormat` identifiers: the GGML type id, the CUDA kernel format id,
//! and the format name.

use super::QuantFormat;
use crate::error::{Error, Result};

impl QuantFormat {
    /// GGML type ID for GGUF compatibility
    pub const fn ggml_type_id(self) -> u32 {
        match self {
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::IQ2XXS => 16,
            Self::IQ2XS => 17,
            Self::IQ3XXS => 18,
            Self::IQ1S => 19,
            Self::IQ4NL => 20,
            Self::IQ3S => 21,
            Self::IQ2S => 22,
            Self::IQ4XS => 23,
            // ggml.h: 24=I8, 25=I16, 26=I32, 27=I64, 28=F64 are non-quant types, skipped here.
            Self::IQ1M => 29,
            Self::TQ1_0 => 34,
            Self::TQ2_0 => 35,
            // Not assigned by upstream ggml; the value is the one the files carry.
            Self::Q1_0 => 41,
            Self::Q2_0 => 42,
            Self::PQ2_0 => 142,
            Self::PTQ1_0 => 143,
        }
    }

    /// Internal format ID for generic CUDA kernel dispatch.
    /// Must match FMT_* constants in dequant_generic.cu.
    pub const fn format_id(self) -> u32 {
        match self {
            Self::Q4_0 => 0,
            Self::Q4_1 => 1,
            Self::Q5_0 => 2,
            Self::Q5_1 => 3,
            Self::Q8_0 => 4,
            Self::Q8_1 => 5,
            Self::Q2K => 6,
            Self::Q3K => 7,
            Self::Q4K => 8,
            Self::Q5K => 9,
            Self::Q6K => 10,
            Self::Q8K => 11,
            Self::IQ1S => 12,
            Self::IQ1M => 13,
            Self::IQ2XXS => 14,
            Self::IQ2XS => 15,
            Self::IQ2S => 16,
            Self::IQ3XXS => 17,
            Self::IQ3S => 18,
            Self::IQ4NL => 19,
            Self::IQ4XS => 20,
            Self::TQ1_0 => 21,
            Self::TQ2_0 => 22,
            Self::PQ2_0 => 23,
            Self::PTQ1_0 => 24,
            Self::Q1_0 => 25,
            Self::Q2_0 => 26,
        }
    }

    /// Construct from GGML type ID
    pub fn from_ggml_type_id(id: u32) -> Result<Self> {
        match id {
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            16 => Ok(Self::IQ2XXS),
            17 => Ok(Self::IQ2XS),
            18 => Ok(Self::IQ3XXS),
            19 => Ok(Self::IQ1S),
            20 => Ok(Self::IQ4NL),
            21 => Ok(Self::IQ3S),
            22 => Ok(Self::IQ2S),
            23 => Ok(Self::IQ4XS),
            29 => Ok(Self::IQ1M),
            34 => Ok(Self::TQ1_0),
            35 => Ok(Self::TQ2_0),
            41 => Ok(Self::Q1_0),
            42 => Ok(Self::Q2_0),
            142 => Ok(Self::PQ2_0),
            143 => Ok(Self::PTQ1_0),
            _ => Err(Error::UnsupportedQuantFormat {
                format: format!("GGML type ID {}", id),
            }),
        }
    }

    /// Format name string
    pub const fn name(self) -> &'static str {
        match self {
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::IQ1S => "IQ1_S",
            Self::IQ1M => "IQ1_M",
            Self::IQ2XXS => "IQ2_XXS",
            Self::IQ2XS => "IQ2_XS",
            Self::IQ2S => "IQ2_S",
            Self::IQ3XXS => "IQ3_XXS",
            Self::IQ3S => "IQ3_S",
            Self::IQ4NL => "IQ4_NL",
            Self::IQ4XS => "IQ4_XS",
            Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0",
            Self::PQ2_0 => "PQ2_0",
            Self::PTQ1_0 => "PTQ1_0",
            Self::Q1_0 => "Q1_0",
            Self::Q2_0 => "Q2_0",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_roundtrip() {
        let formats = [
            QuantFormat::Q4_0,
            QuantFormat::Q4_1,
            QuantFormat::Q5_0,
            QuantFormat::Q5_1,
            QuantFormat::Q8_0,
            QuantFormat::Q8_1,
            QuantFormat::Q2K,
            QuantFormat::Q3K,
            QuantFormat::Q4K,
            QuantFormat::Q5K,
            QuantFormat::Q6K,
            QuantFormat::Q8K,
            QuantFormat::IQ1S,
            QuantFormat::IQ1M,
            QuantFormat::IQ2XXS,
            QuantFormat::IQ2XS,
            QuantFormat::IQ2S,
            QuantFormat::IQ3XXS,
            QuantFormat::IQ3S,
            QuantFormat::IQ4NL,
            QuantFormat::IQ4XS,
            QuantFormat::TQ1_0,
            QuantFormat::TQ2_0,
            QuantFormat::PQ2_0,
            QuantFormat::PTQ1_0,
            QuantFormat::Q1_0,
            QuantFormat::Q2_0,
        ];
        for fmt in &formats {
            let id = fmt.ggml_type_id();
            let recovered = QuantFormat::from_ggml_type_id(id).unwrap();
            assert_eq!(
                *fmt, recovered,
                "roundtrip failed for {:?} (id={})",
                fmt, id
            );
        }
    }

    #[test]
    fn test_from_ggml_unknown() {
        assert!(QuantFormat::from_ggml_type_id(999).is_err());
    }

    #[test]
    fn test_ggml_type_id_24_is_not_iq1m() {
        // Regression: id 24 is GGML_TYPE_I8 (ggml.h), not IQ1_M (which is 29).
        // A prior bug wrote 24 for IQ1_M, mislabeling IQ1_M tensors as int8.
        assert!(QuantFormat::from_ggml_type_id(24).is_err());
        assert_eq!(QuantFormat::IQ1M.ggml_type_id(), 29);
    }

    #[test]
    fn test_ggml_type_id_29_is_iq1m() {
        assert_eq!(
            QuantFormat::from_ggml_type_id(29).unwrap(),
            QuantFormat::IQ1M
        );
    }

    /// `#define FMT_<NAME>` spelling in the CUDA kernels, for the variants
    /// where it drops the `_` that `name()` keeps (K-quants, I-quants).
    /// All other variants use `name()` unchanged.
    fn cuda_define_name(fmt: QuantFormat) -> String {
        match fmt {
            QuantFormat::Q2K
            | QuantFormat::Q3K
            | QuantFormat::Q4K
            | QuantFormat::Q5K
            | QuantFormat::Q6K
            | QuantFormat::Q8K
            | QuantFormat::IQ1S
            | QuantFormat::IQ1M
            | QuantFormat::IQ2XXS
            | QuantFormat::IQ2XS
            | QuantFormat::IQ2S
            | QuantFormat::IQ3XXS
            | QuantFormat::IQ3S
            | QuantFormat::IQ4NL
            | QuantFormat::IQ4XS => fmt.name().replace('_', ""),
            other => other.name().to_string(),
        }
    }

    /// Guards against Rust/CUDA drift: a format added in Rust without a
    /// matching `FMT_*` id in `format_ids.cuh` leaves CUDA output
    /// uninitialized. Both `.cu` files include the same header, so checking
    /// it once covers both.
    #[test]
    fn test_format_ids_match_cuda_kernels() {
        let format_ids = include_str!("../cuda/kernels/format_ids.cuh");
        let formats = [
            QuantFormat::Q4_0,
            QuantFormat::Q4_1,
            QuantFormat::Q5_0,
            QuantFormat::Q5_1,
            QuantFormat::Q8_0,
            QuantFormat::Q8_1,
            QuantFormat::Q2K,
            QuantFormat::Q3K,
            QuantFormat::Q4K,
            QuantFormat::Q5K,
            QuantFormat::Q6K,
            QuantFormat::Q8K,
            QuantFormat::IQ1S,
            QuantFormat::IQ1M,
            QuantFormat::IQ2XXS,
            QuantFormat::IQ2XS,
            QuantFormat::IQ2S,
            QuantFormat::IQ3XXS,
            QuantFormat::IQ3S,
            QuantFormat::IQ4NL,
            QuantFormat::IQ4XS,
            QuantFormat::TQ1_0,
            QuantFormat::TQ2_0,
            QuantFormat::PQ2_0,
            QuantFormat::PTQ1_0,
            QuantFormat::Q1_0,
            QuantFormat::Q2_0,
        ];
        for fmt in formats {
            let needle = format!("#define FMT_{} {}", cuda_define_name(fmt), fmt.format_id());
            let found = format_ids.lines().any(|line| {
                let mut parts = line.split_whitespace();
                parts.next() == Some("#define")
                    && parts.next() == Some(&format!("FMT_{}", cuda_define_name(fmt)))
                    && parts.next() == Some(&fmt.format_id().to_string())
            });
            assert!(
                found,
                "{} missing or mismatched in format_ids.cuh: {:?}",
                needle, fmt
            );
        }
    }
}
