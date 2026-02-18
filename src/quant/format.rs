//! Quantization format definitions (GGUF-compatible)
//!
//! Each format defines a fixed block structure. All invariants are encoded in the enum.
//!
//! # Packing axis contract (CRITICAL for dequant/quant_matmul agreement)
//!
//! Quantization is applied along the LAST axis (innermost, contiguous in memory).
//! For a weight matrix `[out_features, in_features]`, blocks run along `in_features`.
//! Example: `[4096, 4096]` with Q4_K (block_size=256) → each row = 16 blocks.

use crate::error::{Error, Result};
use std::fmt;

/// Quantization formats (GGUF-compatible)
///
/// Invariants enforced by QuantFormat:
/// - `block_size`: number of logical elements per block (always a power of 2)
/// - `block_bytes`: exact byte count per block (fixed, not variable)
/// - Logical element count MUST be a multiple of `block_size`
/// - Block data is tightly packed — no inter-block padding
/// - Byte order: little-endian (matches GGUF spec)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    // Simple blocks (32 elements per block)
    /// 4-bit with single scale. block_size=32, block_bytes=18
    Q4_0,
    /// 4-bit with scale and minimum. block_size=32, block_bytes=20
    Q4_1,
    /// 5-bit with single scale. block_size=32, block_bytes=22
    Q5_0,
    /// 5-bit with scale and minimum. block_size=32, block_bytes=24
    Q5_1,
    /// 8-bit with single scale. block_size=32, block_bytes=34
    Q8_0,
    /// 8-bit with scale and sum. block_size=32, block_bytes=36
    Q8_1,

    // K-quants (256 elements per super-block)
    /// K-quant 2-bit. block_size=256, block_bytes=84
    Q2K,
    /// K-quant 3-bit. block_size=256, block_bytes=110
    Q3K,
    /// K-quant 4-bit. block_size=256, block_bytes=144
    Q4K,
    /// K-quant 5-bit. block_size=256, block_bytes=176
    Q5K,
    /// K-quant 6-bit. block_size=256, block_bytes=210
    Q6K,
    /// K-quant 8-bit. block_size=256, block_bytes=292
    Q8K,

    // I-quants (grid-based, 256 elements per super-block)
    /// 1.5625 bpw using 2048-entry grid. block_size=256, block_bytes=50
    IQ1S,
    /// 1.75 bpw with per-block 3-bit scales. block_size=256, block_bytes=56
    IQ1M,
    /// True 2-bit with 256-entry grid. block_size=256, block_bytes=66
    IQ2XXS,
    /// 2.3125 bpw with explicit sub-block scales. block_size=256, block_bytes=74
    IQ2XS,
    /// 2.5625 bpw with separate high bits. block_size=256, block_bytes=82
    IQ2S,
    /// True 3-bit with 256-entry grid. block_size=256, block_bytes=98
    IQ3XXS,
    /// 3.4375 bpw with sub-block scales. block_size=256, block_bytes=110
    IQ3S,
    /// Non-linear 4-bit with learned codebook. block_size=32, block_bytes=18
    IQ4NL,
    /// 4.25 bpw with super-block structure. block_size=256, block_bytes=136
    IQ4XS,

    // TQ-quants (ternary, 256 elements per block)
    /// Tied-scale ternary (-1, 0, +1). block_size=256, block_bytes=54
    TQ1_0,
    /// Tied-scale 2-bit. block_size=256, block_bytes=66
    TQ2_0,
}

impl QuantFormat {
    /// Number of logical elements per block
    pub const fn block_size(self) -> usize {
        match self {
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::IQ4NL => 32,
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Q8K
            | Self::IQ1S
            | Self::IQ1M
            | Self::IQ2XXS
            | Self::IQ2XS
            | Self::IQ2S
            | Self::IQ3XXS
            | Self::IQ3S
            | Self::IQ4XS
            | Self::TQ1_0
            | Self::TQ2_0 => 256,
        }
    }

    /// Exact byte count per block
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            Self::IQ1S => 50,
            Self::IQ1M => 56,
            Self::IQ2XXS => 66,
            Self::IQ2XS => 74,
            Self::IQ2S => 82,
            Self::IQ3XXS => 98,
            Self::IQ3S => 110,
            Self::IQ4NL => 18,
            Self::IQ4XS => 136,
            Self::TQ1_0 => 54,
            Self::TQ2_0 => 66,
        }
    }

    /// Total storage bytes for `numel` logical elements.
    ///
    /// Returns `Err` if `numel` is not a multiple of `block_size`.
    /// Uses `Result` (not panic) because `numel` may come from untrusted GGUF metadata.
    pub fn storage_bytes(self, numel: usize) -> Result<usize> {
        let bs = self.block_size();
        if numel % bs != 0 {
            return Err(Error::QuantError {
                reason: format!(
                    "{}: element count {} is not a multiple of block_size {}",
                    self.name(),
                    numel,
                    bs,
                ),
            });
        }
        Ok((numel / bs) * self.block_bytes())
    }

    /// Number of blocks for `numel` logical elements.
    pub fn num_blocks(self, numel: usize) -> Result<usize> {
        let bs = self.block_size();
        if numel % bs != 0 {
            return Err(Error::QuantError {
                reason: format!(
                    "{}: element count {} is not a multiple of block_size {}",
                    self.name(),
                    numel,
                    bs,
                ),
            });
        }
        Ok(numel / bs)
    }

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
            Self::IQ1M => 24,
            Self::TQ1_0 => 34,
            Self::TQ2_0 => 35,
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
            24 => Ok(Self::IQ1M),
            34 => Ok(Self::TQ1_0),
            35 => Ok(Self::TQ2_0),
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
        }
    }
}

impl fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_sizes() {
        assert_eq!(QuantFormat::Q4_0.block_size(), 32);
        assert_eq!(QuantFormat::Q4K.block_size(), 256);
        assert_eq!(QuantFormat::IQ4NL.block_size(), 32);
        assert_eq!(QuantFormat::TQ1_0.block_size(), 256);
    }

    #[test]
    fn test_block_bytes() {
        assert_eq!(QuantFormat::Q4_0.block_bytes(), 18);
        assert_eq!(QuantFormat::Q4K.block_bytes(), 144);
        assert_eq!(QuantFormat::Q6K.block_bytes(), 210);
        assert_eq!(QuantFormat::Q8K.block_bytes(), 292);
        assert_eq!(QuantFormat::IQ4NL.block_bytes(), 18);
    }

    #[test]
    fn test_storage_bytes() {
        // Q4_0: 32 elements per block, 18 bytes per block
        assert_eq!(QuantFormat::Q4_0.storage_bytes(32).unwrap(), 18);
        assert_eq!(QuantFormat::Q4_0.storage_bytes(64).unwrap(), 36);
        assert_eq!(QuantFormat::Q4_0.storage_bytes(1024).unwrap(), 576);

        // Q4K: 256 elements per block, 144 bytes per block
        assert_eq!(QuantFormat::Q4K.storage_bytes(256).unwrap(), 144);
        assert_eq!(QuantFormat::Q4K.storage_bytes(4096).unwrap(), 2304);
    }

    #[test]
    fn test_storage_bytes_alignment_error() {
        assert!(QuantFormat::Q4_0.storage_bytes(33).is_err());
        assert!(QuantFormat::Q4K.storage_bytes(100).is_err());
    }

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
    fn test_num_blocks() {
        assert_eq!(QuantFormat::Q4_0.num_blocks(32).unwrap(), 1);
        assert_eq!(QuantFormat::Q4_0.num_blocks(1024).unwrap(), 32);
        assert_eq!(QuantFormat::Q4K.num_blocks(4096).unwrap(), 16);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", QuantFormat::Q4K), "Q4_K");
        assert_eq!(format!("{}", QuantFormat::IQ2XXS), "IQ2_XXS");
    }
}
