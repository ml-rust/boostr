//! The `QuantFormat` enum, its block geometry (`block_size`,
//! `block_bytes`, storage and block counts), and its `Display`.

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

    // Lowbit quants: wide-block, sub-2-bit formats not defined by upstream ggml
    /// 1-bit sign, one f16 scale. block_size=128, block_bytes=18
    Q1_0,
    /// 2-bit codes 0..3 map to -1..+2, one f16 scale. block_size=64, block_bytes=18
    Q2_0,
    /// Q2_0 codec at group 128. block_size=128, block_bytes=34
    PQ2_0,
    /// TQ1_0 trit packing at group 128. block_size=128, block_bytes=28
    PTQ1_0,
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
            Self::Q2_0 => 64,
            Self::Q1_0 | Self::PQ2_0 | Self::PTQ1_0 => 128,
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
            Self::Q1_0 => 18,
            Self::Q2_0 => 18,
            Self::PQ2_0 => 34,
            Self::PTQ1_0 => 28,
        }
    }

    /// Total storage bytes for `numel` logical elements.
    ///
    /// Returns `Err` if `numel` is not a multiple of `block_size`.
    /// Uses `Result` (not panic) because `numel` may come from untrusted GGUF metadata.
    pub fn storage_bytes(self, numel: usize) -> Result<usize> {
        let bs = self.block_size();
        if !numel.is_multiple_of(bs) {
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
        if !numel.is_multiple_of(bs) {
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
        assert_eq!(QuantFormat::PQ2_0.block_size(), 128);
        assert_eq!(QuantFormat::PTQ1_0.block_size(), 128);
        assert_eq!(QuantFormat::Q1_0.block_size(), 128);
        assert_eq!(QuantFormat::Q2_0.block_size(), 64);
    }

    #[test]
    fn test_block_bytes() {
        assert_eq!(QuantFormat::Q4_0.block_bytes(), 18);
        assert_eq!(QuantFormat::Q4K.block_bytes(), 144);
        assert_eq!(QuantFormat::Q6K.block_bytes(), 210);
        assert_eq!(QuantFormat::Q8K.block_bytes(), 292);
        assert_eq!(QuantFormat::IQ4NL.block_bytes(), 18);
        assert_eq!(QuantFormat::PQ2_0.block_bytes(), 34);
        assert_eq!(QuantFormat::PTQ1_0.block_bytes(), 28);
        assert_eq!(QuantFormat::Q1_0.block_bytes(), 18);
        assert_eq!(QuantFormat::Q2_0.block_bytes(), 18);
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
    fn test_num_blocks() {
        assert_eq!(QuantFormat::Q4_0.num_blocks(32).unwrap(), 1);
        assert_eq!(QuantFormat::Q4_0.num_blocks(1024).unwrap(), 32);
        assert_eq!(QuantFormat::Q4K.num_blocks(4096).unwrap(), 16);
        assert_eq!(QuantFormat::PQ2_0.num_blocks(4096).unwrap(), 32);
        assert_eq!(QuantFormat::PTQ1_0.num_blocks(128).unwrap(), 1);
        assert!(QuantFormat::PTQ1_0.num_blocks(64).is_err());
        assert_eq!(QuantFormat::Q1_0.num_blocks(256).unwrap(), 2);
        assert_eq!(QuantFormat::Q2_0.num_blocks(256).unwrap(), 4);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", QuantFormat::Q4K), "Q4_K");
        assert_eq!(format!("{}", QuantFormat::IQ2XXS), "IQ2_XXS");
        assert_eq!(format!("{}", QuantFormat::PQ2_0), "PQ2_0");
        assert_eq!(format!("{}", QuantFormat::PTQ1_0), "PTQ1_0");
        assert_eq!(format!("{}", QuantFormat::Q1_0), "Q1_0");
        assert_eq!(format!("{}", QuantFormat::Q2_0), "Q2_0");
    }
}
