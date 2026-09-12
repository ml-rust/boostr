//! Block encodings: GGML's block-quantized layouts carried as TCF payloads.
//! Identifiers occupy `0x0200`-`0x02FF`.
//!
//! A block-encoded payload is the GGML block stream for the tensor, row-major,
//! byte for byte what a GGUF file stores for the same `ggml_type`: every row
//! is `dims[rank-1] / block_elems` consecutive blocks of `block_bytes`. The
//! element layout inside a block is GGML's (`ggml-common.h`), which TCF
//! neither restates nor decodes. What TCF adds around the bytes is the
//! container: the mandatory activation contract, the module and calibration
//! records, the digests, and the proof vector a reader checks with its own
//! block decoder.
//!
//! The identifier's low byte is the `ggml_type` number, so a GGUF tensor
//! maps to its TCF encoding without a table. That is a convention for
//! assignment: decode is still the table lookup below, never arithmetic on
//! the identifier.

use crate::tcf::error::TcfError;

/// Base of the block-encoding identifier range.
const BLOCK_BASE: u16 = 0x0200;

/// One GGML block layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockEncoding {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    Iq1M,
    Tq1_0,
    Tq2_0,
}

impl BlockEncoding {
    /// Every block encoding, in identifier order.
    pub const ALL: [Self; 23] = [
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q8_1,
        Self::Q2K,
        Self::Q3K,
        Self::Q4K,
        Self::Q5K,
        Self::Q6K,
        Self::Q8K,
        Self::Iq2Xxs,
        Self::Iq2Xs,
        Self::Iq3Xxs,
        Self::Iq1S,
        Self::Iq4Nl,
        Self::Iq3S,
        Self::Iq2S,
        Self::Iq4Xs,
        Self::Iq1M,
        Self::Tq1_0,
        Self::Tq2_0,
    ];

    /// The `ggml_type` number this layout carries in GGUF.
    #[must_use]
    pub const fn ggml_type(self) -> u16 {
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
            Self::Iq2Xxs => 16,
            Self::Iq2Xs => 17,
            Self::Iq3Xxs => 18,
            Self::Iq1S => 19,
            Self::Iq4Nl => 20,
            Self::Iq3S => 21,
            Self::Iq2S => 22,
            Self::Iq4Xs => 23,
            Self::Iq1M => 29,
            Self::Tq1_0 => 34,
            Self::Tq2_0 => 35,
        }
    }

    /// Encode as the wire `u16` identifier: `0x0200 + ggml_type`.
    #[must_use]
    pub const fn to_u16(self) -> u16 {
        BLOCK_BASE + self.ggml_type()
    }

    /// Elements per block.
    #[must_use]
    pub const fn block_elems(self) -> u64 {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Iq4Nl => 32,
            _ => 256,
        }
    }

    /// Bytes per block, `ggml-common.h`'s `sizeof(block_*)`.
    #[must_use]
    pub const fn block_bytes(self) -> u64 {
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
            Self::Iq2Xxs => 66,
            Self::Iq2Xs => 74,
            Self::Iq3Xxs => 98,
            Self::Iq1S => 50,
            Self::Iq4Nl => 18,
            Self::Iq3S => 110,
            Self::Iq2S => 82,
            Self::Iq4Xs => 136,
            Self::Iq1M => 56,
            Self::Tq1_0 => 54,
            Self::Tq2_0 => 66,
        }
    }

    /// GGML's name for the layout, as GGUF tooling prints it.
    #[must_use]
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
            Self::Iq2Xxs => "IQ2_XXS",
            Self::Iq2Xs => "IQ2_XS",
            Self::Iq3Xxs => "IQ3_XXS",
            Self::Iq1S => "IQ1_S",
            Self::Iq4Nl => "IQ4_NL",
            Self::Iq3S => "IQ3_S",
            Self::Iq2S => "IQ2_S",
            Self::Iq4Xs => "IQ4_XS",
            Self::Iq1M => "IQ1_M",
            Self::Tq1_0 => "TQ1_0",
            Self::Tq2_0 => "TQ2_0",
        }
    }

    /// Payload bytes for a tensor of `shape` (the leading `rank` dims).
    ///
    /// # Errors
    /// - [`TcfError::InvalidQuantShape`] if `rank < 2` or the row width is
    ///   not a whole number of blocks.
    /// - [`TcfError::TileArithmeticOverflow`] if the length overflows `u64`.
    pub fn payload_bytes(self, shape: &[u64], rank: u32, tensor_id: u32) -> Result<u64, TcfError> {
        if rank < 2 || shape.len() < 2 {
            return Err(TcfError::InvalidQuantShape { tensor_id });
        }
        let row = shape[shape.len() - 1];
        if row == 0 || !row.is_multiple_of(self.block_elems()) {
            return Err(TcfError::InvalidQuantShape { tensor_id });
        }
        let blocks_per_row = row / self.block_elems();
        let mut rows: u64 = 1;
        for dim in &shape[..shape.len() - 1] {
            rows = rows
                .checked_mul(*dim)
                .ok_or(TcfError::TileArithmeticOverflow)?;
        }
        rows.checked_mul(blocks_per_row)
            .and_then(|blocks| blocks.checked_mul(self.block_bytes()))
            .ok_or(TcfError::TileArithmeticOverflow)
    }
}

impl TryFrom<u16> for BlockEncoding {
    type Error = TcfError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        // Table lookup, never `value - BLOCK_BASE` interpreted as a type.
        Self::ALL
            .into_iter()
            .find(|e| e.to_u16() == value)
            .ok_or(TcfError::UnsupportedEncoding { raw: value })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_are_unique_and_in_range() {
        let mut ids: Vec<u16> = BlockEncoding::ALL.iter().map(|e| e.to_u16()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), BlockEncoding::ALL.len());
        assert!(ids.iter().all(|id| (0x0200..=0x02FF).contains(id)));
    }

    #[test]
    fn roundtrips_through_u16() {
        for e in BlockEncoding::ALL {
            assert_eq!(BlockEncoding::try_from(e.to_u16()), Ok(e));
        }
        assert!(BlockEncoding::try_from(0x0200).is_err());
        assert!(BlockEncoding::try_from(0x0218).is_err());
        assert!(BlockEncoding::try_from(0x0300).is_err());
    }

    #[test]
    fn q4_k_payload_matches_gguf_arithmetic() {
        // [4096, 4096] Q4_K: 16 blocks of 144 bytes per row.
        assert_eq!(
            BlockEncoding::Q4K.payload_bytes(&[4096, 4096], 2, 7),
            Ok(4096 * 16 * 144)
        );
        // 32-element blocks with a row of 64.
        assert_eq!(
            BlockEncoding::Q8_0.payload_bytes(&[3, 64], 2, 7),
            Ok(3 * 2 * 34)
        );
    }

    #[test]
    fn row_width_must_be_whole_blocks() {
        assert_eq!(
            BlockEncoding::Q4K.payload_bytes(&[8, 200], 2, 9),
            Err(TcfError::InvalidQuantShape { tensor_id: 9 })
        );
        assert_eq!(
            BlockEncoding::Q4K.payload_bytes(&[256], 1, 9),
            Err(TcfError::InvalidQuantShape { tensor_id: 9 })
        );
    }
}
