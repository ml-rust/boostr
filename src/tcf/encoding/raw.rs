//! Raw (unquantized) encodings. FORMAT.md Section 12. Identifiers occupy
//! `0x0001`-`0x00FF`.

use crate::tcf::error::TcfError;

/// A raw, unquantized on-disk encoding. Decode is a table lookup, never
/// arithmetic derived from the identifier (Section 12: "a convention for
/// assignment, never a decoder").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RawEncoding {
    F32,
    F16,
    Bf16,
    /// 8-bit float, no block scale.
    F8E4M3,
    /// 8-bit float, no block scale.
    F8E5M2,
    I8,
    I16,
    I32,
    U8,
    U16,
    U32,
}

impl RawEncoding {
    /// Encode as the wire `u16` identifier.
    #[must_use]
    pub const fn to_u16(self) -> u16 {
        match self {
            Self::F32 => 0x0001,
            Self::F16 => 0x0002,
            Self::Bf16 => 0x0003,
            Self::F8E4M3 => 0x0004,
            Self::F8E5M2 => 0x0005,
            Self::I8 => 0x0010,
            Self::I16 => 0x0011,
            Self::I32 => 0x0012,
            Self::U8 => 0x0013,
            Self::U16 => 0x0014,
            Self::U32 => 0x0015,
        }
    }

    /// This encoding's element size in bytes. Section 8.0.1.
    ///
    /// A raw tensor's `logical_payload_bytes` is `product(dims) * width`,
    /// determined by the shape and the encoding and never declared, so this
    /// is the only place the width lives: the format stores no raw dtype
    /// width, and a reader that guessed one could not reject a truncated or
    /// padded payload.
    #[must_use]
    pub const fn width_bytes(self) -> u64 {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::Bf16 | Self::I16 | Self::U16 => 2,
            Self::F8E4M3 | Self::F8E5M2 | Self::I8 | Self::U8 => 1,
        }
    }
}

impl TryFrom<u16> for RawEncoding {
    type Error = TcfError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0x0001 => Ok(Self::F32),
            0x0002 => Ok(Self::F16),
            0x0003 => Ok(Self::Bf16),
            0x0004 => Ok(Self::F8E4M3),
            0x0005 => Ok(Self::F8E5M2),
            0x0010 => Ok(Self::I8),
            0x0011 => Ok(Self::I16),
            0x0012 => Ok(Self::I32),
            0x0013 => Ok(Self::U8),
            0x0014 => Ok(Self::U16),
            0x0015 => Ok(Self::U32),
            other => Err(TcfError::UnsupportedEncoding { raw: other }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_every_variant() {
        for v in [
            RawEncoding::F32,
            RawEncoding::F16,
            RawEncoding::Bf16,
            RawEncoding::F8E4M3,
            RawEncoding::F8E5M2,
            RawEncoding::I8,
            RawEncoding::I16,
            RawEncoding::I32,
            RawEncoding::U8,
            RawEncoding::U16,
            RawEncoding::U32,
        ] {
            assert_eq!(RawEncoding::try_from(v.to_u16()), Ok(v));
        }
    }

    #[test]
    fn identifiers_match_spec() {
        assert_eq!(RawEncoding::F32.to_u16(), 0x0001);
        assert_eq!(RawEncoding::F16.to_u16(), 0x0002);
        assert_eq!(RawEncoding::Bf16.to_u16(), 0x0003);
        assert_eq!(RawEncoding::F8E4M3.to_u16(), 0x0004);
        assert_eq!(RawEncoding::F8E5M2.to_u16(), 0x0005);
        assert_eq!(RawEncoding::I8.to_u16(), 0x0010);
        assert_eq!(RawEncoding::I16.to_u16(), 0x0011);
        assert_eq!(RawEncoding::I32.to_u16(), 0x0012);
        assert_eq!(RawEncoding::U8.to_u16(), 0x0013);
        assert_eq!(RawEncoding::U16.to_u16(), 0x0014);
        assert_eq!(RawEncoding::U32.to_u16(), 0x0015);
    }

    /// Section 8.0.1: the width table, every variant, verbatim.
    #[test]
    fn width_bytes_matches_section_8_0_1() {
        assert_eq!(RawEncoding::F32.width_bytes(), 4);
        assert_eq!(RawEncoding::I32.width_bytes(), 4);
        assert_eq!(RawEncoding::U32.width_bytes(), 4);
        assert_eq!(RawEncoding::F16.width_bytes(), 2);
        assert_eq!(RawEncoding::Bf16.width_bytes(), 2);
        assert_eq!(RawEncoding::I16.width_bytes(), 2);
        assert_eq!(RawEncoding::U16.width_bytes(), 2);
        assert_eq!(RawEncoding::F8E4M3.width_bytes(), 1);
        assert_eq!(RawEncoding::F8E5M2.width_bytes(), 1);
        assert_eq!(RawEncoding::I8.width_bytes(), 1);
        assert_eq!(RawEncoding::U8.width_bytes(), 1);
    }

    #[test]
    fn rejects_gaps_and_out_of_range() {
        assert!(RawEncoding::try_from(0x0000).is_err());
        assert!(RawEncoding::try_from(0x0006).is_err());
        assert!(RawEncoding::try_from(0x000f).is_err());
        assert!(RawEncoding::try_from(0x0016).is_err());
        assert!(RawEncoding::try_from(0x0100).is_err());
    }
}
