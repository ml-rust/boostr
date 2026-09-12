//! `Encoding`: the unified TCF encoding identifier — a raw dtype or a GGML
//! block encoding. See `FORMAT.md`.

use crate::tcf::encoding::block::BlockEncoding;
use crate::tcf::encoding::raw::RawEncoding;
use crate::tcf::error::TcfError;

/// The `TensorRecord.encoding` / `ModuleRecord.preferred_encoding` /
/// `ModuleRecord.fallback_encoding` value space: raw dtypes occupy
/// `0x0001`-`0x00FF`, GGML block encodings `0x0200`-`0x02FF`. The range
/// `0x0100`-`0x01FF` held TCF's own tile encodings; they were removed after
/// measurement found no advantage over the GGML blocks at any bit width, and
/// the range stays unassigned so an old file is refused by name rather than
/// misread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Encoding {
    Raw(RawEncoding),
    /// A GGML block layout carried verbatim; TCF wraps it, never decodes it.
    Block(BlockEncoding),
}

impl Encoding {
    /// Encode as the wire `u16` identifier.
    #[must_use]
    pub const fn to_u16(self) -> u16 {
        match self {
            Self::Raw(r) => r.to_u16(),
            Self::Block(b) => b.to_u16(),
        }
    }

    /// Whether this encoding stores quantized values and therefore carries a
    /// semantic digest and a proof vector. Raw encodings store literal
    /// values and carry neither.
    #[must_use]
    pub const fn is_quantized(self) -> bool {
        matches!(self, Self::Block(_))
    }
}

impl TryFrom<u16> for Encoding {
    type Error = TcfError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        if let Ok(raw) = RawEncoding::try_from(value) {
            return Ok(Self::Raw(raw));
        }
        if let Ok(block) = BlockEncoding::try_from(value) {
            return Ok(Self::Block(block));
        }
        Err(TcfError::UnsupportedEncoding { raw: value })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_raw_encodings() {
        assert_eq!(
            Encoding::try_from(0x0001),
            Ok(Encoding::Raw(RawEncoding::F32))
        );
        assert_eq!(
            Encoding::try_from(0x0015),
            Ok(Encoding::Raw(RawEncoding::U32))
        );
        assert_eq!(Encoding::Raw(RawEncoding::F16).to_u16(), 0x0002);
        assert!(!Encoding::Raw(RawEncoding::F32).is_quantized());
    }

    #[test]
    fn resolves_block_encodings() {
        assert_eq!(
            Encoding::try_from(0x0200 + 12),
            Ok(Encoding::Block(BlockEncoding::Q4K))
        );
        assert!(Encoding::Block(BlockEncoding::Q4K).is_quantized());
    }

    #[test]
    fn rejects_unassigned_values_including_the_retired_native_range() {
        for raw in [0x0000u16, 0x0104, 0x0107, 0x0126, 0x0200, 0x0300] {
            assert_eq!(
                Encoding::try_from(raw),
                Err(TcfError::UnsupportedEncoding { raw })
            );
        }
    }
}
