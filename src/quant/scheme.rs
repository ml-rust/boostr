//! `QuantScheme`: which codec a [`crate::quant::QuantTensor`]'s packed bytes
//! follow, and how those bytes are addressed.
//!
//! One quantized-weight abstraction carries every quantized codec the runtime
//! executes. A second tensor type per codec would force a second `DequantOps`,
//! a second `QuantMatmulOps`, and a second path through `Linear`, `VarBuilder`,
//! and every inference server built on boostr.
//!
//! The two codecs address their bytes differently, and the difference is not
//! cosmetic:
//!
//! | | GGUF | TCF |
//! | --- | --- | --- |
//! | unit | fixed block | 64-element tile |
//! | packing | blocks contiguous along the last axis | whole planes over the whole tensor |
//! | scales | inside each block | separate plane, one or two levels |
//! | trailing unit | always whole | a partial super-block is charged in full |
//!
//! So a scheme is an enum, not a wider `QuantFormat`. Adding TCF arms to
//! `QuantFormat` would give every consumer of its documented "blocks packed
//! along the last axis" contract a variant that violates it — silently, in
//! row-stride arithmetic, which is the class of bug that has shipped wrong
//! weights from this codebase before.

use crate::error::{Error, Result};
use crate::quant::format::QuantFormat;
use crate::quant::tcf::TcfEncoding;

/// The codec and byte addressing of a quantized payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantScheme {
    /// GGUF/GGML block quantization. Blocks are packed contiguously along the
    /// last axis, and each block carries its own scales.
    Gguf(QuantFormat),
    /// TCF native quantization. Planes span the whole tensor, and a two-level
    /// encoding's sub-scales are bit-packed per 4-tile super-block.
    Tcf(TcfEncoding),
}

impl QuantScheme {
    /// The codec's name for this format, e.g. `"Q4_K"` or `"Q6S16D_T64"`.
    #[must_use]
    pub fn name(self) -> String {
        match self {
            Self::Gguf(format) => format.name().to_string(),
            Self::Tcf(encoding) => encoding.name(),
        }
    }

    /// The GGUF block format, or `None` for any other codec.
    #[must_use]
    pub const fn gguf(self) -> Option<QuantFormat> {
        match self {
            Self::Gguf(format) => Some(format),
            Self::Tcf(_) => None,
        }
    }

    /// The TCF encoding, or `None` for any other codec.
    #[must_use]
    pub const fn tcf(self) -> Option<TcfEncoding> {
        match self {
            Self::Tcf(encoding) => Some(encoding),
            Self::Gguf(_) => None,
        }
    }

    /// `true` when whole rows of this scheme's payload are contiguous byte
    /// runs, so a row gather is a byte gather.
    ///
    /// GGUF answers `true`; TCF answers `false`, because its planes span the
    /// whole tensor and a row's codes, scales, and super values live in four
    /// separate places.
    #[must_use]
    pub const fn is_row_blocked(self) -> bool {
        matches!(self, Self::Gguf(_))
    }

    /// Exact packed bytes a tensor of `shape` occupies under this scheme.
    ///
    /// # Errors
    /// [`Error::QuantError`] when `shape` is empty, or when the scheme's own
    /// divisibility rule rejects it.
    pub fn payload_bytes(self, shape: &[usize]) -> Result<usize> {
        match self {
            Self::Gguf(format) => {
                let numel: usize = shape.iter().product();
                format.storage_bytes(numel)
            }
            Self::Tcf(encoding) => encoding.payload_bytes(shape),
        }
    }

    /// Independently decodable units in a tensor of `shape`: a GGUF block, or
    /// a TCF execution tile.
    ///
    /// Both divisions are exact once [`Self::validate`] has passed, so this is
    /// infallible; an unvalidated shape yields a truncated count rather than a
    /// panic.
    #[must_use]
    pub fn decode_units(self, shape: &[usize]) -> usize {
        let numel: usize = shape.iter().product();
        match self {
            Self::Gguf(format) => numel.checked_div(format.block_size()).unwrap_or(0),
            Self::Tcf(encoding) => numel.checked_div(encoding.tile()).unwrap_or(0),
        }
    }

    /// Check `shape` and a candidate payload length against this scheme.
    ///
    /// # Errors
    /// [`Error::QuantError`] when `shape` is empty, when the last dimension is
    /// not a whole number of blocks (GGUF) or tiles (TCF), or when
    /// `payload_len` disagrees with [`Self::payload_bytes`].
    pub fn validate(self, shape: &[usize], payload_len: usize) -> Result<()> {
        if shape.is_empty() {
            return Err(Error::QuantError {
                reason: "QuantTensor shape must be non-empty".into(),
            });
        }
        let last_dim = shape.last().copied().unwrap_or(0);
        let unit = match self {
            Self::Gguf(format) => format.block_size(),
            Self::Tcf(encoding) => encoding.tile(),
        };
        if unit == 0 || !last_dim.is_multiple_of(unit) {
            return Err(Error::QuantError {
                reason: format!(
                    "last dimension {last_dim} is not a multiple of {}'s unit width {unit}",
                    self.name(),
                ),
            });
        }

        let expected = self.payload_bytes(shape)?;
        if payload_len != expected {
            return Err(Error::QuantError {
                reason: format!(
                    "expected {expected} bytes for {} with shape {shape:?}, got {payload_len} bytes",
                    self.name(),
                ),
            });
        }
        Ok(())
    }
}

impl From<QuantFormat> for QuantScheme {
    fn from(format: QuantFormat) -> Self {
        Self::Gguf(format)
    }
}

impl From<TcfEncoding> for QuantScheme {
    fn from(encoding: TcfEncoding) -> Self {
        Self::Tcf(encoding)
    }
}

impl std::fmt::Display for QuantScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tcf_core::NativeEncoding;

    #[test]
    fn a_gguf_scheme_keeps_quant_formats_own_block_arithmetic() {
        let scheme = QuantScheme::from(QuantFormat::Q4K);
        assert_eq!(
            scheme.payload_bytes(&[4096, 4096]).expect("bytes"),
            4096 * 16 * 144
        );
        assert_eq!(scheme.decode_units(&[4096, 4096]), 4096 * 16);
        assert!(scheme.is_row_blocked());
        assert_eq!(scheme.gguf(), Some(QuantFormat::Q4K));
        assert_eq!(scheme.tcf(), None);
    }

    #[test]
    fn a_tcf_scheme_is_not_row_blocked() {
        let scheme = QuantScheme::from(TcfEncoding::new(NativeEncoding::Q6S16DT64));
        assert!(!scheme.is_row_blocked());
        assert_eq!(scheme.gguf(), None);
        assert_eq!(scheme.name(), "Q6S16D_T64");
    }

    /// The whole reason TCF cannot be a `QuantFormat` arm: 15 tiles is not a
    /// whole number of 256-element super-blocks, yet the tensor is legal and
    /// its payload is exactly the size below.
    #[test]
    fn a_tcf_shape_with_a_partial_super_block_validates() {
        let scheme = QuantScheme::from(TcfEncoding::new(NativeEncoding::Q6S16DT64));
        let bytes = scheme.payload_bytes(&[3, 320]).expect("bytes");
        assert_eq!(bytes, 15 * 52 + 8);
        scheme.validate(&[3, 320], bytes).expect("validates");
    }

    #[test]
    fn a_wrong_payload_length_names_both_sizes() {
        let scheme = QuantScheme::from(QuantFormat::Q4_0);
        let err = scheme.validate(&[32], 17).expect_err("rejects");
        let text = err.to_string();
        assert!(text.contains("18"), "{text}");
        assert!(text.contains("17"), "{text}");
    }

    #[test]
    fn a_last_dimension_that_is_not_a_whole_unit_is_rejected() {
        assert!(
            QuantScheme::from(QuantFormat::Q4_0)
                .validate(&[33], 18)
                .is_err()
        );
        let tcf = QuantScheme::from(TcfEncoding::new(NativeEncoding::Q4S32T64));
        assert!(tcf.validate(&[3, 100], 0).is_err());
    }

    #[test]
    fn an_empty_shape_is_rejected() {
        assert!(
            QuantScheme::from(QuantFormat::Q4_0)
                .validate(&[], 18)
                .is_err()
        );
    }
}
