//! [`TcfEncoding`]: a TCF native quantized encoding, as a runtime descriptor.
//!
//! A TCF payload is NOT block-structured the way [`crate::quant::QuantFormat`]
//! describes. `QuantFormat` states one flat `block_size`/`block_bytes` pair and
//! packs blocks contiguously along the last axis. TCF instead stores whole
//! planes over the whole tensor — every tile's codes, then every group's
//! scales, then every group's minima, then every super-block's super-scale and
//! super-minimum (SPECIFICATION.md Section 14, Section 14.5, Section 14.6) —
//! and two of its encodings carry a second scale level whose sub-fields are
//! bit-packed per 4-tile super-block.
//!
//! So this type does not restate any of that. It holds the encoding
//! identifier, and every size question is answered by `tcf-core`'s own
//! [`QuantLayout`], which is the crate that defines the layout. Nothing here
//! holds a plane offset, a bit position, or a byte count: a second copy of a
//! block layout is exactly what this format exists to prevent (MIGRATION.md
//! Section 4.5.3).

use tcf_core::{Encoding, NativeEncoding, QuantLayout, tile_count};

use crate::error::{Error, Result};
use crate::format::tcf::{encoding_name, tcf_error};

/// A TCF native quantized encoding.
///
/// Wraps [`NativeEncoding`] so the runtime's quantized-weight abstraction
/// carries one boostr-owned type, and so the size math below has one home.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TcfEncoding(NativeEncoding);

impl TcfEncoding {
    /// Wrap a `tcf-core` encoding identifier.
    #[must_use]
    pub const fn new(encoding: NativeEncoding) -> Self {
        Self(encoding)
    }

    /// The wrapped `tcf-core` encoding identifier.
    #[must_use]
    pub const fn native(self) -> NativeEncoding {
        self.0
    }

    /// This encoding's full on-disk layout: geometry plus scale form.
    /// Section 12.1, Section 13.3, Section 13.4.
    #[must_use]
    pub fn layout(self) -> QuantLayout {
        self.0.layout()
    }

    /// The spec's name for this encoding, e.g. `"Q6S16D_T64"`. Section 12.
    #[must_use]
    pub fn name(self) -> String {
        encoding_name(Encoding::Native(self.0))
    }

    /// Execution tile width, in logical elements. Always 64 in v1.
    #[must_use]
    pub fn tile(self) -> usize {
        usize::from(self.0.geometry().tile)
    }

    /// Number of execution tiles a tensor of `shape` occupies. Section 12.3.
    ///
    /// Tiles never cross a row boundary, so the last dimension alone decides
    /// tile placement. `tcf-core` owns the formula.
    ///
    /// # Errors
    /// [`Error::QuantError`] when the rank is outside `1..=8`, the last
    /// dimension is not a multiple of the tile width, or the count overflows.
    pub fn tile_count(self, shape: &[usize]) -> Result<u64> {
        let dims = dims_u64(shape)?;
        let rank = u32::try_from(dims.len()).map_err(|_| Error::QuantError {
            reason: format!("{}: rank {} exceeds u32", self.name(), dims.len()),
        })?;
        tile_count(&dims, rank, self.0.geometry().tile)
            .map_err(|e| tcf_error(&format!("{} tile count", self.name()), e))
    }

    /// Exact payload bytes a tensor of `shape` occupies, alignment padding
    /// excluded. Section 8.0.1, Section 12.2, Section 13.3, Section 13.4.
    ///
    /// A partial trailing super-block is charged in full, which is why this
    /// cannot be `numel / block_size * block_bytes`.
    ///
    /// # Errors
    /// Every error [`Self::tile_count`] raises, plus an overflow inside
    /// `tcf-core`'s span arithmetic.
    pub fn payload_bytes(self, shape: &[usize]) -> Result<usize> {
        let tiles = self.tile_count(shape)?;
        let bytes = self
            .layout()
            .logical_payload_bytes(tiles)
            .map_err(|e| tcf_error(&format!("{} payload bytes", self.name()), e))?;
        usize::try_from(bytes).map_err(|_| Error::QuantError {
            reason: format!("{}: payload of {bytes} bytes exceeds usize", self.name()),
        })
    }
}

impl From<NativeEncoding> for TcfEncoding {
    fn from(encoding: NativeEncoding) -> Self {
        Self(encoding)
    }
}

/// A row-major shape widened to the `u64` dimensions `tcf-core` takes.
fn dims_u64(shape: &[usize]) -> Result<Vec<u64>> {
    if shape.is_empty() {
        return Err(Error::QuantError {
            reason: "TCF tensor shape must be non-empty".into(),
        });
    }
    shape
        .iter()
        .map(|d| {
            u64::try_from(*d).map_err(|_| Error::QuantError {
                reason: format!("TCF tensor dimension {d} exceeds u64"),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Section 12.2: a flat encoding spends its whole budget per tile, so
    /// the payload is exactly `tiles * logical_bytes_per_tile`.
    #[test]
    fn a_flat_encodings_payload_is_a_whole_number_of_tiles() {
        let encoding = TcfEncoding::new(NativeEncoding::Q4S32T64);
        assert_eq!(encoding.tile_count(&[3, 320]).expect("tiles"), 15);
        assert_eq!(encoding.payload_bytes(&[3, 320]).expect("bytes"), 15 * 36);
    }

    /// Section 13.3: a partial trailing super-block is charged in full, so
    /// 15 tiles cost four super-scales, not `15 / 4`.
    #[test]
    fn a_partial_trailing_super_block_is_charged_in_full() {
        let encoding = TcfEncoding::new(NativeEncoding::Q6S16DT64);
        assert_eq!(encoding.tile_count(&[3, 320]).expect("tiles"), 15);
        // 15 tiles * 52 per-tile bytes + 4 super-blocks * 2 super-scale bytes.
        assert_eq!(
            encoding.payload_bytes(&[3, 320]).expect("bytes"),
            15 * 52 + 8
        );
    }

    /// Section 13.4: both sub-planes and both super values are charged per
    /// super-block, so the trailing partial block costs a whole 16 bytes.
    #[test]
    fn the_two_level_asymmetric_payload_matches_section_13_4() {
        let encoding = TcfEncoding::new(NativeEncoding::Q4AS32DT64);
        assert_eq!(
            encoding.payload_bytes(&[3, 320]).expect("bytes"),
            15 * 32 + 4 * 16
        );
        // Whole super-blocks reproduce Q4_K's 144 bytes per 256 elements.
        assert_eq!(encoding.payload_bytes(&[1, 256]).expect("bytes"), 144);
    }

    #[test]
    fn a_row_width_that_is_not_a_whole_number_of_tiles_is_rejected() {
        let encoding = TcfEncoding::new(NativeEncoding::Q4S32T64);
        let err = encoding.tile_count(&[2, 100]).expect_err("rejects");
        assert!(err.to_string().contains("E_"), "{err}");
    }

    #[test]
    fn an_empty_shape_is_rejected() {
        let encoding = TcfEncoding::new(NativeEncoding::Q8S32T64);
        assert!(encoding.tile_count(&[]).is_err());
    }

    #[test]
    fn the_name_comes_from_the_one_encoding_name_table() {
        assert_eq!(
            TcfEncoding::new(NativeEncoding::Q4AS32DT64).name(),
            "Q4AS32D_T64"
        );
    }
}
