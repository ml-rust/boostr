//! Plane offsets and geometry of one TCF payload, as CUDA kernel arguments.
//!
//! # Why the offsets are computed here and not in the kernel
//!
//! A CUDA kernel cannot call `tcf-core`, so the read-direction bit positions
//! had to be written a second time in `kernels/tcf.cuh`. Everything that did
//! NOT have to be is kept here instead: plane order, plane sizes, and the byte
//! offset of each plane are read off `tcf-core`'s own [`QuantLayout`], exactly
//! as `quant::cpu::kernels::tcf`'s gather does, and handed to the kernel as
//! numbers.
//!
//! So a layout change — a new scale form, a plane that grows — reaches the
//! device as different arguments rather than as a constant that silently went
//! stale. MIGRATION.md Section 4.5.3 forbids a second copy of a block layout,
//! and this is how much of that copy can be avoided when the consumer is a GPU.

use tcf_core::{QuantLayout, ScaleForm};

use crate::error::{Error, Result};
use crate::format::tcf::tcf_error;
use crate::quant::TcfEncoding;

/// `ScaleForm::Flat` as the kernel's discriminant. Mirrors `TCF_SCALE_FLAT`.
const SCALE_FORM_FLAT: u32 = 0;
/// `ScaleForm::TwoLevelU8` as the kernel's discriminant.
const SCALE_FORM_TWO_LEVEL_U8: u32 = 1;
/// `ScaleForm::TwoLevelU6M6` as the kernel's discriminant.
const SCALE_FORM_TWO_LEVEL_U6M6: u32 = 2;

/// The execution tile width every v1 encoding uses, in logical elements.
///
/// `tcf-core` fixes it: `checked_groups_per_tile` rejects any other width
/// because `Code64` holds exactly 64 codes. The kernels assume it too, so
/// [`TcfLaunchArgs::new`] refuses anything else rather than launching a kernel
/// whose tile strides no longer describe the payload.
const TILE: usize = 64;

/// Everything a TCF CUDA kernel needs about one payload's layout.
///
/// Every field is either read off [`QuantLayout`] or derived from the shape.
/// None is a hard-coded byte count.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TcfLaunchArgs {
    /// Execution tiles in the tensor.
    pub tiles: u64,
    /// Byte offset of the 6-bit high-two-bit code sub-plane (Section 14.2).
    /// Zero for a 4-bit or 8-bit encoding, which has one code sub-plane.
    pub code_high_off: u64,
    /// Byte offset of the scale plane.
    pub scale_off: u64,
    /// Byte offset of the minimum plane.
    pub min_off: u64,
    /// Byte offset of the super-scale plane.
    pub super_off: u64,
    /// Byte offset of the super-minimum plane.
    pub super_min_off: u64,
    /// Bits per code: 4, 6, or 8.
    pub bits: u32,
    /// Quantization group width, in logical elements.
    pub group: u32,
    /// Groups per execution tile: `tile / group`, so 1, 2, or 4.
    pub groups_per_tile: u32,
    /// `1` for a symmetric encoding, `0` for an asymmetric one.
    pub symmetric: u32,
    /// The scale form's kernel discriminant.
    pub scale_form: u32,
    /// Bytes one super-block's bit-packed sub-scale plane occupies.
    pub sub_block_bytes: u32,
}

impl TcfLaunchArgs {
    /// Derive the launch arguments for `encoding` over a tensor of `shape`.
    ///
    /// # Errors
    /// [`Error::QuantError`] when the tile width is not 64, when a plane span
    /// exceeds `u64`, or when `shape` is not tileable. [`Error::ModelError`]
    /// carrying the spec's `E_*` code when `tcf-core`'s own span arithmetic
    /// rejects the tile count.
    pub(crate) fn new(encoding: TcfEncoding, shape: &[usize]) -> Result<Self> {
        let layout = encoding.layout();
        let geometry = layout.geometry;
        let name = encoding.name();

        if usize::from(geometry.tile) != TILE {
            return Err(Error::QuantError {
                reason: format!(
                    "{name}: the CUDA kernels decode a {TILE}-element tile, layout states {}",
                    geometry.tile
                ),
            });
        }
        let groups_per_tile = geometry
            .checked_groups_per_tile()
            .map_err(|e| tcf_error(&format!("{name} group count"), e))?;
        if !matches!(geometry.bits, 4 | 6 | 8) {
            return Err(Error::QuantError {
                reason: format!(
                    "{name}: Section 14 defines no {}-bit code packing",
                    geometry.bits
                ),
            });
        }

        let tiles = encoding.tile_count(shape)?;
        let plane_bytes = |bytes: std::result::Result<u64, tcf_core::TcfError>| -> Result<u64> {
            bytes.map_err(|e| tcf_error(&format!("{name} plane bytes"), e))
        };

        let code_off = tiles
            .checked_mul(u64::from(geometry.code_bytes_per_tile()))
            .ok_or_else(|| overflow(&name))?;
        // Section 14.2: the low-nibble sub-plane is `tile / 2` bytes per tile
        // and the high-two-bit sub-plane follows it whole.
        let code_high_off = if geometry.bits == 6 {
            tiles
                .checked_mul((TILE / 2) as u64)
                .ok_or_else(|| overflow(&name))?
        } else {
            0
        };

        let scale_off = code_off;
        let min_off = scale_off
            .checked_add(plane_bytes(layout.scale_plane_bytes(tiles))?)
            .ok_or_else(|| overflow(&name))?;
        let super_off = min_off
            .checked_add(plane_bytes(layout.min_plane_bytes(tiles))?)
            .ok_or_else(|| overflow(&name))?;
        let super_min_off = super_off
            .checked_add(plane_bytes(layout.super_scale_bytes(tiles))?)
            .ok_or_else(|| overflow(&name))?;

        Ok(Self {
            tiles,
            code_high_off,
            scale_off,
            min_off,
            super_off,
            super_min_off,
            bits: u32::from(geometry.bits),
            group: u32::from(geometry.group),
            groups_per_tile: u32::from(groups_per_tile),
            symmetric: u32::from(geometry.symmetric),
            scale_form: scale_form_id(layout),
            sub_block_bytes: layout.sub_scale_bytes_per_block(),
        })
    }
}

/// The kernel discriminant of a layout's scale form.
fn scale_form_id(layout: QuantLayout) -> u32 {
    match layout.scale_form {
        ScaleForm::Flat => SCALE_FORM_FLAT,
        ScaleForm::TwoLevelU8 => SCALE_FORM_TWO_LEVEL_U8,
        ScaleForm::TwoLevelU6M6 => SCALE_FORM_TWO_LEVEL_U6M6,
    }
}

/// A plane span exceeded `u64`.
fn overflow(name: &str) -> Error {
    Error::QuantError {
        reason: format!("{name}: TCF plane offset overflows u64"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tcf_core::NativeEncoding;

    /// The offsets must reproduce the plane order Section 14 fixes: codes,
    /// scales, minima, super-scales, super-minima. Checked against the
    /// layout's own totals rather than against restated byte counts.
    #[test]
    fn every_encodings_offsets_partition_its_payload() {
        for native in [
            NativeEncoding::Q4S32T64,
            NativeEncoding::Q4AS32T64,
            NativeEncoding::Q4AS64T64,
            NativeEncoding::Q6S32T64,
            NativeEncoding::Q8S32T64,
            NativeEncoding::Q6S16DT64,
            NativeEncoding::Q4AS32DT64,
        ] {
            let encoding = TcfEncoding::new(native);
            // 15 tiles: three whole super-blocks and a partial fourth.
            let shape = [3usize, 320];
            let args = TcfLaunchArgs::new(encoding, &shape).expect("args");
            let layout = encoding.layout();
            assert_eq!(args.tiles, 15, "{native:?}");

            let total = encoding.payload_bytes(&shape).expect("bytes") as u64;
            let super_min_bytes = layout.super_min_bytes(args.tiles).expect("super min");
            assert_eq!(args.super_min_off + super_min_bytes, total, "{native:?}");
            assert!(args.scale_off <= args.min_off, "{native:?}");
            assert!(args.min_off <= args.super_off, "{native:?}");
            assert!(args.super_off <= args.super_min_off, "{native:?}");
        }
    }

    /// A 6-bit encoding's high sub-plane starts after the whole low sub-plane
    /// and ends where the scale plane begins. Section 14.2.
    #[test]
    fn a_six_bit_code_plane_splits_into_two_whole_sub_planes() {
        let encoding = TcfEncoding::new(NativeEncoding::Q6S32T64);
        let args = TcfLaunchArgs::new(encoding, &[3, 320]).expect("args");
        assert_eq!(args.code_high_off, 15 * 32);
        assert_eq!(args.scale_off, 15 * 48);
    }

    /// A 4-bit or 8-bit code plane is one sub-plane, so the high offset is
    /// never read and stays zero.
    #[test]
    fn a_four_or_eight_bit_code_plane_has_no_high_sub_plane() {
        for native in [NativeEncoding::Q4S32T64, NativeEncoding::Q8S32T64] {
            let args = TcfLaunchArgs::new(TcfEncoding::new(native), &[3, 320]).expect("args");
            assert_eq!(args.code_high_off, 0, "{native:?}");
        }
    }

    /// The two-level asymmetric form spends both sub-planes per super-block,
    /// which is the one field the bit-packed reader is addressed by.
    #[test]
    fn the_two_level_asymmetric_sub_planes_are_six_bytes_per_super_block() {
        let args = TcfLaunchArgs::new(TcfEncoding::new(NativeEncoding::Q4AS32DT64), &[3, 320])
            .expect("args");
        assert_eq!(args.scale_form, SCALE_FORM_TWO_LEVEL_U6M6);
        assert_eq!(args.sub_block_bytes, 6);
        assert_eq!(args.groups_per_tile, 2);
        assert_eq!(args.symmetric, 0);
    }

    #[test]
    fn a_row_width_that_is_not_a_whole_tile_is_refused() {
        let encoding = TcfEncoding::new(NativeEncoding::Q4S32T64);
        assert!(TcfLaunchArgs::new(encoding, &[2, 100]).is_err());
    }
}
