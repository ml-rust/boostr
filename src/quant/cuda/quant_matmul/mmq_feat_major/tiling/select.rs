//! Picks the (feature tile, token tile, cadence) triple a shape runs at —
//! the tiling rule this whole module exists to apply. Validates the forced
//! variants and their errors here; the automatic policy for
//! [`FeatTile::Auto`] is `super::auto::select_auto_tiling`. The token tile at
//! one feature tile comes from `variant::select_variant`.

use crate::error::{Error, Result};

use super::super::formats::FeatMajorFormat;
use super::FeatTile;
use super::auto::select_auto_tiling;
use super::geometry::{
    Cadence, FEAT_TILE_DEFAULT, FEAT_TILE_NARROW, FEAT_TILE_SMALL, GROUP_CADENCE_MIN_X,
    SMALL_VARIANTS, group_compiled,
};
use super::tile::Tiling;
use super::variant::select_variant;

/// The tiling rule. `Ok(None)` means no compiled variant fits the device.
///
/// A batch the single-warp tile covers in one token tile takes it, on the
/// formats that compile it: the decode regime, where a block's K walk is a
/// serial chain and the tile count is what fills the device. Every feature
/// tile forms the same bits, so this reads only `m`.
///
/// Otherwise the default feature tile is tried first and is the answer for
/// every format without the narrow tile. Then the narrow tile's own token
/// tile for `m` decides which lever is in play:
///
/// - Short token tile (below [`GROUP_CADENCE_MIN_X`]): only the two-half
///   cadence exists, with the default tile's activation footprint, so the
///   narrow tile is pure block count. It is taken exactly when the default
///   tiling is starved: fewer output tiles than two waves of SMs, the bound
///   `use_split_launch` draws. Below it the tile-parallel grid leaves SMs
///   idle for the whole K walk, and the split-K pair can only split what few
///   tiles there are.
/// - Wide token tile: per-group activation staging dominates, and the grid
///   picks the cadence. Under one block per SM nothing co-resides, so the
///   full-group cadence's halved barriers are pure gain. At or above it the
///   two-half cadence keeps the smaller activation tile and the
///   co-residency the group cadence would spend. The grid counted is the one
///   launched: tiles times the split count when the split-K pair runs. It
///   is taken while the default tiling is still starved; past starvation
///   the default tile stays.
///
/// The starvation test runs on the default tiling's own token tile; the
/// narrow tiling re-selects its token tile for `m`. `k` enters only through
/// the split count, which reads `k`, `n` and `sms` and never the tiling, so
/// the cadence choice cannot feed back into it. `prefers_tile_parallel` is
/// the format's measured launch pick on this device
/// (`super::tile_parallel_tune::prefers_tile_parallel`); it reaches only the
/// launched-grid count of the cadence choice.
///
/// A forced tiling bypasses the rule. A forced tiling the format does not
/// compile is an error, not a silent fallback: a caller forcing a tiling is
/// measuring that tiling.
#[allow(clippy::too_many_arguments)]
pub(in crate::quant::cuda::quant_matmul::mmq_feat_major) fn select_tiling(
    m: u32,
    n: u32,
    k: u32,
    smem_limit: u32,
    sms: u32,
    format: &FeatMajorFormat,
    feat_tile: FeatTile,
    prefers_tile_parallel: bool,
) -> Result<Option<Tiling>> {
    let at = |tile: u32, cadence: Cadence| {
        select_variant(m, smem_limit, format, tile, cadence).map(|mmq_x| Tiling {
            feat_tile: tile,
            mmq_x,
            cadence,
        })
    };
    let not_compiled = |what: String, hint: &str| Error::QuantError {
        reason: format!(
            "MMQ {what} is not compiled for {}; {hint}",
            format.kernel_infix
        ),
    };
    let small_max = *SMALL_VARIANTS.last().unwrap_or(&0);
    match feat_tile {
        FeatTile::Force(tile)
            if tile != FEAT_TILE_DEFAULT && tile != FEAT_TILE_NARROW && tile != FEAT_TILE_SMALL =>
        {
            Err(not_compiled(
                format!("feature tile {tile}"),
                &format!("force {FEAT_TILE_DEFAULT}, {FEAT_TILE_NARROW} or {FEAT_TILE_SMALL}"),
            ))
        }
        FeatTile::Force(tile) if tile != FEAT_TILE_DEFAULT && !format.narrow_tile => {
            Err(not_compiled(
                format!("feature tile {tile}"),
                &format!("force {FEAT_TILE_DEFAULT} or use the automatic tile"),
            ))
        }
        FeatTile::Force(tile) if tile == FEAT_TILE_SMALL && m > small_max => Err(not_compiled(
            format!("feature tile {tile} past {small_max} tokens"),
            "it covers one token tile of at most that many; lower m",
        )),
        FeatTile::Force(tile) => Ok(at(tile, Cadence::Halves)),
        FeatTile::ForceNarrowGroup if !format.narrow_tile => Err(not_compiled(
            format!("feature tile {FEAT_TILE_NARROW}"),
            &format!("force {FEAT_TILE_DEFAULT} or use the automatic tile"),
        )),
        FeatTile::ForceNarrowGroup => {
            let Some(narrow) = at(FEAT_TILE_NARROW, Cadence::Group) else {
                return Ok(None);
            };
            if !group_compiled(narrow.feat_tile, narrow.mmq_x) {
                return Err(not_compiled(
                    format!(
                        "group cadence at feature tile {FEAT_TILE_NARROW} token tile {}",
                        narrow.mmq_x
                    ),
                    &format!("it exists from token tile {GROUP_CADENCE_MIN_X} up; raise m"),
                ));
            }
            Ok(Some(narrow))
        }
        FeatTile::Auto => Ok(select_auto_tiling(
            m,
            n,
            k,
            smem_limit,
            sms,
            format,
            prefers_tile_parallel,
        )),
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the forced-tiling paths; the automatic policy is
    //! tested in `auto`'s own test module.

    use super::super::super::formats::{Q4_K, Q6_K, Q8_0};
    use super::super::Cadence;
    use super::super::geometry::{FEAT_TILE_DEFAULT, FEAT_TILE_NARROW, FEAT_TILE_SMALL};
    use super::super::tile::Tiling;
    use super::*;

    /// Ample limit: every compiled variant fits.
    const WIDE: u32 = 1 << 20;

    /// A K under the split gate, so the geometry cases below see one range.
    const SHORT_K: u32 = 1024;

    /// SM count the geometry cases below are written against.
    const SMS: u32 = 28;

    /// The rule at the format's fallback launch pick, which is what the
    /// cases below were written against.
    fn pick(
        m: u32,
        n: u32,
        k: u32,
        sms: u32,
        format: &FeatMajorFormat,
        feat_tile: FeatTile,
    ) -> Result<Option<Tiling>> {
        select_tiling(
            m,
            n,
            k,
            WIDE,
            sms,
            format,
            feat_tile,
            format.prefers_tile_parallel_fallback,
        )
    }

    fn wide(mmq_x: u32) -> Tiling {
        Tiling {
            feat_tile: FEAT_TILE_DEFAULT,
            mmq_x,
            cadence: Cadence::Halves,
        }
    }

    fn narrow(mmq_x: u32) -> Tiling {
        Tiling {
            feat_tile: FEAT_TILE_NARROW,
            mmq_x,
            cadence: Cadence::Halves,
        }
    }

    fn group(mmq_x: u32) -> Tiling {
        Tiling {
            feat_tile: FEAT_TILE_NARROW,
            mmq_x,
            cadence: Cadence::Group,
        }
    }

    fn small(mmq_x: u32) -> Tiling {
        Tiling {
            feat_tile: FEAT_TILE_SMALL,
            mmq_x,
            cadence: Cadence::Halves,
        }
    }

    #[test]
    fn the_single_warp_tile_can_be_forced_within_its_range() {
        assert_eq!(
            pick(22, 1024, SHORT_K, SMS, &Q4_K, FeatTile::Force(16)).ok(),
            None
        );
        assert_eq!(
            pick(16, 1024, SHORT_K, SMS, &Q4_K, FeatTile::Force(16)).expect("tiling rule"),
            Some(small(16))
        );
        assert!(pick(8, 1024, SHORT_K, SMS, &Q8_0, FeatTile::Force(16)).is_err());
    }

    #[test]
    fn a_forced_tile_bypasses_the_rule() {
        // Forced default at a starved shape.
        assert_eq!(
            pick(22, 1024, SHORT_K, SMS, &Q4_K, FeatTile::Force(128)).expect("tiling rule"),
            Some(wide(24))
        );
        // Forced narrow at a shape that fills two waves.
        assert_eq!(
            pick(22, 8192, SHORT_K, SMS, &Q4_K, FeatTile::Force(64)).expect("tiling rule"),
            Some(narrow(24))
        );
        // Forced narrow on the two-half cadence where the automatic rule picks
        // the group cadence.
        assert_eq!(
            pick(44, 1024, SHORT_K, SMS, &Q6_K, FeatTile::Force(64)).expect("tiling rule"),
            Some(narrow(48))
        );
        // Forced group cadence where the automatic rule picks the two-half one.
        assert_eq!(
            pick(44, 4096, SHORT_K, SMS, &Q6_K, FeatTile::ForceNarrowGroup).expect("tiling rule"),
            Some(group(48))
        );
    }

    #[test]
    fn a_forced_tile_the_format_lacks_is_an_error() {
        assert!(pick(22, 1024, SHORT_K, SMS, &Q8_0, FeatTile::Force(64)).is_err());
        assert!(pick(22, 1024, SHORT_K, SMS, &Q4_K, FeatTile::Force(96)).is_err());
        assert!(pick(44, 1024, SHORT_K, SMS, &Q8_0, FeatTile::ForceNarrowGroup).is_err());
        // The group cadence exists only from x40 up: m=22 selects x24.
        assert!(pick(22, 1024, SHORT_K, SMS, &Q4_K, FeatTile::ForceNarrowGroup).is_err());
    }
}
