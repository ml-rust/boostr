//! Picks the (feature tile, token tile, cadence) triple a shape runs at —
//! the tiling rule this whole module exists to apply. The token tile at one
//! feature tile comes from `variant::select_variant`.

use crate::error::{Error, Result};

use super::super::formats::FeatMajorFormat;
use super::FeatTile;
use super::geometry::{
    Cadence, FEAT_TILE_DEFAULT, FEAT_TILE_NARROW, FEAT_TILE_SMALL, GROUP_CADENCE_MIN_X,
    SMALL_VARIANTS, group_compiled,
};
use super::split_k::{split_count, use_split_launch};
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
/// the cadence choice cannot feed back into it.
///
/// A forced tiling bypasses the rule. A forced tiling the format does not
/// compile is an error, not a silent fallback: a caller forcing a tiling is
/// measuring that tiling.
pub(in crate::quant::cuda::quant_matmul::mmq_feat_major) fn select_tiling(
    m: u32,
    n: u32,
    k: u32,
    smem_limit: u32,
    sms: u32,
    format: &FeatMajorFormat,
    feat_tile: FeatTile,
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
        FeatTile::Auto => {
            if format.narrow_tile
                && m <= small_max
                && let Some(small) = at(FEAT_TILE_SMALL, Cadence::Halves)
            {
                return Ok(Some(small));
            }
            let Some(wide) = at(FEAT_TILE_DEFAULT, Cadence::Halves) else {
                return Ok(None);
            };
            if !format.narrow_tile {
                return Ok(Some(wide));
            }
            let Some(narrow) = at(FEAT_TILE_NARROW, Cadence::Halves) else {
                return Ok(Some(wide));
            };
            let starved = wide.tiles(m, n) < 2 * sms;
            if group_compiled(narrow.feat_tile, narrow.mmq_x) {
                // Same token tile on the group cadence, if its doubled
                // activation tile still fits the device.
                let group = Tiling {
                    cadence: Cadence::Group,
                    ..narrow
                };
                let tiles = narrow.tiles(m, n);
                let splits = split_count(k, n, sms);
                let launched = if use_split_launch(splits, tiles, narrow.feat_tile, sms, format) {
                    tiles * splits
                } else {
                    tiles
                };
                if launched < sms && group.smem_bytes(format) <= smem_limit {
                    return Ok(Some(group));
                }
            }
            Ok(Some(if starved { narrow } else { wide }))
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the feature-major MMQ launch decisions.

    use super::super::super::formats::{IQ3_XXS, Q4_K, Q6_K, Q8_0};
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

    fn auto(m: u32, n: u32, format: &FeatMajorFormat) -> Option<Tiling> {
        select_tiling(m, n, SHORT_K, WIDE, SMS, format, FeatTile::Auto).expect("tiling rule")
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
    fn a_decode_batch_takes_the_single_warp_tile() {
        // One token tile of the single-warp tile covers the batch, whatever
        // the width: the rule reads `m` alone.
        assert_eq!(auto(1, 1024, &Q4_K), Some(small(8)));
        assert_eq!(auto(8, 64, &Q6_K), Some(small(8)));
        assert_eq!(auto(9, 8192, &Q4_K), Some(small(16)));
        assert_eq!(auto(16, 1024, &Q4_K), Some(small(16)));
        // Past its widest token tile the rule falls through to the others.
        assert_eq!(auto(17, 1024, &Q4_K), Some(narrow(24)));
        // A format without the tile never takes it.
        assert_eq!(auto(1, 1024, &Q8_0), Some(wide(8)));
    }

    #[test]
    fn the_single_warp_tile_can_be_forced_within_its_range() {
        assert_eq!(
            select_tiling(22, 1024, SHORT_K, WIDE, SMS, &Q4_K, FeatTile::Force(16)).ok(),
            None
        );
        assert_eq!(
            select_tiling(16, 1024, SHORT_K, WIDE, SMS, &Q4_K, FeatTile::Force(16))
                .expect("tiling rule"),
            Some(small(16))
        );
        assert!(select_tiling(8, 1024, SHORT_K, WIDE, SMS, &Q8_0, FeatTile::Force(16)).is_err());
    }

    #[test]
    fn a_starved_shape_with_a_short_token_tile_takes_the_narrow_tile() {
        // 22 tokens x 1024 features: one token tile by 8 feature tiles is 8
        // blocks, under two waves of SMS, and x24 keeps the staging chain
        // short. The narrow tile doubles the grid to 16.
        assert_eq!(auto(22, 1024, &Q4_K), Some(narrow(24)));
        // 22 x 256 at Q4_K: 2 default tiles, 4 narrow. Well under one wave
        // either way, and the short token tile still wins.
        assert_eq!(auto(22, 256, &Q4_K), Some(narrow(24)));
        // 22 x 4096: 32 default tiles is over one wave but under two, so the
        // default tiling is still starved, and x24 takes the narrow tile.
        assert_eq!(auto(22, 4096, &Q4_K), Some(narrow(24)));
    }

    #[test]
    fn a_wide_token_tile_picks_the_cadence_by_its_grid() {
        // 44 x 1024 at Q6_K: 8 default tiles, 16 narrow, under one per SM.
        // Nothing co-resides, so the group cadence's halved barriers are pure
        // gain.
        assert_eq!(auto(44, 1024, &Q6_K), Some(group(48)));
        // 44 x 256 at Q6_K: 2 default tiles, 4 narrow. Same verdict.
        assert_eq!(auto(44, 256, &Q6_K), Some(group(48)));
        // 44 x 2048 at Q6_K: the narrow grid is 32 x48 blocks, at least one per
        // SM, so the group cadence's doubled activation tile would cost the
        // co-residency; the default tiling's 16 tiles are still starved, so the
        // narrow tile stays on the two-half cadence.
        assert_eq!(auto(44, 2048, &Q6_K), Some(narrow(48)));
        // 44 x 4096 at Q6_K: 32 default tiles, under two waves, 64 narrow. Same
        // verdict.
        assert_eq!(auto(44, 4096, &Q6_K), Some(narrow(48)));
        // 44 x 8192 at Q6_K: 64 default tiles fill two waves; the default tile
        // stays.
        assert_eq!(auto(44, 8192, &Q6_K), Some(wide(48)));
    }

    #[test]
    fn a_shape_that_fills_two_waves_keeps_the_default_tile() {
        // 22 x 8192: 64 default tiles already exceed two waves of SMS.
        assert_eq!(auto(22, 8192, &Q4_K), Some(wide(24)));
        // Prefill-shaped: 512 tokens x 4096 features is 4 x 32 = 128 tiles.
        assert_eq!(auto(512, 4096, &Q4_K), Some(wide(128)));
    }

    #[test]
    fn the_starvation_test_uses_the_default_tiling_token_tile() {
        // 100 tokens x 1024 features: x112 covers the batch in one token tile,
        // so 8 default tiles are starved; the narrow tile then re-selects its
        // own widest token tile, x64, and its grid is 16 x 2 = 32 blocks. That
        // reaches one block per SM, so the two-half cadence is the one taken.
        assert_eq!(auto(100, 1024, &Q4_K), Some(narrow(64)));
        // 100 x 256: the narrow grid is 2 x 2 = 4 blocks, under the SM count, so
        // the re-selected x64 runs the group cadence.
        assert_eq!(auto(100, 256, &Q4_K), Some(group(64)));
    }

    #[test]
    fn a_format_without_the_narrow_tile_never_picks_it() {
        // The same starved geometry as the Q4_K case above.
        assert_eq!(auto(22, 1024, &Q8_0), Some(wide(24)));
        assert_eq!(auto(22, 256, &IQ3_XXS), Some(wide(24)));
    }

    #[test]
    fn no_sm_count_keeps_the_default_tile() {
        // A profile that reports no compute units cannot be starved.
        assert_eq!(
            select_tiling(22, 1024, SHORT_K, WIDE, 0, &Q4_K, FeatTile::Auto).expect("tiling rule"),
            Some(wide(24))
        );
    }

    #[test]
    fn a_forced_tile_bypasses_the_rule() {
        // Forced default at a starved shape.
        assert_eq!(
            select_tiling(22, 1024, SHORT_K, WIDE, SMS, &Q4_K, FeatTile::Force(128))
                .expect("tiling rule"),
            Some(wide(24))
        );
        // Forced narrow at a shape that fills two waves.
        assert_eq!(
            select_tiling(22, 8192, SHORT_K, WIDE, SMS, &Q4_K, FeatTile::Force(64))
                .expect("tiling rule"),
            Some(narrow(24))
        );
        // Forced narrow on the two-half cadence where the automatic rule picks
        // the group cadence.
        assert_eq!(
            select_tiling(44, 1024, SHORT_K, WIDE, SMS, &Q6_K, FeatTile::Force(64))
                .expect("tiling rule"),
            Some(narrow(48))
        );
        // Forced group cadence where the automatic rule picks the two-half one.
        assert_eq!(
            select_tiling(
                44,
                4096,
                SHORT_K,
                WIDE,
                SMS,
                &Q6_K,
                FeatTile::ForceNarrowGroup
            )
            .expect("tiling rule"),
            Some(group(48))
        );
    }

    #[test]
    fn a_forced_tile_the_format_lacks_is_an_error() {
        assert!(select_tiling(22, 1024, SHORT_K, WIDE, SMS, &Q8_0, FeatTile::Force(64)).is_err());
        assert!(select_tiling(22, 1024, SHORT_K, WIDE, SMS, &Q4_K, FeatTile::Force(96)).is_err());
        assert!(
            select_tiling(
                44,
                1024,
                SHORT_K,
                WIDE,
                SMS,
                &Q8_0,
                FeatTile::ForceNarrowGroup
            )
            .is_err()
        );
        // The group cadence exists only from x40 up: m=22 selects x24.
        assert!(
            select_tiling(
                22,
                1024,
                SHORT_K,
                WIDE,
                SMS,
                &Q4_K,
                FeatTile::ForceNarrowGroup
            )
            .is_err()
        );
    }

    #[test]
    fn the_group_cadence_counts_the_launched_grid() {
        // 44 x 1024, Q4_K: the narrow tile's 16 tiles at x48 sit under one
        // block per SM, so a one-range K takes the group cadence.
        let short = select_tiling(44, 1024, SHORT_K, WIDE, SMS, &Q4_K, FeatTile::Auto)
            .expect("tiling rule")
            .expect("a variant fits");
        assert_eq!(short.feat_tile, FEAT_TILE_NARROW);
        assert_eq!(short.cadence, Cadence::Group);
        // At K=4096 the split-K pair launches 16 x 4 blocks, which co-reside,
        // so the two-half cadence keeps its smaller activation tile.
        let long = select_tiling(44, 1024, 4096, WIDE, SMS, &Q4_K, FeatTile::Auto)
            .expect("tiling rule")
            .expect("a variant fits");
        assert_eq!(long.feat_tile, FEAT_TILE_NARROW);
        assert_eq!(long.mmq_x, short.mmq_x);
        assert_eq!(long.cadence, Cadence::Halves);
    }

    #[test]
    fn split_k_composes_with_the_narrow_tile() {
        // 22 x 1024 at K=4096, Q6_K: 8 default tiles are starved, the narrow
        // tile gives 16, and 16 is under two waves of SMS, so a long K still
        // takes the split-K pair.
        let tiling = auto(22, 1024, &Q6_K).expect("a variant fits");
        assert_eq!(tiling.feat_tile, FEAT_TILE_NARROW);
        assert_eq!(tiling.tiles(22, 1024), 16);
        let splits = split_count(4096, 1024, SMS);
        assert!(splits > 1);
        assert!(use_split_launch(
            splits,
            tiling.tiles(22, 1024),
            tiling.feat_tile,
            SMS,
            &Q6_K
        ));
        // The same shape at K=1024 keeps the narrow tile on the tile-parallel
        // grid: the split count's K gate is independent of the feature tile.
        assert_eq!(split_count(1024, 1024, SMS), 1);
        // The split count reads N at the default feature tile, so the narrow
        // tile's doubled tile count leaves it unchanged.
        assert_eq!(split_count(4096, 1024, SMS), splits);
    }
}
