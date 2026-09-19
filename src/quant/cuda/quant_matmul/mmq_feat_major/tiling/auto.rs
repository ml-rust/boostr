//! The automatic tiling policy `select::select_tiling` applies for
//! [`FeatTile::Auto`](super::FeatTile): the single-warp decode shortcut, the
//! default-vs-narrow feature tile, and the cadence choice between them.

use super::super::formats::FeatMajorFormat;
use super::geometry::{
    Cadence, FEAT_TILE_DEFAULT, FEAT_TILE_NARROW, FEAT_TILE_SMALL, SMALL_VARIANTS, group_compiled,
};
use super::split_k::{split_count, use_split_launch};
use super::tile::Tiling;
use super::variant::select_variant;

/// The tiling rule for [`FeatTile::Auto`](super::FeatTile). `None` means no
/// compiled variant fits the device; see `select::select_tiling` for the
/// full rule, including the forced variants.
pub(super) fn select_auto_tiling(
    m: u32,
    n: u32,
    k: u32,
    smem_limit: u32,
    sms: u32,
    format: &FeatMajorFormat,
    prefers_tile_parallel: bool,
) -> Option<Tiling> {
    let at = |tile: u32, cadence: Cadence| {
        select_variant(m, smem_limit, format, tile, cadence).map(|mmq_x| Tiling {
            feat_tile: tile,
            mmq_x,
            cadence,
        })
    };
    let small_max = *SMALL_VARIANTS.last().unwrap_or(&0);
    if format.narrow_tile
        && m <= small_max
        && let Some(small) = at(FEAT_TILE_SMALL, Cadence::Halves)
    {
        return Some(small);
    }
    let wide = at(FEAT_TILE_DEFAULT, Cadence::Halves)?;
    if !format.narrow_tile {
        return Some(wide);
    }
    let Some(narrow) = at(FEAT_TILE_NARROW, Cadence::Halves) else {
        return Some(wide);
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
        let launched =
            if use_split_launch(splits, tiles, narrow.feat_tile, sms, prefers_tile_parallel) {
                tiles * splits
            } else {
                tiles
            };
        if launched < sms && group.smem_bytes(format) <= smem_limit {
            return Some(group);
        }
    }
    Some(if starved { narrow } else { wide })
}

#[cfg(test)]
mod tests {
    //! Unit tests for the automatic tiling policy, driven through
    //! `select::select_tiling` at `FeatTile::Auto` — the forced-tiling paths
    //! are tested in `select`'s own test module.

    use super::super::super::formats::{IQ3_XXS, Q4_K, Q6_K, Q8_0};
    use super::super::FeatTile;
    use super::super::geometry::{FEAT_TILE_DEFAULT, FEAT_TILE_NARROW};
    use super::super::select::select_tiling;
    use super::super::tile::Tiling;
    use super::*;
    use crate::error::Result;

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

    fn auto(m: u32, n: u32, format: &FeatMajorFormat) -> Option<Tiling> {
        pick(m, n, SHORT_K, SMS, format, FeatTile::Auto).expect("tiling rule")
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
            pick(22, 1024, SHORT_K, 0, &Q4_K, FeatTile::Auto).expect("tiling rule"),
            Some(wide(24))
        );
    }

    #[test]
    fn the_group_cadence_counts_the_launched_grid() {
        // 44 x 1024, Q4_K: the narrow tile's 16 tiles at x48 sit under one
        // block per SM, so a one-range K takes the group cadence.
        let short = pick(44, 1024, SHORT_K, SMS, &Q4_K, FeatTile::Auto)
            .expect("tiling rule")
            .expect("a variant fits");
        assert_eq!(short.feat_tile, FEAT_TILE_NARROW);
        assert_eq!(short.cadence, Cadence::Group);
        // At K=4096 the split-K pair launches 16 x 4 blocks, which co-reside,
        // so the two-half cadence keeps its smaller activation tile.
        let long = pick(44, 1024, 4096, SMS, &Q4_K, FeatTile::Auto)
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
            Q6_K.prefers_tile_parallel_fallback
        ));
        // The same shape at K=1024 keeps the narrow tile on the tile-parallel
        // grid: the split count's K gate is independent of the feature tile.
        assert_eq!(split_count(1024, 1024, SMS), 1);
        // The split count reads N at the default feature tile, so the narrow
        // tile's doubled tile count leaves it unchanged.
        assert_eq!(split_count(4096, 1024, SMS), splits);
    }
}
