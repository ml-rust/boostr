//! Picks the token-tile variant and the (feature tile, cadence) pair a shape
//! runs at — the tiling rule this whole module exists to apply.

use crate::error::{Error, Result};

use super::super::formats::FeatMajorFormat;
use super::FeatTile;
use super::geometry::{
    Cadence, FEAT_TILE_DEFAULT, FEAT_TILE_NARROW, GROUP_CADENCE_MIN_X, group_compiled, smem_bytes,
    variants_at,
};
use super::tile::Tiling;

/// Picks the token tile at `feat_tile` that launches the fewest token tiles
/// for `m`, breaking ties toward the smaller tile because it costs fewer
/// registers and less shared memory. `None` means no variant fits the device.
///
/// The fit test uses `cadence`'s shared-memory request. The caller that wants
/// the group cadence must ask with it: its doubled activation tile can push
/// a token tile over the limit that the halves cadence keeps.
pub(in crate::quant::cuda::quant_matmul::mmq_feat_major) fn select_variant(
    m: u32,
    smem_limit: u32,
    format: &FeatMajorFormat,
    feat_tile: u32,
    cadence: Cadence,
) -> Option<u32> {
    let mut best: Option<(u32, u32)> = None;
    for &mmq_x in variants_at(feat_tile) {
        if smem_bytes(format, feat_tile, mmq_x, cadence) > smem_limit {
            continue;
        }
        let tiles = m.div_ceil(mmq_x);
        if best.is_none_or(|(_, best_tiles)| tiles < best_tiles) {
            best = Some((mmq_x, tiles));
        }
        if tiles == 1 {
            break;
        }
    }
    best.map(|(mmq_x, _)| mmq_x)
}

/// The tiling rule. `Ok(None)` means no compiled variant fits the device.
///
/// The default feature tile is tried first and is the answer for every
/// format without the narrow tile. Otherwise the narrow tile's own token
/// tile for `m` decides which lever is in play:
///
/// - Short token tile (below [`GROUP_CADENCE_MIN_X`]): only the two-half
///   cadence exists, with the default tile's activation footprint, so the
///   narrow tile is pure block count. It is taken exactly when the default
///   tiling is starved: fewer output tiles than two waves of SMs, the bound
///   `use_stream_k` draws. Below it the tile-parallel grid leaves SMs idle
///   for the whole K walk, and stream-k can only split what few tiles there
///   are.
/// - Wide token tile: per-group activation staging dominates, and the grid
///   picks the cadence. Under one block per SM nothing co-resides, so the
///   full-group cadence's halved barriers are pure gain. At or above it the
///   two-half cadence keeps the smaller activation tile and the
///   co-residency the group cadence would spend. It is taken while the
///   default tiling is still starved; past starvation the default tile
///   stays.
///
/// The starvation test runs on the default tiling's own token tile; the
/// narrow tiling re-selects its token tile for `m`. All of it runs before the
/// stream-k decision, which then applies to whichever tiling was chosen.
///
/// A forced tiling bypasses the rule. A forced tiling the format does not
/// compile is an error, not a silent fallback: a caller forcing a tiling is
/// measuring that tiling.
pub(in crate::quant::cuda::quant_matmul::mmq_feat_major) fn select_tiling(
    m: u32,
    n: u32,
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
    match feat_tile {
        FeatTile::Force(tile) if tile != FEAT_TILE_DEFAULT && tile != FEAT_TILE_NARROW => {
            Err(not_compiled(
                format!("feature tile {tile}"),
                &format!("force {FEAT_TILE_DEFAULT} or {FEAT_TILE_NARROW}"),
            ))
        }
        FeatTile::Force(tile) if tile == FEAT_TILE_NARROW && !format.narrow_tile => {
            Err(not_compiled(
                format!("feature tile {tile}"),
                &format!("force {FEAT_TILE_DEFAULT} or use the automatic tile"),
            ))
        }
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
                if narrow.tiles(m, n) < sms && group.smem_bytes(format) <= smem_limit {
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
    use super::super::geometry::{FEAT_TILE_DEFAULT, FEAT_TILE_NARROW, NARROW_VARIANTS, VARIANTS};
    use super::super::stream_k::use_stream_k;
    use super::super::tile::Tiling;
    use super::super::{Cadence, Role};
    use super::*;

    /// Ample limit: every compiled variant fits.
    const WIDE: u32 = 1 << 20;

    /// SM count the geometry cases below are written against.
    const SMS: u32 = 28;

    fn auto(m: u32, n: u32, format: &FeatMajorFormat) -> Option<Tiling> {
        select_tiling(m, n, WIDE, SMS, format, FeatTile::Auto).expect("tiling rule")
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

    #[test]
    fn selects_the_smallest_tile_that_covers_one_batch() {
        assert_eq!(
            select_variant(1, WIDE, &Q8_0, FEAT_TILE_DEFAULT, Cadence::Halves),
            Some(8)
        );
        assert_eq!(
            select_variant(8, WIDE, &Q8_0, FEAT_TILE_DEFAULT, Cadence::Halves),
            Some(8)
        );
        assert_eq!(
            select_variant(9, WIDE, &Q8_0, FEAT_TILE_DEFAULT, Cadence::Halves),
            Some(16)
        );
        assert_eq!(
            select_variant(128, WIDE, &Q8_0, FEAT_TILE_DEFAULT, Cadence::Halves),
            Some(128)
        );
    }

    #[test]
    fn ties_go_to_the_smaller_tile() {
        // 129 needs two tiles at every variant from 80 up, so the scan keeps 80.
        assert_eq!(
            select_variant(129, WIDE, &Q8_0, FEAT_TILE_DEFAULT, Cadence::Halves),
            Some(80)
        );
    }

    #[test]
    fn honours_the_shared_memory_limit() {
        // Only the smallest variants fit under a limit set just above x8.
        let at = |x| smem_bytes(&Q8_0, FEAT_TILE_DEFAULT, x, Cadence::Halves);
        assert_eq!(
            select_variant(1024, at(8), &Q8_0, FEAT_TILE_DEFAULT, Cadence::Halves),
            Some(8)
        );
        assert_eq!(
            select_variant(1024, at(24), &Q8_0, FEAT_TILE_DEFAULT, Cadence::Halves),
            Some(24)
        );
        assert_eq!(
            select_variant(1024, 0, &Q8_0, FEAT_TILE_DEFAULT, Cadence::Halves),
            None
        );
    }

    #[test]
    fn shared_memory_grows_only_with_the_token_tile() {
        assert_eq!(
            smem_bytes(&Q8_0, FEAT_TILE_DEFAULT, 8, Cadence::Halves),
            4 * (128 * 76 + 8 * 36)
        );
        assert_eq!(
            smem_bytes(&Q8_0, FEAT_TILE_DEFAULT, 128, Cadence::Halves),
            57344
        );
        assert!(VARIANTS.iter().all(|&x| {
            smem_bytes(&Q8_0, FEAT_TILE_DEFAULT, x, Cadence::Halves)
                <= smem_bytes(&Q8_0, FEAT_TILE_DEFAULT, 128, Cadence::Halves)
        }));
    }

    #[test]
    fn the_narrow_tile_halves_the_weight_tile_and_the_group_cadence_doubles_the_activation_tile() {
        // Half the weight rows; one 128-k half of the activation rows on the
        // two-half cadence, both halves on the full-group cadence.
        assert_eq!(
            smem_bytes(&Q4_K, FEAT_TILE_NARROW, 24, Cadence::Halves),
            4 * (64 * 84 + 24 * 36)
        );
        assert_eq!(
            smem_bytes(&Q4_K, FEAT_TILE_NARROW, 48, Cadence::Halves),
            4 * (64 * 84 + 48 * 36)
        );
        assert_eq!(
            smem_bytes(&Q4_K, FEAT_TILE_NARROW, 48, Cadence::Group),
            4 * (64 * 84 + 2 * 48 * 36)
        );
        // The doubled activation tile costs less than the weight rows it freed
        // at every narrow token tile, so the narrow request is always smaller.
        assert!(NARROW_VARIANTS.iter().all(|&x| {
            smem_bytes(&Q4_K, FEAT_TILE_NARROW, x, Cadence::Group)
                < smem_bytes(&Q4_K, FEAT_TILE_DEFAULT, x, Cadence::Halves)
        }));
    }

    #[test]
    fn the_narrow_tile_only_offers_its_compiled_token_tiles() {
        // 100 tokens want x112 at the default tile; the narrow list stops at 64.
        assert_eq!(
            select_variant(100, WIDE, &Q4_K, FEAT_TILE_DEFAULT, Cadence::Halves),
            Some(112)
        );
        assert_eq!(
            select_variant(100, WIDE, &Q4_K, FEAT_TILE_NARROW, Cadence::Halves),
            Some(64)
        );
        assert_eq!(
            select_variant(22, WIDE, &Q4_K, FEAT_TILE_NARROW, Cadence::Halves),
            Some(24)
        );
    }

    #[test]
    fn threads_follow_the_feature_tile() {
        let wide = wide(24);
        let narrow = narrow(24);
        assert_eq!(wide.threads(), 256);
        assert_eq!(narrow.threads(), 128);
    }

    #[test]
    fn kernel_names_follow_the_macro_spelling() {
        let wide = wide(24);
        let narrow = narrow(24);
        assert_eq!(
            wide.kernel_name(&Q4_K, Role::TileParallel),
            "quant_mmq_q4_k_q8_1_mma_x24"
        );
        assert_eq!(
            wide.kernel_name(&Q4_K, Role::StreamK),
            "quant_mmq_q4_k_q8_1_mma_sk_x24"
        );
        assert_eq!(
            narrow.kernel_name(&Q6_K, Role::TileParallel),
            "quant_mmq_q6_k_q8_1_mma_y64_x24"
        );
        assert_eq!(
            narrow.kernel_name(&Q6_K, Role::StreamK),
            "quant_mmq_q6_k_q8_1_mma_sk_y64_x24"
        );
        assert_eq!(
            narrow.kernel_name(&Q6_K, Role::Fixup),
            "quant_mmq_q6_k_q8_1_mma_fixup_y64_x24"
        );
        let group = group(48);
        assert_eq!(
            group.kernel_name(&Q6_K, Role::TileParallel),
            "quant_mmq_q6_k_q8_1_mma_y64g_x48"
        );
        assert_eq!(
            group.kernel_name(&Q6_K, Role::StreamK),
            "quant_mmq_q6_k_q8_1_mma_sk_y64g_x48"
        );
        assert_eq!(
            group.kernel_name(&Q6_K, Role::Fixup),
            "quant_mmq_q6_k_q8_1_mma_fixup_y64g_x48"
        );
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
            select_tiling(22, 1024, WIDE, 0, &Q4_K, FeatTile::Auto).expect("tiling rule"),
            Some(wide(24))
        );
    }

    #[test]
    fn a_forced_tile_bypasses_the_rule() {
        // Forced default at a starved shape.
        assert_eq!(
            select_tiling(22, 1024, WIDE, SMS, &Q4_K, FeatTile::Force(128)).expect("tiling rule"),
            Some(wide(24))
        );
        // Forced narrow at a shape that fills two waves.
        assert_eq!(
            select_tiling(22, 8192, WIDE, SMS, &Q4_K, FeatTile::Force(64)).expect("tiling rule"),
            Some(narrow(24))
        );
        // Forced narrow on the two-half cadence where the automatic rule picks
        // the group cadence.
        assert_eq!(
            select_tiling(44, 1024, WIDE, SMS, &Q6_K, FeatTile::Force(64)).expect("tiling rule"),
            Some(narrow(48))
        );
        // Forced group cadence where the automatic rule picks the two-half one.
        assert_eq!(
            select_tiling(44, 4096, WIDE, SMS, &Q6_K, FeatTile::ForceNarrowGroup)
                .expect("tiling rule"),
            Some(group(48))
        );
    }

    #[test]
    fn a_forced_tile_the_format_lacks_is_an_error() {
        assert!(select_tiling(22, 1024, WIDE, SMS, &Q8_0, FeatTile::Force(64)).is_err());
        assert!(select_tiling(22, 1024, WIDE, SMS, &Q4_K, FeatTile::Force(96)).is_err());
        assert!(select_tiling(44, 1024, WIDE, SMS, &Q8_0, FeatTile::ForceNarrowGroup).is_err());
        // The group cadence exists only from x40 up: m=22 selects x24.
        assert!(select_tiling(22, 1024, WIDE, SMS, &Q4_K, FeatTile::ForceNarrowGroup).is_err());
    }

    #[test]
    fn stream_k_composes_with_the_narrow_tile() {
        // 22 x 1024 at K=4096, Q6_K: 8 default tiles are starved, the narrow
        // tile gives 16, and 16 is under two waves of SMS, so a long K still
        // takes stream-k.
        let tiling = auto(22, 1024, &Q6_K).expect("a variant fits");
        assert_eq!(tiling.feat_tile, FEAT_TILE_NARROW);
        assert_eq!(tiling.tiles(22, 1024), 16);
        assert!(use_stream_k(tiling.tiles(22, 1024), SMS, 4096, &Q6_K));
        // The same shape at K=1024 keeps the narrow tile on the tile-parallel
        // grid: the stream-k gate on K is independent of the feature tile.
        assert!(!use_stream_k(tiling.tiles(22, 1024), SMS, 1024, &Q6_K));
    }
}
