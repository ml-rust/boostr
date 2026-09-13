//! Unit tests for the feature-major MMQ launch decisions.

use super::super::formats::{IQ3_XXS, Q4_1, Q4_K, Q6_K, Q8_0};
use super::*;

/// Ample limit: every compiled variant fits.
const WIDE: u32 = 1 << 20;

/// SM count the geometry cases below are written against.
const SMS: u32 = 28;

fn auto(m: u32, n: u32, format: &FeatMajorFormat) -> Option<Tiling> {
    select_tiling(m, n, WIDE, SMS, format, FeatTile::Auto).expect("tiling rule")
}

#[test]
fn stream_k_only_when_the_tiles_leave_the_device_short() {
    // 32 tiles is under two waves of SMS, leaving the second wave nearly
    // empty. Q8_0's veto does not fire yet: 3 * tiles < 4 * SMS.
    assert!(use_stream_k(32, SMS, 4096, &Q8_0));
    // Two full waves already fill it, so the tile-parallel grid wins.
    assert!(!use_stream_k(56, SMS, 4096, &Q8_0));
    // No SM count reported: fall back to the tile-parallel grid.
    assert!(!use_stream_k(32, 0, 4096, &Q8_0));
}

#[test]
fn a_short_k_keeps_the_tile_parallel_grid() {
    // 16 tiles is under two waves of SMS, so it would take stream-k, but a
    // K walk this short cannot amortise the partial stores and the fixup
    // pass.
    assert!(!use_stream_k(16, SMS, 1024, &Q4_1));
    assert!(!use_stream_k(16, SMS, 2047, &Q4_1));
    assert!(use_stream_k(16, SMS, 2048, &Q4_1));
}

#[test]
fn a_flagged_format_vetoes_stream_k_once_the_tile_count_passes_the_threshold() {
    // 3 * tiles >= 4 * SMS: the tile count has passed about four thirds
    // of SMS, so a flagged format takes the tile-parallel grid.
    assert!(!use_stream_k(40, SMS, 4096, &Q8_0));
    // Two full waves fill the device regardless of the flag.
    assert!(!use_stream_k(56, SMS, 4096, &Q8_0));
    // An unflagged format is unaffected by the veto term at the same
    // geometry where a flagged format is vetoed.
    assert!(use_stream_k(40, SMS, 4096, &Q4_1));
}

#[test]
fn the_veto_lifts_below_four_thirds_of_the_sm_count() {
    // 16 tiles leaves most of SMS with no tile at all. The tile-parallel
    // grid cannot fill the device there, so stream-k wins even for a
    // format that vetoes it once the tile count passes the threshold.
    assert!(use_stream_k(16, SMS, 4096, &IQ3_XXS));
    assert!(use_stream_k(37, SMS, 4096, &IQ3_XXS));
    // 3 * tiles just clears 4 * SMS: the veto fires just past the threshold.
    assert!(!use_stream_k(38, SMS, 4096, &IQ3_XXS));
}

#[test]
fn selects_the_smallest_tile_that_covers_one_batch() {
    assert_eq!(select_variant(1, WIDE, &Q8_0, FEAT_TILE_DEFAULT), Some(8));
    assert_eq!(select_variant(8, WIDE, &Q8_0, FEAT_TILE_DEFAULT), Some(8));
    assert_eq!(select_variant(9, WIDE, &Q8_0, FEAT_TILE_DEFAULT), Some(16));
    assert_eq!(
        select_variant(128, WIDE, &Q8_0, FEAT_TILE_DEFAULT),
        Some(128)
    );
}

#[test]
fn ties_go_to_the_smaller_tile() {
    // 129 needs two tiles at every variant from 80 up, so the scan keeps 80.
    assert_eq!(
        select_variant(129, WIDE, &Q8_0, FEAT_TILE_DEFAULT),
        Some(80)
    );
}

#[test]
fn honours_the_shared_memory_limit() {
    // Only the smallest variants fit under a limit set just above x8.
    let at = |x| smem_bytes(&Q8_0, FEAT_TILE_DEFAULT, x);
    assert_eq!(
        select_variant(1024, at(8), &Q8_0, FEAT_TILE_DEFAULT),
        Some(8)
    );
    assert_eq!(
        select_variant(1024, at(24), &Q8_0, FEAT_TILE_DEFAULT),
        Some(24)
    );
    assert_eq!(select_variant(1024, 0, &Q8_0, FEAT_TILE_DEFAULT), None);
}

#[test]
fn shared_memory_grows_only_with_the_token_tile() {
    assert_eq!(
        smem_bytes(&Q8_0, FEAT_TILE_DEFAULT, 8),
        4 * (128 * 76 + 8 * 36)
    );
    assert_eq!(smem_bytes(&Q8_0, FEAT_TILE_DEFAULT, 128), 57344);
    assert!(VARIANTS.iter().all(|&x| {
        smem_bytes(&Q8_0, FEAT_TILE_DEFAULT, x) <= smem_bytes(&Q8_0, FEAT_TILE_DEFAULT, 128)
    }));
}

#[test]
fn the_narrow_tile_halves_the_weight_tile_and_keeps_the_activation_tile() {
    // Half the weight rows, the same `mmq_x` activation rows.
    assert_eq!(
        smem_bytes(&Q4_K, FEAT_TILE_NARROW, 24),
        4 * (64 * 84 + 24 * 36)
    );
    assert_eq!(
        smem_bytes(&Q4_K, FEAT_TILE_DEFAULT, 24) - smem_bytes(&Q4_K, FEAT_TILE_NARROW, 24),
        4 * 64 * 84
    );
}

#[test]
fn the_narrow_tile_only_offers_its_compiled_token_tiles() {
    // 100 tokens want x112 at the default tile; the narrow list stops at 64.
    assert_eq!(
        select_variant(100, WIDE, &Q4_K, FEAT_TILE_DEFAULT),
        Some(112)
    );
    assert_eq!(select_variant(100, WIDE, &Q4_K, FEAT_TILE_NARROW), Some(64));
    assert_eq!(select_variant(22, WIDE, &Q4_K, FEAT_TILE_NARROW), Some(24));
}

#[test]
fn threads_follow_the_feature_tile() {
    let wide = Tiling {
        feat_tile: FEAT_TILE_DEFAULT,
        mmq_x: 24,
    };
    let narrow = Tiling {
        feat_tile: FEAT_TILE_NARROW,
        mmq_x: 24,
    };
    assert_eq!(wide.threads(), 256);
    assert_eq!(narrow.threads(), 128);
}

#[test]
fn kernel_names_follow_the_macro_spelling() {
    let wide = Tiling {
        feat_tile: FEAT_TILE_DEFAULT,
        mmq_x: 24,
    };
    let narrow = Tiling {
        feat_tile: FEAT_TILE_NARROW,
        mmq_x: 24,
    };
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
}

#[test]
fn a_starved_shape_with_a_short_token_tile_takes_the_narrow_tile() {
    // 22 tokens x 1024 features: one token tile by 8 feature tiles is 8
    // blocks, under two waves of SMS, and x24 keeps the staging chain
    // short. The narrow tile doubles the grid to 16.
    assert_eq!(
        auto(22, 1024, &Q4_K),
        Some(Tiling {
            feat_tile: FEAT_TILE_NARROW,
            mmq_x: 24
        })
    );
    // 22 x 256 at Q4_K: 2 default tiles, 4 narrow. Well under one wave
    // either way, and the short token tile still wins.
    assert_eq!(
        auto(22, 256, &Q4_K),
        Some(Tiling {
            feat_tile: FEAT_TILE_NARROW,
            mmq_x: 24
        })
    );
    // 22 x 4096: 32 default tiles is over one wave but under two, so the
    // default tiling is still starved, and x24 takes the narrow tile.
    assert_eq!(
        auto(22, 4096, &Q4_K),
        Some(Tiling {
            feat_tile: FEAT_TILE_NARROW,
            mmq_x: 24
        })
    );
}

#[test]
fn a_wide_token_tile_takes_the_narrow_tile_only_when_its_grid_covers_the_sms() {
    // 44 x 2048 at Q6_K: 16 default tiles; the narrow grid is 32 x48
    // blocks, at least one per SM, so the doubled blocks co-reside and
    // repay the longer staging chain.
    assert_eq!(
        auto(44, 2048, &Q6_K),
        Some(Tiling {
            feat_tile: FEAT_TILE_NARROW,
            mmq_x: 48
        })
    );
    // 44 x 1024 at Q6_K: 8 default tiles are starved, but the narrow grid
    // is only 16 x48 blocks for SMS: the longer chain is paid and
    // nothing overlaps it, so the default tile stays.
    assert_eq!(
        auto(44, 1024, &Q6_K),
        Some(Tiling {
            feat_tile: FEAT_TILE_DEFAULT,
            mmq_x: 48
        })
    );
    // 44 x 256 at Q6_K: 2 default tiles, 4 narrow. Same verdict.
    assert_eq!(
        auto(44, 256, &Q6_K),
        Some(Tiling {
            feat_tile: FEAT_TILE_DEFAULT,
            mmq_x: 48
        })
    );
}

#[test]
fn a_shape_that_fills_two_waves_keeps_the_default_tile() {
    // 22 x 8192: 64 default tiles already exceed two waves of SMS.
    assert_eq!(
        auto(22, 8192, &Q4_K),
        Some(Tiling {
            feat_tile: FEAT_TILE_DEFAULT,
            mmq_x: 24
        })
    );
    // Prefill-shaped: 512 tokens x 4096 features is 4 x 32 = 128 tiles.
    assert_eq!(
        auto(512, 4096, &Q4_K),
        Some(Tiling {
            feat_tile: FEAT_TILE_DEFAULT,
            mmq_x: 128
        })
    );
}

#[test]
fn the_starvation_test_uses_the_default_tiling_token_tile() {
    // 100 tokens x 1024 features: x112 covers the batch in one token tile,
    // so 8 default tiles are starved; the narrow tile then re-selects its
    // own widest token tile and launches 16 x 2 = 32 blocks, which covers
    // SMS.
    assert_eq!(
        auto(100, 1024, &Q4_K),
        Some(Tiling {
            feat_tile: FEAT_TILE_NARROW,
            mmq_x: 64
        })
    );
}

#[test]
fn a_format_without_the_narrow_tile_never_picks_it() {
    // The same starved geometry as the Q4_K case above.
    assert_eq!(
        auto(22, 1024, &Q8_0),
        Some(Tiling {
            feat_tile: FEAT_TILE_DEFAULT,
            mmq_x: 24
        })
    );
    assert_eq!(
        auto(22, 256, &IQ3_XXS),
        Some(Tiling {
            feat_tile: FEAT_TILE_DEFAULT,
            mmq_x: 24
        })
    );
}

#[test]
fn no_sm_count_keeps_the_default_tile() {
    // A profile that reports no compute units cannot be starved.
    assert_eq!(
        select_tiling(22, 1024, WIDE, 0, &Q4_K, FeatTile::Auto).expect("tiling rule"),
        Some(Tiling {
            feat_tile: FEAT_TILE_DEFAULT,
            mmq_x: 24
        })
    );
}

#[test]
fn a_forced_tile_bypasses_the_rule() {
    // Forced default at a starved shape.
    assert_eq!(
        select_tiling(22, 1024, WIDE, SMS, &Q4_K, FeatTile::Force(128)).expect("tiling rule"),
        Some(Tiling {
            feat_tile: FEAT_TILE_DEFAULT,
            mmq_x: 24
        })
    );
    // Forced narrow at a shape that fills two waves.
    assert_eq!(
        select_tiling(22, 8192, WIDE, SMS, &Q4_K, FeatTile::Force(64)).expect("tiling rule"),
        Some(Tiling {
            feat_tile: FEAT_TILE_NARROW,
            mmq_x: 24
        })
    );
    // Forced narrow where the automatic rule keeps the default tile.
    assert_eq!(
        select_tiling(44, 1024, WIDE, SMS, &Q6_K, FeatTile::Force(64)).expect("tiling rule"),
        Some(Tiling {
            feat_tile: FEAT_TILE_NARROW,
            mmq_x: 48
        })
    );
}

#[test]
fn a_forced_tile_the_format_lacks_is_an_error() {
    assert!(select_tiling(22, 1024, WIDE, SMS, &Q8_0, FeatTile::Force(64)).is_err());
    assert!(select_tiling(22, 1024, WIDE, SMS, &Q4_K, FeatTile::Force(96)).is_err());
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
