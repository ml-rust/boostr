//! Unit tests for the feature-major MMQ launch decisions.

use super::super::formats::{IQ3_XXS, Q4_1, Q8_0};
use super::*;

/// Ample limit: every compiled variant fits.
const WIDE: u32 = 1 << 20;

#[test]
fn declares_that_it_reassociates() {
    const { assert!(CONTRACT.reassociates) };
}

#[test]
fn stream_k_only_when_the_tiles_leave_the_device_short() {
    // 32 tiles across 28 SMs leaves the second wave nearly empty. Q8_0's
    // veto does not fire yet: 3*32 = 96 < 4*28 = 112.
    assert!(use_stream_k(32, 28, &Q8_0));
    // Two full waves already fill it, so the tile-parallel grid wins.
    assert!(!use_stream_k(56, 28, &Q8_0));
    // No SM count reported: fall back to the tile-parallel grid.
    assert!(!use_stream_k(32, 0, &Q8_0));
}

#[test]
fn a_flagged_format_vetoes_stream_k_once_the_tile_count_passes_the_threshold() {
    // 3*40 = 120 >= 4*28 = 112: the tile count has passed about four thirds
    // of the SM count, so a flagged format takes the tile-parallel grid.
    assert!(!use_stream_k(40, 28, &Q8_0));
    // Two full waves fill the device regardless of the flag.
    assert!(!use_stream_k(56, 28, &Q8_0));
    // An unflagged format is unaffected by the veto term at the same
    // geometry where a flagged format is vetoed.
    assert!(use_stream_k(40, 28, &Q4_1));
}

#[test]
fn the_veto_lifts_below_four_thirds_of_the_sm_count() {
    // 16 tiles leaves 12 of 28 SMs with no tile at all. The tile-parallel
    // grid cannot fill the device there, so stream-k wins even for a
    // format that vetoes it once the tile count passes the threshold.
    assert!(use_stream_k(16, 28, &IQ3_XXS));
    assert!(use_stream_k(37, 28, &IQ3_XXS));
    // 3*38 = 114 >= 4*28 = 112: the veto fires just past the threshold.
    assert!(!use_stream_k(38, 28, &IQ3_XXS));
}

#[test]
fn selects_the_smallest_tile_that_covers_one_batch() {
    assert_eq!(select_variant(1, WIDE, &Q8_0), Some(8));
    assert_eq!(select_variant(8, WIDE, &Q8_0), Some(8));
    assert_eq!(select_variant(9, WIDE, &Q8_0), Some(16));
    assert_eq!(select_variant(128, WIDE, &Q8_0), Some(128));
}

#[test]
fn ties_go_to_the_smaller_tile() {
    // 129 needs two tiles at every variant from 80 up, so the scan keeps 80.
    assert_eq!(select_variant(129, WIDE, &Q8_0), Some(80));
}

#[test]
fn honours_the_shared_memory_limit() {
    // Only the smallest variants fit under a limit set just above x8.
    assert_eq!(select_variant(1024, smem_bytes(&Q8_0, 8), &Q8_0), Some(8));
    assert_eq!(select_variant(1024, smem_bytes(&Q8_0, 24), &Q8_0), Some(24));
    assert_eq!(select_variant(1024, 0, &Q8_0), None);
}

#[test]
fn shared_memory_grows_only_with_the_token_tile() {
    assert_eq!(smem_bytes(&Q8_0, 8), 4 * (128 * 76 + 8 * 36));
    assert_eq!(smem_bytes(&Q8_0, 128), 57344);
    assert!(
        VARIANTS
            .iter()
            .all(|&x| smem_bytes(&Q8_0, x) <= smem_bytes(&Q8_0, 128))
    );
}
