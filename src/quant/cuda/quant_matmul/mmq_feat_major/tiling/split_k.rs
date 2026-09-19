//! The split-K rule: how many ranges the K walk is cut into, and whether
//! those ranges run as the split-K pair or back to back inside the
//! tile-parallel grid.
//!
//! Every output element is the sum of its range partials in range order,
//! whichever launch forms it. The split count therefore fixes the float
//! sequence, and it is chosen from K, N and the device alone: never from M,
//! the tile count, or the token tile. A row's result is then the same bits
//! at every batch size and at every tile position. Only the launch choice
//! ([`use_split_launch`]) reads the tile count, and it is bit-neutral.

use super::geometry::{FEAT_TILE_DEFAULT, FEAT_TILE_NARROW};

/// Shortest K the split is worth. The partial stores and the fixup pass are
/// a fixed cost per split, paid once however short the K walk is, so a short
/// K cannot amortise them.
const SPLIT_K_MIN_K: u32 = 2048;

/// Most ranges K is cut into. Bounds the workspace and the fixup's serial
/// walk.
const SPLIT_K_MAX: u32 = 16;

/// Fewest 256-k groups one range holds. A range shorter than this spends
/// more on its prologue, partial store and fixup share than it saves in
/// device fill, so the count is capped at `groups / MIN_GROUPS_PER_RANGE`.
const MIN_GROUPS_PER_RANGE: u32 = 4;

/// Activation k-blocks (32 elements) per 256-k staging group, the kernel's
/// `MMQF_ITER_B`. Split boundaries land on multiples of it.
const GROUP_K_BLOCKS: u32 = 8;

/// Ranges the K walk of a `k`-deep, `n`-wide product is cut into.
///
/// Counts feature tiles at the default feature tile, so the narrow tile's
/// doubled tile count never moves the answer: `n` fixes it. With `ntf`
/// feature tiles and one token tile, `ceil(sms / ntf)` ranges put one block
/// on every SM. That is capped by [`MIN_GROUPS_PER_RANGE`] and
/// [`SPLIT_K_MAX`]. The smallest count from there up to the cap that divides
/// the 256-k group count is taken, so every range is the same length; when
/// none divides, the uncapped-from-below count itself. Never more ranges
/// than groups.
pub(in crate::quant::cuda::quant_matmul::mmq_feat_major) const fn split_count(
    k: u32,
    n: u32,
    sms: u32,
) -> u32 {
    let groups = (k / 32).div_ceil(GROUP_K_BLOCKS);
    if k < SPLIT_K_MIN_K || groups == 0 || sms == 0 || n == 0 {
        return 1;
    }
    let ntf = n.div_ceil(FEAT_TILE_DEFAULT);
    let by_groups = groups / MIN_GROUPS_PER_RANGE;
    let cap = if by_groups < SPLIT_K_MAX {
        by_groups
    } else {
        SPLIT_K_MAX
    };
    let cap = if cap < 1 { 1 } else { cap };
    let need = sms.div_ceil(ntf);
    let need = if need > cap { cap } else { need };
    let mut s = need;
    let mut chosen = need;
    while s <= cap {
        if groups.is_multiple_of(s) {
            chosen = s;
            break;
        }
        s += 1;
    }
    if chosen > groups { groups } else { chosen }
}

/// Whether the ranges run as the split-K pair rather than back to back in
/// the tile-parallel grid. Both give the same bits; this is a schedule choice.
///
/// The pair pays partial stores and a fixup pass for the wave a ragged tile
/// count leaves half empty. Once the tiles fill the device, tile-parallel
/// wins and needs no workspace.
///
/// `tiles` is the launched tile count at `feat_tile`. The thresholds are
/// written for the narrow tile's four-warp blocks; a single-warp tile counts
/// four tiles as one, so the same shape takes the same schedule at every
/// feature tile.
///
/// `prefers_tile_parallel` is the format's measured pick on this device
/// (`super::tile_parallel_tune::prefers_tile_parallel`, never the
/// descriptor's fallback constant read directly). A format that prefers the
/// grid vetoes the pair, but only once the tile count passes about four
/// thirds of the SM count: past that point the split saves too little to
/// cover the fixup pass. Below that threshold the tile-parallel grid cannot
/// fill the device, and the pair wins for every format, veto or not.
pub(in crate::quant::cuda::quant_matmul::mmq_feat_major) const fn use_split_launch(
    splits: u32,
    tiles: u32,
    feat_tile: u32,
    sms: u32,
    prefers_tile_parallel: bool,
) -> bool {
    let tiles = if feat_tile < FEAT_TILE_NARROW {
        tiles * feat_tile / FEAT_TILE_NARROW
    } else {
        tiles
    };
    splits > 1 && sms > 0 && tiles < 2 * sms && !(prefers_tile_parallel && 3 * tiles >= 4 * sms)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SM count the geometry cases below are written against.
    const SMS: u32 = 28;

    /// A format whose measured pick is the tile-parallel grid.
    const GRID: bool = true;

    /// A format whose measured pick is the split-K pair.
    const PAIR: bool = false;

    #[test]
    fn a_short_k_is_one_range() {
        assert_eq!(split_count(1024, 1024, SMS), 1);
        assert_eq!(split_count(2047, 1024, SMS), 1);
        assert_eq!(split_count(0, 1024, SMS), 1);
        assert_eq!(split_count(4096, 1024, 0), 1);
    }

    #[test]
    fn the_count_reads_k_n_and_the_device_only() {
        // 8 feature tiles need 4 ranges; 4 divides the 24 groups of K=6144.
        assert_eq!(split_count(6144, 1024, SMS), 4);
        // 16 groups: 4 divides.
        assert_eq!(split_count(4096, 1024, SMS), 4);
        // 8 groups: the range floor caps it at 2.
        assert_eq!(split_count(2048, 1024, SMS), 2);
        // 12 feature tiles need 3; 3 divides 24, and 4 is the first divisor
        // of 16 from 3 up.
        assert_eq!(split_count(6144, 1536, SMS), 3);
        assert_eq!(split_count(4096, 1536, SMS), 4);
        // 16 feature tiles need 2.
        assert_eq!(split_count(6144, 2048, SMS), 2);
        // 32 feature tiles fill the device unsplit.
        assert_eq!(split_count(6144, 4096, SMS), 1);
    }

    #[test]
    fn the_range_floor_binds_and_the_ragged_case_keeps_need() {
        // One feature tile asks for 28 ranges: 24 groups allow 6, which
        // divides 24.
        assert_eq!(split_count(6144, 64, SMS), 6);
        // K=6176 is 193 k-blocks, 25 groups: 4 ranges are asked for, the
        // floor allows 6, and 5 is the first divisor of 25 from 4 up.
        assert_eq!(split_count(6176, 1024, SMS), 5);
        // K=6176 at one feature tile: the floor allows 6, which does not
        // divide 25, so 6 ragged ranges.
        assert_eq!(split_count(6176, 64, SMS), 6);
        // Never more ranges than groups: K=2048 is 8 groups, floor 2.
        assert_eq!(split_count(2048, 64, SMS), 2);
    }

    #[test]
    fn a_single_range_never_takes_the_pair() {
        assert!(!use_split_launch(1, 8, FEAT_TILE_NARROW, SMS, PAIR));
    }

    #[test]
    fn the_pair_only_when_the_tiles_leave_the_device_short() {
        // 32 tiles is under two waves of SMS, leaving the second wave nearly
        // empty. The veto does not fire yet: 3 * tiles < 4 * SMS.
        assert!(use_split_launch(8, 32, FEAT_TILE_NARROW, SMS, GRID));
        // Two full waves already fill it, so the tile-parallel grid wins.
        assert!(!use_split_launch(8, 56, FEAT_TILE_NARROW, SMS, GRID));
        // No SM count reported: fall back to the tile-parallel grid.
        assert!(!use_split_launch(8, 32, FEAT_TILE_NARROW, 0, GRID));
    }

    #[test]
    fn a_flagged_format_vetoes_the_pair_past_the_threshold() {
        // 3 * tiles >= 4 * SMS: a flagged format takes the tile-parallel grid.
        assert!(!use_split_launch(8, 40, FEAT_TILE_NARROW, SMS, GRID));
        // An unflagged format is unaffected by the veto term at the same
        // geometry where a flagged format is vetoed.
        assert!(use_split_launch(8, 40, FEAT_TILE_NARROW, SMS, PAIR));
    }

    #[test]
    fn a_single_warp_tile_counts_four_tiles_as_one() {
        // 128 single-warp tiles are 32 narrow-tile equivalents: under two
        // waves, so the pair.
        assert!(use_split_launch(2, 128, 16, SMS, GRID));
        // 256 are 64 equivalents: two full waves, tile-parallel.
        assert!(!use_split_launch(2, 256, 16, SMS, GRID));
        // The default tile counts its tiles as they are.
        assert!(!use_split_launch(2, 64, 128, SMS, GRID));
    }

    #[test]
    fn the_veto_lifts_below_four_thirds_of_the_sm_count() {
        // 16 tiles leaves most of SMS with no tile at all, so the pair wins
        // even for a format that vetoes it past the threshold.
        assert!(use_split_launch(8, 16, FEAT_TILE_NARROW, SMS, GRID));
        assert!(use_split_launch(8, 37, FEAT_TILE_NARROW, SMS, GRID));
        // 3 * tiles just clears 4 * SMS: the veto fires just past the threshold.
        assert!(!use_split_launch(8, 38, FEAT_TILE_NARROW, SMS, GRID));
    }
}
