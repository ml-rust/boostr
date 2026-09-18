//! Picks the token-tile variant at one feature tile: the compiled token
//! tile that launches the fewest token tiles for a batch and fits the
//! device's shared memory.

use super::super::formats::FeatMajorFormat;
use super::geometry::{
    Cadence, FEAT_TILE_DEFAULT, FEAT_TILE_NARROW, FEAT_TILE_SMALL, smem_bytes, variants_at,
};

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

/// Token slots an activation record needs so that every tiling `format`
/// can be launched at for `m` tokens finds its last tile's copy inside it:
/// the most `token_tiles * mmq_x` over the compiled feature tiles and
/// cadences. A record this long serves any of them, which is what one
/// activation shared by several weights needs.
pub(in crate::quant::cuda::quant_matmul::mmq_feat_major) fn record_token_slots(
    m: u32,
    smem_limit: u32,
    format: &FeatMajorFormat,
) -> u32 {
    let tiles: &[u32] = if format.narrow_tile {
        &[FEAT_TILE_SMALL, FEAT_TILE_NARROW, FEAT_TILE_DEFAULT]
    } else {
        &[FEAT_TILE_DEFAULT]
    };
    let mut slots = 0;
    for &tile in tiles {
        for cadence in [Cadence::Halves, Cadence::Group] {
            if let Some(mmq_x) = select_variant(m, smem_limit, format, tile, cadence) {
                slots = slots.max(m.div_ceil(mmq_x) * mmq_x);
            }
        }
    }
    slots
}

#[cfg(test)]
mod tests {
    use super::super::super::formats::{Q4_K, Q8_0};
    use super::super::geometry::{FEAT_TILE_DEFAULT, FEAT_TILE_NARROW, NARROW_VARIANTS, VARIANTS};
    use super::*;

    /// Ample limit: every compiled variant fits.
    const WIDE: u32 = 1 << 20;

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
        // Only the smallest variant fits under a limit set at x8.
        let at = |x| smem_bytes(&Q8_0, FEAT_TILE_DEFAULT, x, Cadence::Halves);
        assert_eq!(
            select_variant(1024, at(8), &Q8_0, FEAT_TILE_DEFAULT, Cadence::Halves),
            Some(8)
        );
        // x24 and x32 stage both activation halves and ask for more than x40,
        // so a limit set at x40 admits x40 and skips them.
        assert!(at(32) > at(40));
        assert_eq!(
            select_variant(1024, at(40), &Q8_0, FEAT_TILE_DEFAULT, Cadence::Halves),
            Some(40)
        );
        assert_eq!(
            select_variant(1024, 0, &Q8_0, FEAT_TILE_DEFAULT, Cadence::Halves),
            None
        );
    }

    #[test]
    fn the_record_covers_every_tiling_of_the_batch() {
        // One decode token: one x8 tile at every feature tile.
        assert_eq!(record_token_slots(1, WIDE, &Q4_K), 8);
        assert_eq!(record_token_slots(8, WIDE, &Q8_0), 8);
        // 100 tokens: x112 at the default tile, two x64 tiles at the narrow one.
        assert_eq!(record_token_slots(100, WIDE, &Q4_K), 128);
        assert_eq!(record_token_slots(100, WIDE, &Q8_0), 112);
        // 129 tokens: two x80 tiles.
        assert_eq!(record_token_slots(129, WIDE, &Q8_0), 160);
    }

    #[test]
    fn shared_memory_grows_only_with_the_token_tile() {
        // Token tiles up to `SMALL_X` stage both activation halves.
        assert_eq!(
            smem_bytes(&Q8_0, FEAT_TILE_DEFAULT, 8, Cadence::Halves),
            4 * (128 * 76 + 2 * 8 * 36)
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
            4 * (64 * 84 + 2 * 24 * 36)
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
    fn the_single_warp_tile_offers_two_token_tiles() {
        assert_eq!(
            select_variant(1, WIDE, &Q4_K, FEAT_TILE_SMALL, Cadence::Halves),
            Some(8)
        );
        assert_eq!(
            select_variant(16, WIDE, &Q4_K, FEAT_TILE_SMALL, Cadence::Halves),
            Some(16)
        );
        // Past 16 tokens it can only tile the batch, never cover it.
        assert_eq!(
            select_variant(17, WIDE, &Q4_K, FEAT_TILE_SMALL, Cadence::Halves),
            Some(16)
        );
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
}
