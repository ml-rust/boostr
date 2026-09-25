//! The row counts a VoxCPM2 generation loop is allowed to run at.
//!
//! The DiT's Euler solver caches one captured graph per shape key, and the
//! key carries the batch size. Running at an arbitrary row count would put a
//! fresh key in that cache on every step and churn the captures, so every
//! width the loop can take comes from ONE short list. Padding to a listed
//! width costs a dead row a full solve per step, which is why the list is
//! short and stops rather than covering every count.
//!
//! This is the single definition. The engine's batch builder and the
//! generation state's stepped width both read it here; a second copy of the
//! list would let the two disagree about which shapes the graph cache holds.

/// Row counts the generation loop runs at, ascending.
pub const BATCH_WIDTHS: [usize; 4] = [1, 2, 4, 8];

/// The width `rows` real rows run at: the smallest allowed width that holds
/// them all, or the largest allowed width when `rows` is past every entry.
///
/// A caller that can exceed the largest entry clamps the result itself; see
/// [`clamp_max_batch`] for the cap that keeps a batch from ever asking.
pub fn batch_width(rows: usize) -> usize {
    let largest = BATCH_WIDTHS[BATCH_WIDTHS.len() - 1];
    BATCH_WIDTHS
        .iter()
        .copied()
        .find(|width| *width >= rows)
        .unwrap_or(largest)
}

/// `max_batch` rounded DOWN to an allowed width, at least 1. Rounding down
/// keeps a batch from ever needing a width the cap forbids.
pub fn clamp_max_batch(max_batch: usize) -> usize {
    BATCH_WIDTHS
        .iter()
        .copied()
        .filter(|width| *width <= max_batch)
        .max()
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_width_rounds_up_to_an_allowed_width() {
        assert_eq!(batch_width(0), 1);
        assert_eq!(batch_width(1), 1);
        assert_eq!(batch_width(2), 2);
        assert_eq!(batch_width(3), 4);
        assert_eq!(batch_width(4), 4);
        assert_eq!(batch_width(5), 8);
        assert_eq!(batch_width(8), 8);
    }

    /// Past the last entry the list has nothing to offer, so the largest
    /// entry comes back and the caller must clamp it against its own row
    /// count rather than trusting it to hold them.
    #[test]
    fn batch_width_saturates_at_the_largest_entry() {
        assert_eq!(batch_width(9), 8);
        assert_eq!(batch_width(64), 8);
    }

    #[test]
    fn max_batch_rounds_down_to_an_allowed_width() {
        assert_eq!(clamp_max_batch(0), 1);
        assert_eq!(clamp_max_batch(1), 1);
        assert_eq!(clamp_max_batch(3), 2);
        assert_eq!(clamp_max_batch(7), 4);
        assert_eq!(clamp_max_batch(8), 8);
        assert_eq!(clamp_max_batch(64), 8);
    }

    /// A clamped cap never lets a batch need a width above the largest
    /// allowed one.
    #[test]
    fn a_clamped_cap_always_admits_its_own_width() {
        for cap in 0..64 {
            let clamped = clamp_max_batch(cap);
            assert_eq!(batch_width(clamped), clamped, "cap {cap}");
        }
    }
}
