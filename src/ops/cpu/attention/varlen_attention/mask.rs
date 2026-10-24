//! Mask predicate shared by every varlen attention backend.

/// True when key `j` is masked for query `i` of one segment.
///
/// `key_offset = max(0, seq_len_k - seq_len_q)` anchors both rules at the
/// bottom-right corner. The causal rule masks `j > key_offset + i`. The window
/// rule masks `j + window <= key_offset + i`. `window == 0` turns the window
/// off. With `causal == false` the window still anchors at `key_offset + i`.
/// The saturating sum keeps a huge `window` from overflowing.
#[inline]
pub(crate) fn varlen_masked(
    i: usize,
    j: usize,
    key_offset: usize,
    causal: bool,
    window: usize,
) -> bool {
    (causal && j > key_offset + i) || (window > 0 && j.saturating_add(window) <= key_offset + i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_off_and_causal_off_masks_nothing() {
        for i in 0..4 {
            for j in 0..4 {
                assert!(!varlen_masked(i, j, 0, false, 0));
            }
        }
    }

    #[test]
    fn causal_masks_future_keys() {
        assert!(!varlen_masked(2, 2, 0, true, 0));
        assert!(varlen_masked(2, 3, 0, true, 0));
        assert!(!varlen_masked(2, 3, 0, false, 0));
    }

    #[test]
    fn window_one_keeps_only_the_anchor_key() {
        // Query 3, offset 0: keep j == 3 only (non-causal masks the rest on the left).
        assert!(varlen_masked(3, 2, 0, true, 1));
        assert!(!varlen_masked(3, 3, 0, true, 1));
        assert!(varlen_masked(3, 4, 0, true, 1));
        // Non-causal: left of anchor masked, right of anchor kept.
        assert!(varlen_masked(3, 2, 0, false, 1));
        assert!(!varlen_masked(3, 3, 0, false, 1));
        assert!(!varlen_masked(3, 4, 0, false, 1));
    }

    #[test]
    fn window_three_keeps_three_keys_causal() {
        // Query 5, offset 0: keys 3, 4, 5 stay.
        let kept: Vec<usize> = (0..8)
            .filter(|&j| !varlen_masked(5, j, 0, true, 3))
            .collect();
        assert_eq!(kept, vec![3, 4, 5]);
    }

    #[test]
    fn window_larger_than_sequence_masks_nothing_extra() {
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(
                    varlen_masked(i, j, 0, true, 100),
                    varlen_masked(i, j, 0, true, 0)
                );
            }
        }
    }

    #[test]
    fn huge_window_masks_nothing_and_does_not_overflow() {
        assert!(!varlen_masked(5, 0, 0, false, usize::MAX));
        assert!(!varlen_masked(5, 0, 0, true, usize::MAX));
    }

    #[test]
    fn unequal_lengths_shift_the_anchor() {
        // seq_len_q = 2, seq_len_k = 5: key_offset = 3.
        // Query 0 anchors at key 3. Causal keeps 0..=3. Window 2 keeps 2..=3.
        let kept0: Vec<usize> = (0..5)
            .filter(|&j| !varlen_masked(0, j, 3, true, 2))
            .collect();
        assert_eq!(kept0, vec![2, 3]);
        // Query 1 anchors at key 4. Window 2 keeps 3..=4.
        let kept1: Vec<usize> = (0..5)
            .filter(|&j| !varlen_masked(1, j, 3, true, 2))
            .collect();
        assert_eq!(kept1, vec![3, 4]);
    }

    #[test]
    fn non_causal_window_keeps_keys_right_of_anchor() {
        // Query 1, offset 3, window 2: anchor key 4. Keys 3 and above stay.
        let kept: Vec<usize> = (0..7)
            .filter(|&j| !varlen_masked(1, j, 3, false, 2))
            .collect();
        assert_eq!(kept, vec![3, 4, 5, 6]);
    }
}
