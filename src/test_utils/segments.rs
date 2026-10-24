//! Packed-document fixtures.

/// Segment ids for back-to-back documents of the given lengths.
pub(crate) fn ids_for(lens: &[usize]) -> Vec<i32> {
    let mut ids = Vec::new();
    for (i, &n) in lens.iter().enumerate() {
        ids.extend(std::iter::repeat_n(i as i32 * 3 + 1, n));
    }
    ids
}
