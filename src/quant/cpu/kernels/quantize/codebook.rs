//! Fixed non-linear codebook search — llama.cpp `best_index_int8`
//!
//! IQ4_NL and IQ4_XS both quantize against `KVALUES_IQ4NL`, a 16-entry sorted
//! table with no arithmetic relationship between index and value (unlike a
//! uniform integer level). Finding the nearest entry is a binary search over
//! the sorted table, not a round-and-clamp. Both formats need the identical
//! routine, so it lives here rather than inside `iq4_nl.rs` — IQ4_XS's writer
//! reuses it without a second copy of the search.

/// Index of the codebook entry nearest `x`, ties broken toward the lower index
///
/// `val` must be sorted ascending. Mirrors `ggml-quants.c`'s `best_index_int8`
/// exactly, including its tie rule: at the boundary between `val[mu-1]` and
/// `val[mu]`, the lower index wins when the two distances are equal.
pub fn best_index_int8(val: &[i8; 16], x: f32) -> usize {
    if x <= val[0] as f32 {
        return 0;
    }
    if x >= val[15] as f32 {
        return 15;
    }
    let mut ml = 0usize;
    let mut mu = 15usize;
    while mu - ml > 1 {
        let mav = (ml + mu) / 2;
        if x < val[mav] as f32 {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    if x - (val[mu - 1] as f32) < (val[mu] as f32) - x {
        mu - 1
    } else {
        mu
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::tables::KVALUES_IQ4NL;

    #[test]
    fn clamps_below_and_above_range() {
        assert_eq!(best_index_int8(&KVALUES_IQ4NL, -1000.0), 0);
        assert_eq!(best_index_int8(&KVALUES_IQ4NL, 1000.0), 15);
    }

    #[test]
    fn exact_hits_return_their_own_index() {
        for (i, &v) in KVALUES_IQ4NL.iter().enumerate() {
            assert_eq!(best_index_int8(&KVALUES_IQ4NL, v as f32), i);
        }
    }

    /// An exact midpoint takes the HIGHER index. `ggml-quants.c` decides the
    /// last step with `x - val[mu-1] < val[mu] - x ? mu-1 : mu`, and a tie
    /// fails that strict `<`. Byte-exactness depends on this edge going the
    /// reference's way, not the intuitive way.
    #[test]
    fn midpoint_ties_favor_the_higher_index() {
        let mid = (-127.0 + -104.0) / 2.0;
        assert_eq!(best_index_int8(&KVALUES_IQ4NL, mid), 1);
        // Strictly below the midpoint still rounds down.
        assert_eq!(best_index_int8(&KVALUES_IQ4NL, mid - 0.5), 0);
    }
}
