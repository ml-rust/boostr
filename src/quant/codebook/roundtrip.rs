//! Whole-tensor round trip: groups a `[out_features, in_features]` weight's
//! flat row-major values into blocks of [`GROUP_SIZE`] along the INPUT
//! dimension, one row at a time, and quantizes each block independently.

use super::levels::Codebook;
use super::quantize::quantize_group;

/// Weights per scale, along the input dimension — fixed, matching the byte
/// cost of the 4-bit encodings this probe compares against (4 bits/weight
/// plus one `f32` scale per 32).
pub(super) const GROUP_SIZE: usize = 32;

/// Quantizes then dequantizes every value in `values` against `codebook`,
/// grouping [`GROUP_SIZE`] consecutive elements at a time WITHIN each row of
/// width `in_features` — a group never crosses a row boundary, so `W`'s
/// columns stay aligned the same way a real block-quantized encoding aligns
/// them.
///
/// - `weights[i]` scores element `i`; see [`quantize_group`] for how a short
///   `weights` slice behaves (never panics, treats a missing entry as
///   `1.0`).
/// - Tail handling: if `in_features` is not a multiple of [`GROUP_SIZE`],
///   the row's final group is quantized on its own (shorter) length — never
///   padded, never dropped. `values.chunks` produces this directly: a row
///   itself shorter than `in_features` (a malformed caller) is likewise
///   grouped on what it actually has, rather than panicking.
/// - `in_features == 0` is a no-op: returns `values` unchanged, since there
///   is no input dimension to group along.
pub fn codebook_round_trip(
    values: &[f32],
    in_features: usize,
    codebook: Codebook,
    weights: &[f32],
) -> Vec<f32> {
    if in_features == 0 {
        return values.to_vec();
    }
    let mut out = Vec::with_capacity(values.len());
    let value_rows = values.chunks(in_features);
    let weight_rows = weights.chunks(in_features);
    for (row, weight_row) in value_rows.zip(weight_rows) {
        for (group, weight_group) in row.chunks(GROUP_SIZE).zip(weight_row.chunks(GROUP_SIZE)) {
            out.extend(quantize_group(group, weight_group, codebook));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_group_shorter_than_group_size_round_trips_without_panicking() {
        // in_features = 5: one short group per row, never a multiple of 32.
        let values: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5, 0.6, -0.1, 0.0, 0.2, -0.3];
        let weights = vec![1.0f32; values.len()];
        let out = codebook_round_trip(&values, 5, Codebook::Uniform, &weights);
        assert_eq!(out.len(), values.len());
    }

    #[test]
    fn a_group_shorter_than_32_is_its_own_group_not_padded() {
        // Single row of length 10 (< GROUP_SIZE): exactly one group of 10.
        let values: Vec<f32> = (0..10).map(|i| i as f32 * 0.1 - 0.5).collect();
        let weights = vec![1.0f32; values.len()];
        let out = codebook_round_trip(&values, 10, Codebook::Nf4, &weights);
        assert_eq!(out.len(), 10);
    }

    #[test]
    fn zero_in_features_is_a_no_op() {
        let values = [0.5f32, -0.25, 0.125];
        let weights = [1.0f32; 3];
        let out = codebook_round_trip(&values, 0, Codebook::Uniform, &weights);
        assert_eq!(out, values.to_vec());
    }

    #[test]
    fn groups_never_cross_a_row_boundary() {
        // in_features = 33: row 0 has one full group of 32 plus a tail
        // group of 1; row 1 starts a FRESH group of 32, not a continuation.
        let in_features = 33;
        let mut values = vec![0.0f32; in_features * 2];
        // The tail element of row 0 and the first element of row 1 are set
        // to the same large magnitude; if grouping crossed the row boundary
        // they would land in the same 32-wide group and share a scale
        // derived from both. They must not: row 0's tail group is a group
        // of exactly 1, quantized to its own nearest level independent of
        // row 1 entirely.
        values[32] = 10.0; // row 0, tail element
        values[in_features] = 0.01; // row 1, first element
        let weights = vec![1.0f32; values.len()];
        let out = codebook_round_trip(&values, in_features, Codebook::Uniform, &weights);
        // Row 0's tail group is a lone element: max_abs == 10.0, so the
        // is = 0 candidate scale is exactly 10.0 and reconstructs it exactly
        // (level 1.0, d = 10.0).
        assert!((out[32] - 10.0).abs() < 1e-3);
        // Row 1's first element quantizes against ITS OWN row's max_abs
        // (0.01), never against row 0's 10.0 — near-exact reconstruction,
        // not a value dominated by row 0's scale.
        assert!((out[in_features] - 0.01).abs() < 1e-3);
    }
}
