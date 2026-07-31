//! ALiBi per-head slopes.
//!
//! ALiBi ("Attention with Linear Biases") encodes position by subtracting a
//! per-head multiple of the key distance from every attention score, instead of
//! rotating Q/K (RoPE) or adding a learned position vector. The whole scheme is
//! therefore one number per head — the slope — and getting those numbers right
//! is the entire correctness surface.

/// Per-head ALiBi slopes for `n_heads` heads and a maximum bias of `max_bias`.
///
/// Reproduces ggml's `ggml_compute_forward_soft_max_f32`, which is what
/// llama.cpp's `ggml_soft_max_ext(..., f_max_alibi_bias)` runs:
///
/// ```text
/// n_head_log2 = 2 ^ floor(log2(n_heads))
/// m0          = 2 ^ (-max_bias / n_head_log2)
/// m1          = 2 ^ (-(max_bias / 2) / n_head_log2)
/// slope(h)    = h < n_head_log2 ? m0 ^ (h + 1)
///                               : m1 ^ (2 * (h - n_head_log2) + 1)
/// ```
///
/// The two-branch form only collapses to the familiar `2^(-max_bias*h/n_heads)`
/// geometric series when `n_heads` is a power of two. jina-bert-v2 has 12 heads,
/// so the second branch is live for heads 8..11 and the simple formula would
/// give the wrong slope for a third of them — with no shape or magnitude change
/// to reveal it.
pub fn alibi_slopes(n_heads: usize, max_bias: f32) -> Vec<f32> {
    if n_heads == 0 {
        return Vec::new();
    }
    let n_head_log2 = 1usize << (usize::BITS - 1 - n_heads.leading_zeros()) as usize;
    let m0 = 2.0f32.powf(-max_bias / n_head_log2 as f32);
    let m1 = 2.0f32.powf(-(max_bias / 2.0) / n_head_log2 as f32);

    (0..n_heads)
        .map(|h| {
            if h < n_head_log2 {
                m0.powi(h as i32 + 1)
            } else {
                m1.powi(2 * (h as i32 - n_head_log2 as i32) + 1)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A power-of-two head count uses only the first branch, so the slopes are
    /// exactly the geometric series `2^(-max_bias * (h+1) / n_heads)`.
    #[test]
    fn power_of_two_head_count_is_the_plain_geometric_series() {
        let slopes = alibi_slopes(8, 8.0);
        let expected: Vec<f32> = (0..8).map(|h| 2.0f32.powf(-(h as f32 + 1.0))).collect();
        assert_eq!(slopes.len(), 8);
        for (got, want) in slopes.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    /// jina-bert-v2-base: 12 heads, max_bias 8. `n_head_log2` is 8, so heads
    /// 0..7 take the first branch and heads 8..11 the interleaved second one.
    /// These are the values ggml produces; the naive `2^(-8h/12)` form does not
    /// reproduce a single one of them.
    #[test]
    fn twelve_heads_uses_both_branches() {
        let slopes = alibi_slopes(12, 8.0);
        assert_eq!(slopes.len(), 12);

        // First branch: m0 = 2^(-8/8) = 0.5, so slope(h) = 0.5^(h+1).
        for (h, got) in slopes.iter().enumerate().take(8) {
            let want = 0.5f32.powi(h as i32 + 1);
            assert!((got - want).abs() < 1e-7, "head {h}: {got}");
        }

        // Second branch: m1 = 2^(-4/8) = 2^-0.5, so slope(8+k) = 2^-(k + 0.5).
        for k in 0..4usize {
            let want = 2.0f32.powf(-(k as f32) - 0.5);
            let got = slopes[8 + k];
            assert!((got - want).abs() < 1e-7, "head {}: {got} vs {want}", 8 + k);
        }

        // Every slope is positive, and each branch decreases within itself.
        // The sequence is NOT monotonic overall — ggml's second branch restarts
        // near 2^-0.5, well above the 2^-8 the first branch ends at. That
        // discontinuity is the formula, not a bug, and asserting a single
        // decreasing run here would be asserting the wrong thing.
        assert!(slopes.iter().all(|s| *s > 0.0));
        assert!(slopes[..8].windows(2).all(|w| w[0] > w[1]));
        assert!(slopes[8..].windows(2).all(|w| w[0] > w[1]));
        assert!(slopes[8] > slopes[7]);
    }

    /// `max_bias = 0` disables the penalty entirely (all slopes 1.0 in ggml's
    /// formula, but the caller never builds a bias in that case — the identity
    /// here is that the slopes are still well-defined and finite).
    #[test]
    fn zero_max_bias_gives_unit_slopes() {
        for s in alibi_slopes(12, 0.0) {
            assert!((s - 1.0).abs() < 1e-7);
        }
    }
}
