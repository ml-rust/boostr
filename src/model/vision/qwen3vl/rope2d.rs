//! Cos/sin tables for the 2D rotary embedding of the vision tower.
//!
//! One head of width `D` rotates pairs `(i, i + D/2)` for `i < D/2`, the
//! split-half layout [`crate::ops::traits::RoPEOps::apply_rope`] expects.
//! The first `D/4` pairs turn by the patch row `y`, the next `D/4` by the
//! patch column `x`, each with its own frequency ladder
//! `theta^(-2k / (D/2))` restarting at `k = 0`. Tokens are enumerated in
//! merge-block order (`by, bx, dy, dx`), the same order the patch tokens
//! take in the encoder.

/// Row-major `[n_tokens, head_dim / 2]` cosine and sine tables.
#[derive(Debug, Clone, PartialEq)]
pub struct Rope2dTables {
    /// Cosines, one row per token.
    pub cos: Vec<f32>,
    /// Sines, one row per token.
    pub sin: Vec<f32>,
    /// Number of tokens (`ph * pw`).
    pub n_tokens: usize,
    /// Pairs per token (`head_dim / 2`).
    pub half_dim: usize,
}

/// Patch `(y, x)` of every token in merge-block order for a `ph x pw`
/// patch grid with `merge x merge` blocks. Both sides must be multiples of
/// `merge`; the caller checks.
pub fn block_order_positions(ph: usize, pw: usize, merge: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(ph * pw);
    for by in (0..ph).step_by(merge) {
        for bx in (0..pw).step_by(merge) {
            for dy in 0..merge {
                for dx in 0..merge {
                    out.push((by + dy, bx + dx));
                }
            }
        }
    }
    out
}

/// Build the tables for a `ph x pw` patch grid.
///
/// `head_dim` must be a multiple of 4. Angles follow the reference: each
/// ladder starts at the raw position and multiplies by
/// `theta^(-2 / (head_dim / 2))` per pair, in `f32`.
pub fn rope2d_tables(
    ph: usize,
    pw: usize,
    merge: usize,
    head_dim: usize,
    theta: f32,
) -> Rope2dTables {
    let half_dim = head_dim / 2;
    let section = head_dim / 4;
    let positions = block_order_positions(ph, pw, merge);
    let n_tokens = positions.len();
    let theta_scale = theta.powf(-2.0 / half_dim as f32);
    let mut cos = Vec::with_capacity(n_tokens * half_dim);
    let mut sin = Vec::with_capacity(n_tokens * half_dim);
    for &(y, x) in &positions {
        let mut angle_y = y as f32;
        let mut angle_x = x as f32;
        for pair in 0..half_dim {
            let angle = if pair < section { angle_y } else { angle_x };
            cos.push(angle.cos());
            sin.push(angle.sin());
            if pair < section {
                angle_y *= theta_scale;
            } else {
                angle_x *= theta_scale;
            }
        }
    }
    Rope2dTables {
        cos,
        sin,
        n_tokens,
        half_dim,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_order_matches_reference_loop() {
        let pos = block_order_positions(4, 6, 2);
        assert_eq!(pos.len(), 24);
        // first block: (0,0) (0,1) (1,0) (1,1); second block starts at x=2
        assert_eq!(&pos[..5], &[(0, 0), (0, 1), (1, 0), (1, 1), (0, 2)]);
        // second block row starts after pw/2 = 3 blocks
        assert_eq!(pos[12], (2, 0));
        assert_eq!(pos[23], (3, 5));
    }

    #[test]
    fn tables_shape_and_first_token_is_identity() {
        let t = rope2d_tables(2, 2, 2, 72, 10000.0);
        assert_eq!(t.n_tokens, 4);
        assert_eq!(t.half_dim, 36);
        assert_eq!(t.cos.len(), 4 * 36);
        // token 0 sits at (0, 0): every angle is zero
        assert!(t.cos[..36].iter().all(|&c| c == 1.0));
        assert!(t.sin[..36].iter().all(|&s| s == 0.0));
    }

    #[test]
    fn ladders_restart_per_section() {
        // token 3 of a 2x2 grid sits at (1, 1): pair k in the row section and
        // pair 18 + k in the column section share the angle scale^k.
        let t = rope2d_tables(2, 2, 2, 72, 10000.0);
        let scale = 10000f32.powf(-2.0 / 36.0);
        let row = &t.cos[3 * 36..4 * 36];
        for k in 0..18 {
            let expect = scale.powi(k as i32).cos();
            assert!((row[k] - expect).abs() < 1e-4, "pair {k}");
            assert!((row[18 + k] - expect).abs() < 1e-4, "pair {}", 18 + k);
        }
    }

    #[test]
    fn row_and_column_sections_take_their_own_position() {
        // token 1 of a 2x2 grid sits at (0, 1): row section identity,
        // column section rotates by 1 * scale^k.
        let t = rope2d_tables(2, 2, 2, 8, 10000.0);
        let row_cos = &t.cos[4..8];
        assert_eq!(&row_cos[..2], &[1.0, 1.0]);
        assert!((row_cos[2] - 1f32.cos()).abs() < 1e-6);
        let scale = 10000f32.powf(-2.0 / 4.0);
        assert!((row_cos[3] - scale.cos()).abs() < 1e-6);
    }
}
