//! I-quants past the IQ2 grid width: `IQ3_XXS`, `IQ3_S` share a 4-component
//! grid; `IQ1_S` is the family's only affine format. All three resolve their
//! index through `mmqf_stage_iq_grid`, same as [`super::grid`]'s IQ2 family.

use super::super::FeatMajorFormat;
use crate::quant::QuantFormat;

/// IQ3_XXS: 98-byte blocks of 256 elements, staged as Q8_0's row byte for byte
/// — 64 quant words plus 8 f32 group scales plus 4 ints of bank padding.
///
/// The family's first 4-COMPONENT grid: each byte of `qs` indexes a 256-entry
/// codebook whose entry expands to FOUR magnitude bytes, not the IQ2 grids'
/// eight, so an 8-element sign sub-group spans TWO consecutive `qs` bytes. The
/// sign byte itself is a sign-table entry named by a 7-bit field of the group's
/// `aux` word, as in IQ2_XXS. The kernel expands both points and folds the sign
/// in while staging, so the staged lanes are signed int8 and the whole
/// `vec_dot` is Q8_0's.
///
/// Its scale is the top nibble of the group's `aux` word and covers that whole
/// 32-element group, which is the granularity `vec_dot` already indexes at, so
/// it stages 8 f32 rather than Q6_K's 16 and needs no wider row. The scale is
/// `d * (0.5 + s) * 0.5`, staged as f32 already multiplied by `d`, for the same
/// parity reason as Q4_K. K must be a whole number of blocks.
///
/// Measured faster on its tile-parallel kernel than on the split-K pair even
/// where the geometric rule would pick the pair — see
/// `prefers_tile_parallel_fallback`.
pub(in crate::quant::cuda::quant_matmul) const IQ3_XXS: FeatMajorFormat = FeatMajorFormat {
    quant_format: QuantFormat::IQ3XXS,
    kernel_infix: "iq3_xxs",
    x_stride: 76,
    k_multiple: 256,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel_fallback: true,
    tile_parallel_key: "mmq_feat_major.iq3_xxs.prefers_tile_parallel",
    narrow_tile: false,
};

/// IQ3_S: 110-byte blocks of 256 elements, staged as Q8_0's row byte for byte —
/// 64 quant words plus 8 f32 group scales plus 4 ints of bank padding.
///
/// It shares IQ3_XXS's 4-component grid, reached by a 9-bit index: eight bits
/// from `qs` and a ninth from `qh`, selecting among 512 codebook points. Its
/// signs are EXPLICIT bits from `signs[32]`, one per element, as in IQ2_S.
///
/// Its 4-bit scale is packed two per byte across `scales[4]`, but indexed by
/// the 32-element GROUP, not by the pair of grid entries — so unlike IQ2_XS and
/// IQ2_S it keeps the 32-element granularity and stays on the Q8_0 row with 8
/// f32 scales. Its scale form is `d * (1 + 2 * s)`, which is not an affine
/// rewrite of the rest of the family's `d * (0.5 + s) * c`. The scale is staged
/// as f32 already multiplied by `d`, for the same parity reason as Q4_K. K must
/// be a whole number of blocks.
pub(in crate::quant::cuda::quant_matmul) const IQ3_S: FeatMajorFormat = FeatMajorFormat {
    quant_format: QuantFormat::IQ3S,
    kernel_infix: "iq3_s",
    x_stride: 76,
    k_multiple: 256,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel_fallback: false,
    tile_parallel_key: "mmq_feat_major.iq3_s.prefers_tile_parallel",
    narrow_tile: false,
};

/// IQ1_S: 50-byte blocks of 256 elements, staged as Q4_K's row int for int —
/// 64 quant words plus 8 `(f32, f32)` pairs plus 4 ints of bank padding. It is
/// the only i-quant that does NOT stage a `vec_dot_d`-shaped row.
///
/// Its 8-component grid is the IQ2 formats' width, but its entries are
/// ALREADY-SIGNED bytes in `{-1, 0, 1}`: there is no sign table and nothing to
/// fold. An 8-element sub-group takes its low 8 index bits from `qs` and three
/// more from its group's `qh` u16, an 11-bit index into 2048 points.
///
/// THE AFFINE VALUE. Its dequantized value is `dl * (g + delta)` with
/// `delta = +/- 0.125` — an affine transform, not a scale times an int8, so a
/// single scaled int32 dot cannot express it. Over one 32-element group the
/// contribution splits as `dl * dot(a, g) + dl * delta * sum(a)`, which is
/// exactly the two-term shape `mmqf_vec_dot_dm` already computes for Q4_K,
/// Q5_K, Q4_1 and Q5_1. So the kernel stages `(dl, dl * delta)` where those
/// formats stage `(d, m)` and reuses their `vec_dot` unchanged. The split is
/// exact: the block sum that second term multiplies is the int16 the
/// activation producer stores in its header word, not a rounded `half`.
///
/// The delta term is ADDITIVE, so the pair's second component is stored
/// unnegated — Q4_1's convention, not Q4_K's `-dmin * m`. The delta's own sign
/// comes from bit 15 of the group's `qh` word and is folded in at staging.
///
/// `dl` and `delta` are both constant across a whole 32-element group, which
/// is the granularity `mmqf_vec_dot_dm` indexes at, so the row carries 8 pairs
/// and needs no widening. Both components are f32, never `half`, for the same
/// parity reason as Q4_K. K must be a whole number of blocks.
pub(in crate::quant::cuda::quant_matmul) const IQ1_S: FeatMajorFormat = FeatMajorFormat {
    quant_format: QuantFormat::IQ1S,
    kernel_infix: "iq1_s",
    x_stride: 84,
    k_multiple: 256,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel_fallback: false,
    tile_parallel_key: "mmq_feat_major.iq1_s.prefers_tile_parallel",
    narrow_tile: false,
};

#[cfg(test)]
mod tests {
    use super::super::super::super::tiling::{Cadence, FEAT_TILE_DEFAULT, VARIANTS, smem_bytes};
    use super::super::super::kquant::Q4_K;
    use super::super::super::legacy::{Q4_1, Q8_0};
    use super::*;

    /// IQ3_XXS expands a 4-component grid, so a sign sub-group costs two grid
    /// points rather than one — but that is spent entirely inside staging. Its
    /// scale covers a whole 32-element group, the granularity
    /// `mmqf_vec_dot_d` already indexes at, so it lands on the Q8_0 row like
    /// IQ2_XXS and not on IQ2_XS's wider one. The two strides must therefore
    /// stay equal, and with them the family's shared-memory request at every
    /// token tile.
    #[test]
    fn the_iq3_xxs_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", IQ3_XXS.kernel_infix, 8),
            "quant_mmq_iq3_xxs_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", IQ3_XXS.kernel_infix, 128),
            "quant_mmq_iq3_xxs_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", IQ3_XXS.kernel_infix, 128),
            "quant_mmq_iq3_xxs_q8_1_mma_fixup_x128"
        );
        assert_eq!(IQ3_XXS.k_multiple, 256);
        assert_eq!(IQ3_XXS.x_stride, Q8_0.x_stride);
        const { assert!(IQ3_XXS.prefers_tile_parallel_fallback) };
        assert!(VARIANTS.iter().all(|&x| smem_bytes(
            &IQ3_XXS,
            FEAT_TILE_DEFAULT,
            x,
            Cadence::Halves
        ) == smem_bytes(
            &Q8_0,
            FEAT_TILE_DEFAULT,
            x,
            Cadence::Halves
        )));
    }

    /// IQ3_S shares IQ3_XXS's grid width and its 32-element scale granularity,
    /// so it stages the same Q8_0 row; its ninth index bit, its explicit signs
    /// and its `d * (1 + 2 * s)` scale form are all consumed during staging.
    /// The strides must therefore stay equal, and with them the family's
    /// shared-memory request at every token tile.
    #[test]
    fn the_iq3_s_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", IQ3_S.kernel_infix, 8),
            "quant_mmq_iq3_s_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", IQ3_S.kernel_infix, 128),
            "quant_mmq_iq3_s_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", IQ3_S.kernel_infix, 128),
            "quant_mmq_iq3_s_q8_1_mma_fixup_x128"
        );
        assert_eq!(IQ3_S.k_multiple, 256);
        assert_eq!(IQ3_S.x_stride, Q8_0.x_stride);
        assert_eq!(IQ3_S.x_stride, IQ3_XXS.x_stride);
        const { assert!(!IQ3_S.prefers_tile_parallel_fallback) };
        assert!(VARIANTS.iter().all(|&x| smem_bytes(
            &IQ3_S,
            FEAT_TILE_DEFAULT,
            x,
            Cadence::Halves
        ) == smem_bytes(
            &Q8_0,
            FEAT_TILE_DEFAULT,
            x,
            Cadence::Halves
        )));
    }

    /// IQ1_S is the family's only AFFINE format: its value is
    /// `dl * (grid + delta)`, so it cannot stage a `vec_dot_d` row. The split
    /// `dl * dot(a, g) + dl * delta * sum(a)` is per 32 elements, which is the
    /// scale/min shape `mmqf_vec_dot_dm` already consumes, so it stages Q4_K's
    /// row — and Q4_1's, whose ADDITIVE min sign it also takes. All three
    /// strides must stay equal, and with them the family's shared-memory
    /// request at every token tile.
    #[test]
    fn the_iq1_s_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", IQ1_S.kernel_infix, 8),
            "quant_mmq_iq1_s_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", IQ1_S.kernel_infix, 128),
            "quant_mmq_iq1_s_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", IQ1_S.kernel_infix, 128),
            "quant_mmq_iq1_s_q8_1_mma_fixup_x128"
        );
        assert_eq!(IQ1_S.k_multiple, 256);
        assert_eq!(IQ1_S.x_stride, Q4_K.x_stride);
        assert_eq!(IQ1_S.x_stride, Q4_1.x_stride);
        assert!(VARIANTS.iter().all(|&x| smem_bytes(
            &IQ1_S,
            FEAT_TILE_DEFAULT,
            x,
            Cadence::Halves
        ) == smem_bytes(
            &Q4_K,
            FEAT_TILE_DEFAULT,
            x,
            Cadence::Halves
        )));
    }
}
