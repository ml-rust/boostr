use super::FeatMajorFormat;

/// Q4_K: 144-byte super-blocks of 256 elements, staged as 64 quant words plus
/// 8 `float2` scale/min pairs (16 ints) plus 4 ints of bank padding. The row is
/// 8 ints wider than Q8_0's because the pair is f32, not `half2`: half rounding
/// on `d * sc` perturbs every 32-element sub-block and pushed the GEMM path
/// outside the GEMV parity bound. K must be a whole number of super-blocks,
/// which also makes every 256-k staging group whole.
pub(in crate::quant::cuda::quant_matmul) const Q4_K: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "q4_k",
    x_stride: 84,
    k_multiple: 256,
    act_scratch_ints_per_token: 0,
};

/// Q5_K: 176-byte super-blocks of 256 elements, staged exactly as Q4_K — 64
/// quant words plus 8 `float2` scale/min pairs (16 ints) plus 4 ints of bank
/// padding. Q5_K is Q4_K with a fifth quant bit from a 32-byte `qh` field, so
/// only the kernel's staging step differs; the staged row and the two-term
/// arithmetic are shared. K must be a whole number of super-blocks.
pub(in crate::quant::cuda::quant_matmul) const Q5_K: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "q5_k",
    x_stride: 84,
    k_multiple: 256,
    act_scratch_ints_per_token: 0,
};

/// Q6_K: 210-byte super-blocks of 256 elements, staged as 64 quant words plus
/// 16 f32 group scales plus 4 ints of bank padding — the same stride as Q4_K,
/// reached by a different split. Q6_K's scale changes every 16 elements, so it
/// stages 16 `d * scale` floats per row rather than 8 scale/min pairs, and the
/// consumer runs two 16-k MMAs per 32-k step. The staged scale is f32 and
/// already multiplied by `d`, for the same parity reason as Q4_K. K must be a
/// whole number of super-blocks.
pub(in crate::quant::cuda::quant_matmul) const Q6_K: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "q6_k",
    x_stride: 84,
    k_multiple: 256,
    act_scratch_ints_per_token: 0,
};

/// Q3_K: 110-byte super-blocks of 256 elements, staged as Q6_K's row int for
/// int — 64 quant words plus 16 f32 group scales plus 4 ints of bank padding.
/// Q3_K shares Q6_K's 16-element scale granularity and its absence of a
/// minimum term, so only the kernel's staging step differs: the quant is two
/// low bits from `qs` plus one INVERTED high bit from `hmask`, biased during
/// staging to a signed [-4, 3] lane. K must be a whole number of super-blocks.
pub(in crate::quant::cuda::quant_matmul) const Q3_K: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "q3_k",
    x_stride: 84,
    k_multiple: 256,
    act_scratch_ints_per_token: 0,
};

/// Q2_K: 84-byte super-blocks of 256 elements, staged as 64 quant words plus
/// 16 `float2` scale/min pairs (32 ints) plus 4 ints of bank padding — the
/// widest row in the family. Q2_K is the only format with BOTH a 16-element
/// scale granularity and a minimum term, so it stages twice Q4_K's scale
/// record and its `vec_dot` runs Q6_K's two 16-k MMAs per 32-k step plus a
/// minimum correction at 16-element granularity.
///
/// That correction needs the activation's sum over each 16 elements, while the
/// shared activation record stores an exact sum per 32. The kernel derives the
/// split into `act_scratch_ints_per_token` ints of shared scratch, once per
/// staged activation tile. K must be a whole number of super-blocks.
///
/// OCCUPANCY. The wider row plus the scratch is the family's largest
/// shared-memory request, so where a device's per-unit shared memory sits at
/// the low end of what this path supports, Q2_K can hold one resident block at
/// token tiles where the 84-stride formats hold two. That is accepted for this
/// format rather than paid for by rounding `d * sc` through `half`, which is
/// what the GEMM/GEMV parity bound rejects.
pub(in crate::quant::cuda::quant_matmul) const Q2_K: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "q2_k",
    x_stride: 100,
    k_multiple: 256,
    act_scratch_ints_per_token: 4,
};

/// IQ4_XS: 136-byte super-blocks of 256 elements, staged as Q8_0's row byte for
/// byte — 64 quant words plus 8 f32 sub-block scales plus 4 ints of bank
/// padding. IQ4_XS is IQ4_NL's 16-entry signed codebook over a super-block: the
/// kernel resolves each 4-bit index during staging, so the staged lanes are
/// signed int8 and the whole `vec_dot` is Q8_0's. Its scale changes every 32
/// elements, which is the granularity that `vec_dot` already indexes at, so it
/// stages 8 f32 rather than Q6_K's 16 and needs no wider row. The staged scale
/// is f32 and already multiplied by `d`, for the same parity reason as Q4_K. K
/// must be a whole number of super-blocks.
pub(in crate::quant::cuda::quant_matmul) const IQ4_XS: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "iq4_xs",
    x_stride: 76,
    k_multiple: 256,
    act_scratch_ints_per_token: 0,
};

/// IQ2_XXS: 66-byte blocks of 256 elements, staged as Q8_0's row byte for byte
/// — 64 quant words plus 8 f32 sub-block scales plus 4 ints of bank padding.
/// The family's first GRID-INDEXED format: each byte of `qs` is an index into
/// a 256-entry codebook whose entry expands to EIGHT magnitude bytes, and a
/// 7-bit field of the same group's `aux` word indexes a sign table that
/// supplies one bit per expanded component. The kernel expands the point and
/// folds the sign in while staging, so the staged lanes are signed int8 and
/// the whole `vec_dot` is Q8_0's. Its scale changes every 32 elements, which
/// is the granularity that `vec_dot` already indexes at, so it stages 8 f32
/// rather than Q6_K's 16 and needs no wider row. The staged scale is f32 and
/// already multiplied by `d`, for the same parity reason as Q4_K. K must be a
/// whole number of blocks.
pub(in crate::quant::cuda::quant_matmul) const IQ2_XXS: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "iq2_xxs",
    x_stride: 76,
    k_multiple: 256,
    act_scratch_ints_per_token: 0,
};

/// IQ2_XS: 74-byte blocks of 256 elements, staged as Q6_K's row int for int —
/// 64 quant words plus 16 f32 group scales plus 4 ints of bank padding. Each of
/// the 32 `u16` in `qs` packs a 9-bit index into a 512-entry codebook, whose
/// entry expands to EIGHT magnitude bytes, under a 7-bit index into a sign
/// table that supplies one bit per expanded component. The kernel expands the
/// point and folds the sign in while staging, so the staged lanes are signed
/// int8, exactly as for IQ2_XXS.
///
/// The row is Q6_K's rather than Q8_0's because the scale granularity differs
/// from IQ2_XXS's: the 4-bit scale is packed two to a `scales` byte, one per
/// two grid entries, so it changes every 16 elements and a single 32-k MMA
/// cannot express it. The staged scale is f32 and already multiplied by `d`,
/// for the same parity reason as Q4_K. K must be a whole number of blocks.
pub(in crate::quant::cuda::quant_matmul) const IQ2_XS: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "iq2_xs",
    x_stride: 84,
    k_multiple: 256,
    act_scratch_ints_per_token: 0,
};

/// IQ2_S: 82-byte blocks of 256 elements, staged as Q6_K's row int for int —
/// 64 quant words plus 16 f32 group scales plus 4 ints of bank padding. Each of
/// the 32 entries takes eight index bits from `qs` and two more from `qh`,
/// selecting among 1024 codebook points of eight magnitude bytes each.
///
/// Its signs are EXPLICIT bits — one byte of eight per entry, from `signs[32]`
/// — with no sign-table indirection, which is the one way it diverges from
/// IQ2_XXS and IQ2_XS. Everything after that byte is reached is shared: the
/// kernel folds the sign in while staging, so the staged lanes are signed int8.
/// Its scale packing is IQ2_XS's exactly, so it carries the same 16-element
/// granularity and the same Q6_K row. The staged scale is f32 and already
/// multiplied by `d`, for the same parity reason as Q4_K. K must be a whole
/// number of blocks.
pub(in crate::quant::cuda::quant_matmul) const IQ2_S: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "iq2_s",
    x_stride: 84,
    k_multiple: 256,
    act_scratch_ints_per_token: 0,
};

#[cfg(test)]
mod tests {
    use super::super::super::dispatch::{FEAT_TILE, VARIANTS, smem_bytes, smem_opt_in_limit};
    use super::super::legacy::{Q4_0, Q4_1, Q5_0, Q5_1, Q8_0};
    use super::*;

    #[test]
    fn the_q4_k_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", Q4_K.kernel_infix, 8),
            "quant_mmq_q4_k_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", Q4_K.kernel_infix, 128),
            "quant_mmq_q4_k_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", Q4_K.kernel_infix, 128),
            "quant_mmq_q4_k_q8_1_mma_fixup_x128"
        );
        assert_eq!(Q4_K.k_multiple, 256);
    }

    #[test]
    fn the_q5_k_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", Q5_K.kernel_infix, 8),
            "quant_mmq_q5_k_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", Q5_K.kernel_infix, 128),
            "quant_mmq_q5_k_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", Q5_K.kernel_infix, 128),
            "quant_mmq_q5_k_q8_1_mma_fixup_x128"
        );
        assert_eq!(Q5_K.k_multiple, 256);
    }

    /// Q5_K stages the Q4_K row verbatim — same quant words, same eight
    /// scale/min pairs — so the two strides must stay equal, and with them the
    /// family's shared-memory request at every token tile.
    #[test]
    fn q5_k_shares_the_q4_k_row_stride() {
        assert_eq!(Q5_K.x_stride, Q4_K.x_stride);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| smem_bytes(&Q5_K, x) == smem_bytes(&Q4_K, x))
        );
    }

    #[test]
    fn the_q6_k_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", Q6_K.kernel_infix, 8),
            "quant_mmq_q6_k_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", Q6_K.kernel_infix, 128),
            "quant_mmq_q6_k_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", Q6_K.kernel_infix, 128),
            "quant_mmq_q6_k_q8_1_mma_fixup_x128"
        );
        assert_eq!(Q6_K.k_multiple, 256);
    }

    /// Q6_K reaches Q4_K's row stride by a different split: 16 f32 group
    /// scales rather than 8 scale/min pairs. Equal strides mean the two share
    /// the family's shared-memory request at every token tile.
    #[test]
    fn q6_k_shares_the_q4_k_row_stride() {
        assert_eq!(Q6_K.x_stride, Q4_K.x_stride);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| smem_bytes(&Q6_K, x) == smem_bytes(&Q4_K, x))
        );
    }

    #[test]
    fn the_q3_k_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", Q3_K.kernel_infix, 8),
            "quant_mmq_q3_k_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", Q3_K.kernel_infix, 128),
            "quant_mmq_q3_k_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", Q3_K.kernel_infix, 128),
            "quant_mmq_q3_k_q8_1_mma_fixup_x128"
        );
        assert_eq!(Q3_K.k_multiple, 256);
    }

    /// Q3_K stages the Q6_K row verbatim — same 64 quant words, same 16 f32
    /// group scales — so the two strides must stay equal, and with them the
    /// family's shared-memory request at every token tile.
    #[test]
    fn q3_k_shares_the_q6_k_row_stride() {
        assert_eq!(Q3_K.x_stride, Q6_K.x_stride);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| smem_bytes(&Q3_K, x) == smem_bytes(&Q6_K, x))
        );
    }

    /// Q4_K's weight row is 8 ints wider than Q8_0's: it stages the scale/min
    /// pair as two f32 rather than one `half2`, because half rounding on
    /// `d * sc` broke the GEMM/GEMV parity bound. The 8 extra ints per row
    /// across the 128-row weight tile are the whole difference in the request,
    /// and it is the same at every token tile because only the activation tile
    /// scales with `mmq_x`.
    #[test]
    fn q4_k_costs_one_extra_scale_word_per_row_over_q8_0() {
        const EXTRA: u32 = 4 * FEAT_TILE * 8;
        assert_eq!(Q4_K.x_stride, Q8_0.x_stride + 8);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| { smem_bytes(&Q4_K, x) == smem_bytes(&Q8_0, x) + EXTRA })
        );
        // The widest variant must still fit what a device grants on opt-in.
        // 96KB per unit is the smallest sm_80-or-later figure the family runs
        // on, and the launcher subtracts the driver's reservation from it.
        assert!(smem_bytes(&Q4_K, 128) <= smem_opt_in_limit(96 * 1024));
    }

    #[test]
    fn the_q2_k_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", Q2_K.kernel_infix, 8),
            "quant_mmq_q2_k_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", Q2_K.kernel_infix, 128),
            "quant_mmq_q2_k_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", Q2_K.kernel_infix, 128),
            "quant_mmq_q2_k_q8_1_mma_fixup_x128"
        );
        assert_eq!(Q2_K.k_multiple, 256);
    }

    /// Q2_K is the only format that asks for activation scratch, and the only
    /// one whose row carries 16 scale/min pairs. Both costs are asserted here
    /// so a change to either shows up as a failing shared-memory figure rather
    /// than as a kernel reading past what the launcher opted in to.
    #[test]
    fn q2_k_is_the_only_format_with_activation_scratch() {
        for f in [
            &Q8_0, &Q4_0, &Q4_1, &Q5_0, &Q5_1, &Q4_K, &Q5_K, &Q6_K, &Q3_K,
        ] {
            assert_eq!(f.act_scratch_ints_per_token, 0);
        }
        assert_eq!(Q2_K.act_scratch_ints_per_token, 4);
        // 64 quant words + 32 ints of scale/min pairs + 4 ints of padding.
        assert_eq!(Q2_K.x_stride, 100);
        // The family's bank-padding rule, asserted in the kernel as well.
        assert_eq!(Q2_K.x_stride % 8, 4);
        // Weight row cost over Q4_K, plus the scratch, at every token tile.
        const EXTRA_ROW: u32 = 4 * FEAT_TILE * 16;
        assert!(
            VARIANTS
                .iter()
                .all(|&x| { smem_bytes(&Q2_K, x) == smem_bytes(&Q4_K, x) + EXTRA_ROW + 4 * x * 4 })
        );
        // The widest variant must still fit what a device grants on opt-in.
        assert!(smem_bytes(&Q2_K, 128) <= smem_opt_in_limit(96 * 1024));
    }

    /// IQ4_XS is the only 256-element format in the family that stages the
    /// Q8_0 row: its codebook values are signed int8 and its scale granularity
    /// is the 32 elements `mmqf_vec_dot_d` already indexes at, so it needs
    /// neither a bias nor a wider scale record. The two strides must therefore
    /// stay equal, and with them the family's shared-memory request at every
    /// token tile.
    #[test]
    fn the_iq4_xs_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", IQ4_XS.kernel_infix, 8),
            "quant_mmq_iq4_xs_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", IQ4_XS.kernel_infix, 128),
            "quant_mmq_iq4_xs_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", IQ4_XS.kernel_infix, 128),
            "quant_mmq_iq4_xs_q8_1_mma_fixup_x128"
        );
        assert_eq!(IQ4_XS.k_multiple, 256);
        assert_eq!(IQ4_XS.x_stride, Q8_0.x_stride);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| smem_bytes(&IQ4_XS, x) == smem_bytes(&Q8_0, x))
        );
    }

    /// IQ2_XXS stages the Q8_0 row like IQ4_XS: staging expands its grid point
    /// to signed int8 and folds the sign in, and its scale granularity is the
    /// 32 elements `mmqf_vec_dot_d` already indexes at, so it needs neither a
    /// bias nor a wider scale record. The two strides must therefore stay
    /// equal, and with them the family's shared-memory request at every token
    /// tile.
    #[test]
    fn the_iq2_xxs_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", IQ2_XXS.kernel_infix, 8),
            "quant_mmq_iq2_xxs_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", IQ2_XXS.kernel_infix, 128),
            "quant_mmq_iq2_xxs_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", IQ2_XXS.kernel_infix, 128),
            "quant_mmq_iq2_xxs_q8_1_mma_fixup_x128"
        );
        assert_eq!(IQ2_XXS.k_multiple, 256);
        assert_eq!(IQ2_XXS.x_stride, Q8_0.x_stride);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| smem_bytes(&IQ2_XXS, x) == smem_bytes(&Q8_0, x))
        );
    }

    /// IQ2_XS expands the same 8-component grid as IQ2_XXS, so its quant words
    /// are Q8_0's, but its 4-bit scale is packed two to a `scales` byte, one
    /// per two grid entries — a change every 16 elements, not every 32. That
    /// puts it on Q6_K's row and Q6_K's `vec_dot`, so the two strides must stay
    /// equal, and with them the family's shared-memory request at every token
    /// tile.
    #[test]
    fn the_iq2_xs_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", IQ2_XS.kernel_infix, 8),
            "quant_mmq_iq2_xs_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", IQ2_XS.kernel_infix, 128),
            "quant_mmq_iq2_xs_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", IQ2_XS.kernel_infix, 128),
            "quant_mmq_iq2_xs_q8_1_mma_fixup_x128"
        );
        assert_eq!(IQ2_XS.k_multiple, 256);
        assert_eq!(IQ2_XS.x_stride, Q6_K.x_stride);
        assert_eq!(IQ2_XS.x_stride, IQ2_XXS.x_stride + 8);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| smem_bytes(&IQ2_XS, x) == smem_bytes(&Q6_K, x))
        );
    }

    /// IQ2_S packs its scales exactly as IQ2_XS does, so it stages the same
    /// Q6_K row; only the sign source differs, and that is consumed during
    /// staging. The three strides must therefore stay equal, and with them the
    /// family's shared-memory request at every token tile.
    #[test]
    fn the_iq2_s_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", IQ2_S.kernel_infix, 8),
            "quant_mmq_iq2_s_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", IQ2_S.kernel_infix, 128),
            "quant_mmq_iq2_s_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", IQ2_S.kernel_infix, 128),
            "quant_mmq_iq2_s_q8_1_mma_fixup_x128"
        );
        assert_eq!(IQ2_S.k_multiple, 256);
        assert_eq!(IQ2_S.x_stride, Q6_K.x_stride);
        assert_eq!(IQ2_S.x_stride, IQ2_XS.x_stride);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| smem_bytes(&IQ2_S, x) == smem_bytes(&Q6_K, x))
        );
    }
}
