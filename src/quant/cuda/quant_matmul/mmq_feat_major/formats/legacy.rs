use super::FeatMajorFormat;

/// Q8_0: 34-byte blocks of 32 elements, staged as 64 quant words plus 8 f32
/// scales plus 4 ints of bank padding.
///
/// Measured faster on its tile-parallel kernel than on stream-k even where the
/// geometric rule would pick stream-k — see `prefers_tile_parallel`'s doc.
pub(in crate::quant::cuda::quant_matmul) const Q8_0: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "q8_0",
    x_stride: 76,
    k_multiple: 32,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel: true,
};

/// Q4_0: 18-byte blocks of 32 elements, staged as Q8_0's row byte for byte —
/// 64 quant words plus 8 f32 scales plus 4 ints of bank padding. Q4_0's quants
/// are unsigned 4-bit biased by 8, and the kernel folds that bias in while
/// staging, so the staged row and the whole `vec_dot` are Q8_0's. K needs only
/// a whole 32-element block, so a row's last 256-k staging group can be
/// partial.
///
/// Measured faster on its tile-parallel kernel than on stream-k even where the
/// geometric rule would pick stream-k — see `prefers_tile_parallel`'s doc.
pub(in crate::quant::cuda::quant_matmul) const Q4_0: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "q4_0",
    x_stride: 76,
    k_multiple: 32,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel: true,
};

/// Q4_1: 20-byte blocks of 32 elements, staged as Q4_K's row int for int — 64
/// quant words plus 8 `float2` scale/min pairs (16 ints) plus 4 ints of bank
/// padding. Q4_1's quants are unsigned 4-bit and its value is `d * q + m`, so
/// the pair staged is `(d, +m)`: the same `vec_dot` as Q4_K, whose value is
/// `d * sc * q - dmin * m` and which therefore stages a NEGATED minimum. One
/// pair covers a whole 32-element block, which is the granularity that
/// `vec_dot` already indexes at. K needs only a whole 32-element block, so a
/// row's last 256-k staging group can be partial.
pub(in crate::quant::cuda::quant_matmul) const Q4_1: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "q4_1",
    x_stride: 84,
    k_multiple: 32,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel: false,
};

/// Q5_0: 22-byte blocks of 32 elements, staged as Q8_0's row byte for byte —
/// 64 quant words plus 8 f32 scales plus 4 ints of bank padding. Q5_0's quant
/// is 4 bits from `qs` plus a fifth from the block's 32-bit `qh`, biased by 16
/// while staging, so the staged row and the whole `vec_dot` are Q8_0's. K needs
/// only a whole 32-element block, so a row's last 256-k staging group can be
/// partial.
pub(in crate::quant::cuda::quant_matmul) const Q5_0: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "q5_0",
    x_stride: 76,
    k_multiple: 32,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel: false,
};

/// Q5_1: 24-byte blocks of 32 elements, staged as Q4_1's row int for int. Q5_1
/// is Q4_1 with a fifth quant bit from a 32-bit `qh`, so only the kernel's
/// staging step differs; the staged row, the `+m` minimum and the two-term
/// arithmetic are shared. K needs only a whole 32-element block, so a row's
/// last 256-k staging group can be partial.
pub(in crate::quant::cuda::quant_matmul) const Q5_1: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "q5_1",
    x_stride: 84,
    k_multiple: 32,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel: false,
};

/// IQ4_NL: 18-byte blocks of 32 elements, staged as Q8_0's row byte for byte —
/// 64 quant words plus 8 f32 scales plus 4 ints of bank padding. The family's
/// first i-quant: each 4-bit field is an INDEX into the 16-entry signed
/// codebook `KVALUES_IQ4NL`, and the kernel resolves it during staging, so the
/// staged lanes are signed int8 and the whole `vec_dot` is Q8_0's. K needs only
/// a whole 32-element block, so a row's last 256-k staging group can be
/// partial.
///
/// Measured faster on its tile-parallel kernel than on stream-k even where the
/// geometric rule would pick stream-k — see `prefers_tile_parallel`'s doc.
pub(in crate::quant::cuda::quant_matmul) const IQ4_NL: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "iq4_nl",
    x_stride: 76,
    k_multiple: 32,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel: true,
};

#[cfg(test)]
mod tests {
    use super::super::super::dispatch::{VARIANTS, smem_bytes};
    use super::super::kquant::Q4_K;
    use super::*;

    #[test]
    fn the_q8_0_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", Q8_0.kernel_infix, 8),
            "quant_mmq_q8_0_q8_1_mma_x8"
        );
        assert_eq!(Q8_0.k_multiple, 32);
        // One of the measured tile-parallel opt-outs.
        const { assert!(Q8_0.prefers_tile_parallel) };
    }

    /// Q4_0 stages into the Q8_0 row, so the two strides must stay equal and
    /// with them the family's shared-memory request at every token tile. Both
    /// are 32-element block formats, so both take the ragged-K multiple.
    #[test]
    fn the_q4_0_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", Q4_0.kernel_infix, 8),
            "quant_mmq_q4_0_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", Q4_0.kernel_infix, 128),
            "quant_mmq_q4_0_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", Q4_0.kernel_infix, 128),
            "quant_mmq_q4_0_q8_1_mma_fixup_x128"
        );
        assert_eq!(Q4_0.k_multiple, 32);
        assert_eq!(Q4_0.x_stride, Q8_0.x_stride);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| smem_bytes(&Q4_0, x) == smem_bytes(&Q8_0, x))
        );
        // One of the measured tile-parallel opt-outs.
        const { assert!(Q4_0.prefers_tile_parallel) };
    }

    /// Q4_1 stages into the Q4_K row, so the two strides must stay equal and
    /// with them the family's shared-memory request at every token tile. Q4_1
    /// is a 32-element block format, so it takes the ragged-K multiple.
    #[test]
    fn the_q4_1_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", Q4_1.kernel_infix, 8),
            "quant_mmq_q4_1_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", Q4_1.kernel_infix, 128),
            "quant_mmq_q4_1_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", Q4_1.kernel_infix, 128),
            "quant_mmq_q4_1_q8_1_mma_fixup_x128"
        );
        assert_eq!(Q4_1.k_multiple, 32);
        assert_eq!(Q4_1.x_stride, Q4_K.x_stride);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| smem_bytes(&Q4_1, x) == smem_bytes(&Q4_K, x))
        );
    }

    /// Q5_0 stages into the Q8_0 row, so the two strides must stay equal and
    /// with them the family's shared-memory request at every token tile. Q5_0
    /// is a 32-element block format, so it takes the ragged-K multiple.
    #[test]
    fn the_q5_0_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", Q5_0.kernel_infix, 8),
            "quant_mmq_q5_0_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", Q5_0.kernel_infix, 128),
            "quant_mmq_q5_0_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", Q5_0.kernel_infix, 128),
            "quant_mmq_q5_0_q8_1_mma_fixup_x128"
        );
        assert_eq!(Q5_0.k_multiple, 32);
        assert_eq!(Q5_0.x_stride, Q8_0.x_stride);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| smem_bytes(&Q5_0, x) == smem_bytes(&Q8_0, x))
        );
    }

    /// Q5_1 stages the Q4_1 row verbatim — same quant words, same eight
    /// scale/min pairs — so the two strides must stay equal, and with them the
    /// family's shared-memory request at every token tile.
    #[test]
    fn the_q5_1_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", Q5_1.kernel_infix, 8),
            "quant_mmq_q5_1_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", Q5_1.kernel_infix, 128),
            "quant_mmq_q5_1_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", Q5_1.kernel_infix, 128),
            "quant_mmq_q5_1_q8_1_mma_fixup_x128"
        );
        assert_eq!(Q5_1.k_multiple, 32);
        assert_eq!(Q5_1.x_stride, Q4_1.x_stride);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| smem_bytes(&Q5_1, x) == smem_bytes(&Q4_1, x))
        );
        const { assert!(!Q5_1.prefers_tile_parallel) };
    }

    /// IQ4_NL stages into the Q8_0 row — the codebook values are signed int8,
    /// so the resolved lanes need no bias and no extra scale record — so the
    /// two strides must stay equal and with them the family's shared-memory
    /// request at every token tile. It is a 32-element block format, so it
    /// takes the ragged-K multiple.
    #[test]
    fn the_iq4_nl_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", IQ4_NL.kernel_infix, 8),
            "quant_mmq_iq4_nl_q8_1_mma_x8"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_sk_x{}", IQ4_NL.kernel_infix, 128),
            "quant_mmq_iq4_nl_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_fixup_x{}", IQ4_NL.kernel_infix, 128),
            "quant_mmq_iq4_nl_q8_1_mma_fixup_x128"
        );
        assert_eq!(IQ4_NL.k_multiple, 32);
        assert_eq!(IQ4_NL.x_stride, Q8_0.x_stride);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| smem_bytes(&IQ4_NL, x) == smem_bytes(&Q8_0, x))
        );
        // One of the measured tile-parallel opt-outs.
        const { assert!(IQ4_NL.prefers_tile_parallel) };
    }
}
