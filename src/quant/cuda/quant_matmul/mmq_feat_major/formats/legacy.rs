use super::FeatMajorFormat;

/// Q8_0: 34-byte blocks of 32 elements, staged as 64 quant words plus 8 f32
/// scales plus 4 ints of bank padding.
///
/// Measured faster on its tile-parallel kernel than on the split-K pair even
/// where the geometric rule would pick the pair — see `prefers_tile_parallel`.
pub(in crate::quant::cuda::quant_matmul) const Q8_0: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "q8_0",
    x_stride: 76,
    k_multiple: 32,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel: true,
    narrow_tile: false,
};

/// Q4_0: 18-byte blocks of 32 elements, staged as Q8_0's row byte for byte —
/// 64 quant words plus 8 f32 scales plus 4 ints of bank padding. Q4_0's quants
/// are unsigned 4-bit biased by 8, and the kernel folds that bias in while
/// staging, so the staged row and the whole `vec_dot` are Q8_0's. K needs only
/// a whole 32-element block, so a row's last 256-k staging group can be
/// partial.
///
/// Measured faster on its tile-parallel kernel than on the split-K pair even
/// where the geometric rule would pick the pair — see `prefers_tile_parallel`.
pub(in crate::quant::cuda::quant_matmul) const Q4_0: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "q4_0",
    x_stride: 76,
    k_multiple: 32,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel: true,
    narrow_tile: false,
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
    narrow_tile: false,
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
    narrow_tile: false,
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
    narrow_tile: false,
};

/// IQ4_NL: 18-byte blocks of 32 elements, staged as Q8_0's row byte for byte —
/// 64 quant words plus 8 f32 scales plus 4 ints of bank padding. The family's
/// first i-quant: each 4-bit field is an INDEX into the 16-entry signed
/// codebook `KVALUES_IQ4NL`, and the kernel resolves it during staging, so the
/// staged lanes are signed int8 and the whole `vec_dot` is Q8_0's. K needs only
/// a whole 32-element block, so a row's last 256-k staging group can be
/// partial.
///
/// Measured faster on its tile-parallel kernel than on the split-K pair even
/// where the geometric rule would pick the pair — see `prefers_tile_parallel`.
pub(in crate::quant::cuda::quant_matmul) const IQ4_NL: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "iq4_nl",
    x_stride: 76,
    k_multiple: 32,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel: true,
    narrow_tile: false,
};

/// PQ2_0: 34-byte blocks of 128 elements, 2-bit codes under ONE f16 scale,
/// staged as Q8_0's row byte for byte — 64 quant words plus 8 f32 scales plus
/// 4 ints of bank padding. The kernel expands each code to the signed int8
/// `code - 1` while staging and writes the block's scale into all four of the
/// 32-element slots the block covers, so the staged row and the whole
/// `vec_dot` are Q8_0's. K needs only a whole 128-element block, so a row's
/// last 256-k staging group can be partial.
///
/// `prefers_tile_parallel` is UNMEASURED for this format and copied from
/// Q8_0, whose staged row and `vec_dot` it shares. Measure with
/// `cargo run --release --features cuda --example mmq_kernel_compare --
/// --format pq2_0 --n 5120 --k 5120 --m 8 --split-k 2`, compare the
/// tile-parallel and split-K lines, and flip this if the pair wins.
pub(in crate::quant::cuda::quant_matmul) const PQ2_0: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "pq2_0",
    x_stride: 76,
    k_multiple: 128,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel: true,
    narrow_tile: false,
};

/// Q2_0: 18-byte blocks of 64 elements, 2-bit codes under ONE f16 scale,
/// staged as Q8_0's row byte for byte. PQ2_0's code space at half the run
/// length: the kernel expands the codes the same way and writes the block's
/// scale into both 32-element slots it covers. K needs only a whole
/// 64-element block, so a row's last 256-k staging group can be partial.
///
/// `prefers_tile_parallel` is UNMEASURED for this format and copied from
/// Q8_0; measure as the PQ2_0 descriptor says, with `--format q2_0`.
pub(in crate::quant::cuda::quant_matmul) const Q2_0: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "q2_0",
    x_stride: 76,
    k_multiple: 64,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel: true,
    narrow_tile: false,
};

/// Q1_0: 18-byte blocks of 128 elements, one sign bit per element under ONE
/// f16 scale, staged as Q8_0's row byte for byte. The kernel expands each bit
/// to the signed int8 `+1`/`-1` while staging and writes the block's scale
/// into all four 32-element slots it covers. K needs only a whole 128-element
/// block, so a row's last 256-k staging group can be partial.
///
/// `prefers_tile_parallel` is UNMEASURED for this format and copied from
/// Q8_0; measure as the PQ2_0 descriptor says, with `--format q1_0`.
pub(in crate::quant::cuda::quant_matmul) const Q1_0: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "q1_0",
    x_stride: 76,
    k_multiple: 128,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel: true,
    narrow_tile: false,
};

#[cfg(test)]
mod tests {
    use super::super::super::tiling::{Cadence, FEAT_TILE_DEFAULT, VARIANTS, smem_bytes};
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
        assert!(VARIANTS.iter().all(
            |&x| smem_bytes(&Q4_0, FEAT_TILE_DEFAULT, x, Cadence::Halves)
                == smem_bytes(&Q8_0, FEAT_TILE_DEFAULT, x, Cadence::Halves)
        ));
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
        assert!(VARIANTS.iter().all(
            |&x| smem_bytes(&Q4_1, FEAT_TILE_DEFAULT, x, Cadence::Halves)
                == smem_bytes(&Q4_K, FEAT_TILE_DEFAULT, x, Cadence::Halves)
        ));
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
        assert!(VARIANTS.iter().all(
            |&x| smem_bytes(&Q5_0, FEAT_TILE_DEFAULT, x, Cadence::Halves)
                == smem_bytes(&Q8_0, FEAT_TILE_DEFAULT, x, Cadence::Halves)
        ));
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
        assert!(VARIANTS.iter().all(
            |&x| smem_bytes(&Q5_1, FEAT_TILE_DEFAULT, x, Cadence::Halves)
                == smem_bytes(&Q4_1, FEAT_TILE_DEFAULT, x, Cadence::Halves)
        ));
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
        assert!(VARIANTS.iter().all(|&x| smem_bytes(
            &IQ4_NL,
            FEAT_TILE_DEFAULT,
            x,
            Cadence::Halves
        ) == smem_bytes(
            &Q8_0,
            FEAT_TILE_DEFAULT,
            x,
            Cadence::Halves
        )));
        // One of the measured tile-parallel opt-outs.
        const { assert!(IQ4_NL.prefers_tile_parallel) };
    }

    /// The three prism formats stage into the Q8_0 row, so their strides must
    /// stay equal to it and with them the family's shared-memory request at
    /// every token tile. Their K multiple is one weight block — 128, 64, 128
    /// — which is finer than a 256-k group, so each takes the ragged tail.
    #[test]
    fn the_prism_descriptors_name_the_compiled_symbols() {
        let prism: [(&FeatMajorFormat, &str, u32); 3] = [
            (&PQ2_0, "pq2_0", 128),
            (&Q2_0, "q2_0", 64),
            (&Q1_0, "q1_0", 128),
        ];
        for (fm, infix, k_multiple) in prism {
            assert_eq!(fm.kernel_infix, infix);
            assert_eq!(
                format!("quant_mmq_{}_q8_1_mma_x{}", fm.kernel_infix, 8),
                format!("quant_mmq_{infix}_q8_1_mma_x8")
            );
            assert_eq!(
                format!("quant_mmq_{}_q8_1_mma_sk_x{}", fm.kernel_infix, 128),
                format!("quant_mmq_{infix}_q8_1_mma_sk_x128")
            );
            assert_eq!(
                format!("quant_mmq_{}_q8_1_mma_fixup_x{}", fm.kernel_infix, 128),
                format!("quant_mmq_{infix}_q8_1_mma_fixup_x128")
            );
            assert_eq!(fm.k_multiple, k_multiple);
            assert!(256u32.is_multiple_of(fm.k_multiple));
            assert_eq!(fm.x_stride, Q8_0.x_stride);
            assert_eq!(fm.act_scratch_ints_per_token, 0);
            assert!(!fm.narrow_tile);
            assert!(VARIANTS.iter().all(|&x| smem_bytes(
                fm,
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
        // Unmeasured: copied from Q8_0 until `mmq_kernel_compare --split-k`
        // says otherwise.
        const { assert!(PQ2_0.prefers_tile_parallel) };
        const { assert!(Q2_0.prefers_tile_parallel) };
        const { assert!(Q1_0.prefers_tile_parallel) };
    }
}
