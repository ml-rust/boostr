//! I-quants that resolve through a flat 16-entry codebook via `__byte_perm`.

use super::super::FeatMajorFormat;

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
    prefers_tile_parallel: false,
};

#[cfg(test)]
mod tests {
    use super::super::super::super::dispatch::{VARIANTS, smem_bytes};
    use super::super::super::legacy::Q8_0;
    use super::*;

    /// IQ4_XS is the first 256-element format in the family that stages the
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
        const { assert!(!IQ4_XS.prefers_tile_parallel) };
        assert!(
            VARIANTS
                .iter()
                .all(|&x| smem_bytes(&IQ4_XS, x) == smem_bytes(&Q8_0, x))
        );
    }
}
