//! TCF native encodings in the feature-major family.

use super::FeatMajorFormat;

/// TCF `Q8S32T64`: whole-tensor planes rather than blocks — a dense row-major
/// int8 code plane, then a dense row-major `binary16` scale plane with one
/// scale per 32 elements and nothing between the two. The encoding is
/// symmetric with a flat scale form, so it stages into the Q8_0 row byte for
/// byte — 64 quant words plus 8 f32 scales plus 4 ints of bank padding — and
/// shares Q8_0's `vec_dot`. Only the addressing in `stage` is its own.
///
/// `k_multiple` is 64: the encoding's execution tile is 64 logical elements
/// and the last dimension must be a whole number of them. A row's last 256-k
/// staging group can still be partial.
///
/// `prefers_tile_parallel` is a measured flag and this format has not been
/// measured — see its doc. Re-measure: run the kernel-comparison example with
/// `--stream-k` at small `m` and flip this if the tile-parallel run wins
/// outside noise.
pub(in crate::quant::cuda::quant_matmul) const TCF_Q8S32T64: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "tcf_q8s32t64",
    x_stride: 76,
    k_multiple: 64,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel: false,
};

#[cfg(test)]
mod tests {
    use super::super::super::dispatch::{VARIANTS, smem_bytes};
    use super::super::legacy::Q8_0;
    use super::*;

    /// The TCF encoding stages into the Q8_0 row, so the two strides must stay
    /// equal and with them the family's shared-memory request at every token
    /// tile. Its tile is 64 elements wide, so it takes the wider K multiple.
    #[test]
    fn the_tcf_q8s32t64_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", TCF_Q8S32T64.kernel_infix, 8),
            "quant_mmq_tcf_q8s32t64_q8_1_mma_x8"
        );
        assert_eq!(
            format!(
                "quant_mmq_{}_q8_1_mma_sk_x{}",
                TCF_Q8S32T64.kernel_infix, 128
            ),
            "quant_mmq_tcf_q8s32t64_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!(
                "quant_mmq_{}_q8_1_mma_fixup_x{}",
                TCF_Q8S32T64.kernel_infix, 128
            ),
            "quant_mmq_tcf_q8s32t64_q8_1_mma_fixup_x128"
        );
        assert_eq!(TCF_Q8S32T64.k_multiple, 64);
        assert_eq!(TCF_Q8S32T64.x_stride, Q8_0.x_stride);
        // The staged row's bank rule, asserted in `mmqf_body` as well: any
        // multiple of 8 puts consecutive rows in one 128-bit segment.
        assert_eq!(TCF_Q8S32T64.x_stride % 8, 4);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| smem_bytes(&TCF_Q8S32T64, x) == smem_bytes(&Q8_0, x))
        );
        // Not measured against stream-k yet.
        const { assert!(!TCF_Q8S32T64.prefers_tile_parallel) };
    }
}
