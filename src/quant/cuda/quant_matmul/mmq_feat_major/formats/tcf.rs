//! TCF native encodings in the feature-major family.

use tcf_core::NativeEncoding;

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

/// TCF `Q4AS32DT64`: 4 bits, a 32-element group, and a two-level `u6`/`m6`
/// scale form over a 256-element super-block that carries both a super-scale
/// and a super-minimum — 4.50 bpw, the same budget as its GGUF competitor
/// `Q4_K`. The planes are whole-tensor and abut: 4-bit codes, the bit-packed
/// 6-bit sub-scales, the bit-packed 6-bit sub-minima, the `bfloat16`
/// super-scales, the `bfloat16` super-minima.
///
/// The staged row is Q4_K's byte for byte — 64 quant words, 8 `float2`
/// scale/min pairs, 4 ints of bank padding — and it shares Q4_K's `vec_dot`.
/// Only the addressing and the three unpack routines in `stage` are its own:
/// the codes are adjacent-pair nibbles rather than ggml's split map, the 6-bit
/// fields are positional rather than ggml's `q4k_scale_min_bytes` scheme, and
/// the super values are `bfloat16` rather than `binary16`.
///
/// `k_multiple` is 256 because a super-block is indexed by the GLOBAL
/// flattened tile number: a feature row starts on a super-block boundary only
/// when its tile count is a multiple of four, which is `K % 256 == 0`. Under
/// that gate every plane offset is a per-row stride derivable from `N` and the
/// block count, which is what lets `stage` address a row directly. Any other K
/// falls back to the global-tile-addressed `tcf` GEMV/GEMM kernels. `Q4_K`
/// takes the same multiple for the same reason.
///
/// `prefers_tile_parallel` is a measured flag and this format has not been
/// measured — see its doc. Re-measure: run the kernel-comparison example with
/// `--stream-k` at small `m` and flip this if the tile-parallel run wins
/// outside noise.
pub(in crate::quant::cuda::quant_matmul) const TCF_Q4AS32DT64: FeatMajorFormat = FeatMajorFormat {
    kernel_infix: "tcf_q4as32dt64",
    x_stride: 84,
    k_multiple: 256,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel: false,
};

/// The feature-major descriptor for a TCF native encoding, or `None` when that
/// encoding has no tensor-core kernel compiled and keeps the `tcf` GEMV/GEMM
/// pair instead.
///
/// One place decides this, so an encoding that gains a kernel reaches the
/// dispatch by being listed here rather than by another equality test in the
/// caller.
pub(in crate::quant::cuda::quant_matmul) fn feat_major_format(
    native: NativeEncoding,
) -> Option<&'static FeatMajorFormat> {
    match native {
        NativeEncoding::Q8S32T64 => Some(&TCF_Q8S32T64),
        NativeEncoding::Q4AS32DT64 => Some(&TCF_Q4AS32DT64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::dispatch::{VARIANTS, smem_bytes};
    use super::super::kquant::Q4_K;
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

    /// The `Q4AS32DT64` encoding stages into the Q4_K row, so the two strides
    /// must stay equal and with them the family's shared-memory request at
    /// every token tile. Its super-block straddles rows unless K is a whole
    /// 256, so it takes Q4_K's K multiple too.
    #[test]
    fn the_tcf_q4as32dt64_descriptor_names_the_compiled_symbols() {
        assert_eq!(
            format!("quant_mmq_{}_q8_1_mma_x{}", TCF_Q4AS32DT64.kernel_infix, 8),
            "quant_mmq_tcf_q4as32dt64_q8_1_mma_x8"
        );
        assert_eq!(
            format!(
                "quant_mmq_{}_q8_1_mma_sk_x{}",
                TCF_Q4AS32DT64.kernel_infix, 128
            ),
            "quant_mmq_tcf_q4as32dt64_q8_1_mma_sk_x128"
        );
        assert_eq!(
            format!(
                "quant_mmq_{}_q8_1_mma_fixup_x{}",
                TCF_Q4AS32DT64.kernel_infix, 128
            ),
            "quant_mmq_tcf_q4as32dt64_q8_1_mma_fixup_x128"
        );
        assert_eq!(TCF_Q4AS32DT64.k_multiple, 256);
        assert_eq!(TCF_Q4AS32DT64.x_stride, Q4_K.x_stride);
        // The staged row's bank rule, asserted in `mmqf_body` as well: any
        // multiple of 8 puts consecutive rows in one 128-bit segment.
        assert_eq!(TCF_Q4AS32DT64.x_stride % 8, 4);
        assert!(
            VARIANTS
                .iter()
                .all(|&x| smem_bytes(&TCF_Q4AS32DT64, x) == smem_bytes(&Q4_K, x))
        );
        // Group width 32 matches the activation record's sub-block, as for
        // Q4_K, so no per-token scratch.
        const { assert!(TCF_Q4AS32DT64.act_scratch_ints_per_token == 0) };
        // Not measured against stream-k yet.
        const { assert!(!TCF_Q4AS32DT64.prefers_tile_parallel) };
    }

    /// Only the two encodings with compiled kernels map to a descriptor. Every
    /// other one keeps the `tcf` GEMV/GEMM pair, so it must map to `None`
    /// rather than to a neighbour's row layout.
    #[test]
    fn only_the_compiled_encodings_have_a_feature_major_descriptor() {
        assert_eq!(
            feat_major_format(NativeEncoding::Q8S32T64).map(|f| f.kernel_infix),
            Some("tcf_q8s32t64")
        );
        assert_eq!(
            feat_major_format(NativeEncoding::Q4AS32DT64).map(|f| f.kernel_infix),
            Some("tcf_q4as32dt64")
        );
        for native in [
            NativeEncoding::Q4S32T64,
            NativeEncoding::Q4AS32T64,
            NativeEncoding::Q4AS64T64,
            NativeEncoding::Q6S32T64,
            NativeEncoding::Q6S16DT64,
        ] {
            assert!(feat_major_format(native).is_none(), "{native:?}");
        }
    }
}
