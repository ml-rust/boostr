//! The four lowbit descriptors — `PQ2_0`, `Q2_0`, `Q1_0`, `PTQ1_0` —
//! all ternary/binary formats staged as Q8_0's row byte for byte.

use super::FeatMajorFormat;
use crate::quant::QuantFormat;

/// PQ2_0: 34-byte blocks of 128 elements, 2-bit codes under ONE f16 scale,
/// staged as Q8_0's row byte for byte — 64 quant words plus 8 f32 scales plus
/// 4 ints of bank padding. The kernel expands each code to the signed int8
/// `code - 1` while staging and writes the block's scale into all four of the
/// 32-element slots the block covers, so the staged row and the whole
/// `vec_dot` are Q8_0's. K needs only a whole 128-element block, so a row's
/// last 256-k staging group can be partial.
///
/// `prefers_tile_parallel_fallback` is measured: `mmq_kernel_compare --format
/// pq2_0 --n 5120 --k 5120 --m 8 --split-k 2` has tile-parallel beat split-K
/// at this geometry, so the grid keeps the flag.
pub(in crate::quant::cuda::quant_matmul) const PQ2_0: FeatMajorFormat = FeatMajorFormat {
    quant_format: QuantFormat::PQ2_0,
    kernel_infix: "pq2_0",
    x_stride: 76,
    k_multiple: 128,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel_fallback: true,
    tile_parallel_key: "mmq_feat_major.pq2_0.prefers_tile_parallel",
    narrow_tile: false,
};

/// Q2_0: 18-byte blocks of 64 elements, 2-bit codes under ONE f16 scale,
/// staged as Q8_0's row byte for byte. PQ2_0's code space at half the run
/// length: the kernel expands the codes the same way and writes the block's
/// scale into both 32-element slots it covers. K needs only a whole
/// 64-element block, so a row's last 256-k staging group can be partial.
///
/// `prefers_tile_parallel_fallback` is measured as the PQ2_0 descriptor says,
/// with `--format q2_0`: tile-parallel beats split-K, so the flag stays set.
pub(in crate::quant::cuda::quant_matmul) const Q2_0: FeatMajorFormat = FeatMajorFormat {
    quant_format: QuantFormat::Q2_0,
    kernel_infix: "q2_0",
    x_stride: 76,
    k_multiple: 64,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel_fallback: true,
    tile_parallel_key: "mmq_feat_major.q2_0.prefers_tile_parallel",
    narrow_tile: false,
};

/// Q1_0: 18-byte blocks of 128 elements, one sign bit per element under ONE
/// f16 scale, staged as Q8_0's row byte for byte. The kernel expands each bit
/// to the signed int8 `+1`/`-1` while staging and writes the block's scale
/// into all four 32-element slots it covers. K needs only a whole 128-element
/// block, so a row's last 256-k staging group can be partial.
///
/// `prefers_tile_parallel_fallback` is measured as the PQ2_0 descriptor says,
/// with `--format q1_0`: tile-parallel beats split-K, so the flag stays set.
pub(in crate::quant::cuda::quant_matmul) const Q1_0: FeatMajorFormat = FeatMajorFormat {
    quant_format: QuantFormat::Q1_0,
    kernel_infix: "q1_0",
    x_stride: 76,
    k_multiple: 128,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel_fallback: true,
    tile_parallel_key: "mmq_feat_major.q1_0.prefers_tile_parallel",
    narrow_tile: false,
};

/// PTQ1_0: 28-byte blocks of 128 elements, base-3 packed trits under ONE f16
/// scale at the END of the block, staged as Q8_0's row byte for byte. The
/// kernel expands each trit to the signed int8 `-1`/`0`/`+1` while staging
/// (`gguf_base3_trit`, one level per 8-element lane group, the `qh` tail
/// apart) and writes the block's scale into all four 32-element slots it
/// covers. K needs only a whole 128-element block, so a row's last 256-k
/// staging group can be partial.
///
/// `prefers_tile_parallel_fallback` is the tuner's pick at the geometry
/// where the flag decides (two token tiles, `n` at the veto threshold). A
/// forced `--split-k 2` at m = 8 (`mmq_kernel_compare --format ptq1_0 --n
/// 5120 --k 5120`) has the split-K pair ahead at that one forced geometry,
/// but the production rule never emits the pair there, so that comparison
/// does not set the flag.
pub(in crate::quant::cuda::quant_matmul) const PTQ1_0: FeatMajorFormat = FeatMajorFormat {
    quant_format: QuantFormat::PTQ1_0,
    kernel_infix: "ptq1_0",
    x_stride: 76,
    k_multiple: 128,
    act_scratch_ints_per_token: 0,
    prefers_tile_parallel_fallback: true,
    tile_parallel_key: "mmq_feat_major.ptq1_0.prefers_tile_parallel",
    narrow_tile: false,
};

#[cfg(test)]
mod tests {
    use super::super::super::tiling::{Cadence, FEAT_TILE_DEFAULT, VARIANTS, smem_bytes};
    use super::super::legacy::Q8_0;
    use super::*;

    /// The four lowbit formats stage into the Q8_0 row, so their strides must
    /// stay equal to it and with them the family's shared-memory request at
    /// every token tile. Their K multiple is one weight block — 128, 64, 128,
    /// 128 — which is finer than a 256-k group, so each takes the ragged tail.
    #[test]
    fn the_lowbit_descriptors_name_the_compiled_symbols() {
        let lowbit: [(&FeatMajorFormat, &str, u32); 4] = [
            (&PQ2_0, "pq2_0", 128),
            (&Q2_0, "q2_0", 64),
            (&Q1_0, "q1_0", 128),
            (&PTQ1_0, "ptq1_0", 128),
        ];
        for (fm, infix, k_multiple) in lowbit {
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
        // The tuner's pick at the deciding geometry: tile-parallel for all
        // four.
        const { assert!(PQ2_0.prefers_tile_parallel_fallback) };
        const { assert!(Q2_0.prefers_tile_parallel_fallback) };
        const { assert!(Q1_0.prefers_tile_parallel_fallback) };
        const { assert!(PTQ1_0.prefers_tile_parallel_fallback) };
    }
}
