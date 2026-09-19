//! The per-format descriptor the feature-major dispatch reads.

use crate::quant::QuantFormat;

/// One weight format's share of the feature-major family. Everything else in
/// this module — variant choice, split count, launch, fixup — is shared.
pub(in crate::quant::cuda::quant_matmul) struct FeatMajorFormat {
    /// The weight format this descriptor serves. Sizes the format's packed
    /// bytes (`QuantFormat::storage_bytes`) where the dispatch needs a
    /// weight of its own, as the tile-parallel probe does.
    pub quant_format: QuantFormat,
    /// Format name inside the kernel symbol, as `MMQ_FM_KERNEL`'s `NAME`.
    pub kernel_infix: &'static str,
    /// Weight row stride in the shared tile, in ints (`FMT::X_STRIDE`).
    pub x_stride: u32,
    /// K must be a whole number of these for the staging map to hold.
    pub k_multiple: u32,
    /// Per-token ints of shared scratch the format's `vec_dot` reads, held in
    /// its own region after the activation tile (`FMT::Y_SCRATCH`).
    ///
    /// Zero for every format whose minimum term is no finer than the shared
    /// activation record's 32-value sub-block, which is all of them but Q2_K.
    /// Q2_K's minimum changes every 16 elements, so the kernel derives the
    /// per-16 split of each stored sum into this region once per staged
    /// activation tile.
    pub act_scratch_ints_per_token: u32,
    /// `true` when the fused tile-parallel grid outruns the split-K pair at
    /// a geometry where `dispatch.rs` picks the pair. Both form the same
    /// bits; this is a schedule choice.
    ///
    /// Both grids hold the same blocks per SM, so residency does not separate
    /// them. The pair divides every tile across blocks and pays partial
    /// stores plus a fixup pass to rejoin them. The finer the split, the more
    /// that pass costs relative to the wave it saves. A format flagged here
    /// gains too little from the split to cover the pass once the
    /// tile-parallel grid nearly fills the device.
    ///
    /// Measured, not derived. No decode property or device capability picks
    /// out which formats those are, so measure the two kernels rather than
    /// infer from the decode shape.
    ///
    /// This constant is the FALLBACK, measured on one Ampere-class part. The
    /// dispatch reads `tiling::prefers_tile_parallel`, which times both
    /// schedules on the device at first use and caches the pick per (device,
    /// format) under [`Self::tile_parallel_key`]; it returns this constant
    /// when tuning is off (`NUMR_CUDA_TUNE=0`) or the probe fails. No
    /// dispatch code reads this field directly.
    ///
    /// At small `m`, `token_tiles` is 1 for every `mmq_x`. Variant selection
    /// cannot change this trade-off.
    ///
    /// Re-measure: run the kernel-comparison example with `--split-k` at
    /// small `m`, compare both kernels per format, flip any format whose
    /// tile-parallel run wins outside noise.
    pub prefers_tile_parallel_fallback: bool,
    /// Tune-cache key of the measured `prefers_tile_parallel` pick:
    /// `mmq_feat_major.<kernel_infix>.prefers_tile_parallel`. One per
    /// format, so two formats never share a probe result.
    pub tile_parallel_key: &'static str,
    /// `true` when the kernel file compiles this format's `_y64_` and `_y16_`
    /// entry points: the narrow feature tile for the CTA-starved small-M,
    /// small-N regime and the single-warp tile for the decode regime — see
    /// `MMQ_FM_KERNEL_Y64` and `MMQ_FM_KERNEL_Y16` in `quant_mmq_mma.cu`.
    ///
    /// Only the formats a K-quant mix places on the small-N projections
    /// compile them; every other format has one feature tile, and the
    /// dispatch never asks it for another. A format joins by adding its
    /// `MMQ_FM_KERNEL_Y64` and `MMQ_FM_KERNEL_Y16` lists to the kernel file
    /// AND flipping this flag; the flag alone would launch a symbol the
    /// module does not hold.
    pub narrow_tile: bool,
}
