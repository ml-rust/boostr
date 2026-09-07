//! The per-format descriptor the feature-major dispatch reads.

/// One weight format's share of the feature-major family. Everything else in
/// this module — variant choice, stream-k decision, launch, fixup — is shared.
pub(in crate::quant::cuda::quant_matmul) struct FeatMajorFormat {
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
    /// `true` when tile-parallel outruns stream-k at a geometry where
    /// `dispatch.rs` picks stream-k.
    ///
    /// Both grids hold the same blocks per SM, so residency does not separate
    /// them. The split does: stream-k divides every tile across blocks and
    /// pays a fixup pass to rejoin the partials. The finer the split, the more
    /// that pass costs relative to the wave it saves. A format flagged here
    /// gains too little from the split to cover the pass once the
    /// tile-parallel grid nearly fills the device.
    ///
    /// Measured, not derived. No decode property or device capability picks
    /// out which formats those are, so measure the two kernels rather than
    /// infer from the decode shape. Taken on one GPU architecture, so it can
    /// differ on others.
    ///
    /// At small `m`, `token_tiles` is 1 for every `mmq_x`. Variant selection
    /// cannot change this trade-off.
    ///
    /// Re-measure: run the kernel-comparison example with `--stream-k` at
    /// small `m`, compare both kernels per format, flip any format whose
    /// tile-parallel run wins outside noise.
    pub prefers_tile_parallel: bool,
}
