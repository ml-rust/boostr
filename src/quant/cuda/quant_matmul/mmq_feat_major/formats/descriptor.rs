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
    /// The rule `tiles < 2 * sms` assumes the wave stream-k saves is what
    /// binds. For most formats it is. For a few it is not: their stream-k
    /// kernel stalls on global-memory dependencies and saturates neither
    /// compute nor bandwidth, so saving a wave buys nothing. The fixup pass
    /// is not the cost.
    ///
    /// A decode that chains a dependent table lookup shows this most
    /// clearly. Stream-k runs one block per SM, too few warps to cover the
    /// second load; the tile-parallel grid packs more blocks per SM and
    /// covers it. That does not account for every format flagged here, so
    /// measure the two kernels rather than infer from the decode shape.
    ///
    /// Measured, not derived. No device capability predicts it. Taken on one
    /// GPU architecture, so it can differ on others.
    ///
    /// At small `m`, `token_tiles` is 1 for every `mmq_x`. Variant selection
    /// cannot change this trade-off.
    ///
    /// Re-measure: run the kernel-comparison example with `--stream-k` at
    /// small `m`, compare both kernels per format, flip any format whose
    /// tile-parallel run wins outside noise.
    pub prefers_tile_parallel: bool,
}
