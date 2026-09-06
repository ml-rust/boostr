mod kquant;
mod legacy;

pub(in crate::quant::cuda::quant_matmul) use kquant::{IQ4_XS, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K};
pub(in crate::quant::cuda::quant_matmul) use legacy::{IQ4_NL, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0};

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
}
