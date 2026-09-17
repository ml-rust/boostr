//! Left-context and window sizes for the windowed VAE decode.
//!
//! `AudioVaeDecoder::forward` has no global-over-time op anywhere in its
//! path (verified: every op is elementwise or a local/causal conv — see
//! `vae/decoder.rs`, `vae/causal_conv1d.rs`, `vae/causal_transpose_conv1d.rs`,
//! `vae/res_unit.rs`). That makes exact chunking possible: decode the latent
//! in fixed windows with enough REAL left context that each window's kept
//! output is what the whole-utterance decode would produce, without ever
//! holding the full-utterance activation tensors in memory at once. No
//! per-layer streaming state is used — every window is a fresh, stateless
//! `forward` call; the repeated context recompute is deliberate.
//!
//! # Deriving the required left context (in LATENT frames)
//!
//! Two causal-conv effects need real (not zero-padded) left history to
//! reproduce the whole-utterance result at a chunk boundary:
//!
//! 1. Each `ResUnit`'s dilated `CausalConv1d` (`kernel=7`, dilations 1, 3, 9)
//!    zero-pads its own left edge (`causal_conv1d.rs`). Fed only real frames
//!    from the window start onward, its output only matches the
//!    whole-utterance output from `dilation*(kernel-1)` samples in. Three
//!    `ResUnit`s stack additively (standard result for stacked causal convs,
//!    same as a WaveNet receptive field): `6*(1+3+9) = 78`
//!    (`RES_UNIT_CONTEXT`), in that block's own OUTPUT-rate (i.e.
//!    post-upsample) time resolution.
//! 2. Each `DecoderBlock`'s `CausalTransposeConv1d` (stride `s`, kernel `2s`)
//!    is a `Valid`-padded (not zero-padded) transposed conv, tail-trimmed by
//!    `s`. Tracing its gather kernel
//!    (`l = (ot - k) / stride`, requiring `l >= 0` and `k < 2*stride`) shows
//!    that, fed only real frames from the window start onward, output index
//!    `ot` matches the whole-utterance result only for `ot >= stride` — so
//!    the upsample itself needs `stride` extra OUTPUT samples of margin.
//!
//! So block `i`'s own context requirement, in block `i`'s OUTPUT-rate units,
//! is `STRIDES[i] + RES_UNIT_CONTEXT`. Converting to LATENT frames divides by
//! the cumulative upsample factor through block `i` (`STRIDES[0..=i]`'s
//! product); summing across all 6 blocks and adding `front_dw`'s own
//! `CAUSAL_KERNEL - 1 = 6` latent-rate frames (it runs directly on the
//! latent, so no rate conversion), then rounding up, gives the minimum:
//!
//! ```text
//! block i (0-indexed, STRIDES = [8,6,5,2,2,2], HOP_LENGTH = 1920):
//!   cumulative_i = product(STRIDES[0..=i])   rest_i = HOP_LENGTH / cumulative_i
//!   context_i    = STRIDES[i] + 78
//!
//!   i  STRIDES[i]  context_i  cumulative_i  rest_i  context_i/cumulative_i
//!   0      8           86          8          240          10.750
//!   1      6           84         48           40           1.750
//!   2      5           83        240            8           0.346
//!   3      2           80        480            4           0.167
//!   4      2           80        960            2           0.083
//!   5      2           80       1920            1           0.042
//!                                              sum =        13.137
//!
//!   DERIVED_MIN_CONTEXT_FRAMES = ceil(13.137 + 6) = ceil(19.137) = 20
//! ```
//!
//! Implemented below with exact integer arithmetic (`context_i * rest_i`
//! summed over a common denominator of `HOP_LENGTH`, since every `rest_i`
//! divides `HOP_LENGTH` exactly — each is a suffix product of `STRIDES`),
//! not floating point, so the compile-time value is exact rather than an
//! approximation of the table above.
//!
//! [`CONTEXT_FRAMES`] rounds [`DERIVED_MIN_CONTEXT_FRAMES`] up to a safety
//! margin; a `const` assertion below fails the build if it is ever set below
//! the derived minimum.

use crate::model::audio::voxcpm::vae::decoder::{
    CAUSAL_KERNEL, HOP_LENGTH, RES_UNIT_DILATIONS, STRIDES,
};

/// Per-`DecoderBlock` context from its three stacked dilated `ResUnit`s,
/// `6*(1+3+9) = 78`, in that block's own OUTPUT-rate units. See the module
/// doc for the derivation.
const RES_UNIT_CONTEXT: usize =
    (CAUSAL_KERNEL - 1) * (RES_UNIT_DILATIONS[0] + RES_UNIT_DILATIONS[1] + RES_UNIT_DILATIONS[2]);

/// Exact minimum left context, in LATENT frames, for a windowed decode to
/// reproduce the whole-utterance `AudioVaeDecoder::forward`. See the module
/// doc for the derivation; this computes the same quantity with exact
/// integer arithmetic rather than the illustrative float table there.
const fn derive_min_context_frames() -> usize {
    const N: usize = STRIDES.len();
    // rest[i] = product(STRIDES[i+1..N]) = HOP_LENGTH / cumulative_i, the
    // suffix product of strides applied AFTER block i. Always divides
    // HOP_LENGTH exactly, since HOP_LENGTH is the full product of STRIDES.
    let mut rest = [1u64; N];
    let mut i = N;
    while i > 1 {
        i -= 1;
        rest[i - 1] = rest[i] * STRIDES[i] as u64;
    }

    // numerator / HOP_LENGTH == sum_i(context_i / cumulative_i), computed
    // over the common denominator HOP_LENGTH so no floating point is needed.
    let mut numerator: u64 = 0;
    let mut j = 0;
    while j < N {
        let context_i = STRIDES[j] as u64 + RES_UNIT_CONTEXT as u64;
        numerator += context_i * rest[j];
        j += 1;
    }

    let hop = HOP_LENGTH as u64;
    let block_frames = numerator.div_ceil(hop);
    let front_dw_frames = (CAUSAL_KERNEL - 1) as u64; // front_dw: kernel 7, dilation 1
    (block_frames + front_dw_frames) as usize
}

/// Exact minimum left context in latent frames (= 20 for this decoder's
/// fixed `STRIDES`/`CAUSAL_KERNEL`/`RES_UNIT_DILATIONS`). See the module doc.
pub(crate) const DERIVED_MIN_CONTEXT_FRAMES: usize = derive_min_context_frames();

/// Left context actually used by the windowed decode, in latent frames.
/// Rounded up from [`DERIVED_MIN_CONTEXT_FRAMES`] (20) to 32 as a safety
/// margin against the derivation above being off by a frame or two on a
/// decoder variant. The cost is 12 extra latent frames of recompute per
/// window.
pub(crate) const CONTEXT_FRAMES: usize = 32;

const _: () = assert!(
    CONTEXT_FRAMES >= DERIVED_MIN_CONTEXT_FRAMES,
    "CONTEXT_FRAMES must be at least the derived minimum left context or \
     windowed decode will not match the whole-utterance decode"
);

/// Latent frames decoded per window (excluding [`CONTEXT_FRAMES`] of left
/// context). Peak activation memory during one window's
/// `AudioVaeDecoder::forward` scales with `WINDOW_FRAMES + CONTEXT_FRAMES`,
/// not `WINDOW_FRAMES` alone — bounding it keeps peak memory roughly constant
/// regardless of total utterance length, at the cost of recomputing
/// `CONTEXT_FRAMES` latent frames' worth of activations on every window
/// after the first.
pub(crate) const WINDOW_FRAMES: usize = 64;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_minimum_matches_the_documented_table() {
        assert_eq!(DERIVED_MIN_CONTEXT_FRAMES, 20);
    }

    #[test]
    fn context_is_a_whole_number_of_default_patches() {
        // `decode_patches_from` unfolds whole patches; a context that is a
        // multiple of the default patch size (4) never over-fetches.
        assert_eq!(CONTEXT_FRAMES % 4, 0);
        assert_eq!(WINDOW_FRAMES % 4, 0);
    }
}
