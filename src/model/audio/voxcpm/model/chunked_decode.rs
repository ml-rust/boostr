//! Overlap-and-trim windowed decode for [`super::loader::VoxCpm2Model::decode_patches`].
//!
//! `AudioVaeDecoder::forward` has no global-over-time op anywhere in its
//! path (verified: every op is elementwise or a local/causal conv — see
//! `vae/decoder.rs`, `vae/causal_conv1d.rs`, `vae/causal_transpose_conv1d.rs`,
//! `vae/res_unit.rs`). That makes exact chunking possible: decode the latent
//! in fixed windows with enough REAL left context that each window's kept
//! output is bit-for-bit what the whole-utterance decode would produce,
//! without ever holding the full-utterance activation tensors in memory at
//! once. No per-layer streaming state is used — every window is a fresh,
//! stateless `forward` call; the repeated context recompute is deliberate.
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

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::client::VoxCpmClient;
use crate::model::audio::voxcpm::vae::decoder::{
    AudioVaeDecoder, CAUSAL_KERNEL, HOP_LENGTH, RES_UNIT_DILATIONS, STRIDES,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Per-`DecoderBlock` context from its three stacked dilated `ResUnit`s,
/// `6*(1+3+9) = 78`, in that block's own OUTPUT-rate units. See the module
/// doc for the derivation.
const RES_UNIT_CONTEXT: usize =
    (CAUSAL_KERNEL - 1) * (RES_UNIT_DILATIONS[0] + RES_UNIT_DILATIONS[1] + RES_UNIT_DILATIONS[2]);

/// Exact minimum left context, in LATENT frames, for a windowed decode to
/// reproduce the whole-utterance `AudioVaeDecoder::forward` bit-for-bit. See
/// the module doc for the derivation; this computes the same quantity with
/// exact integer arithmetic rather than the illustrative float table there.
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

/// Left context actually used by [`decode_latent_windowed`], in latent
/// frames. Rounded up from [`DERIVED_MIN_CONTEXT_FRAMES`] (20) to 32 as a
/// safety margin against the derivation above being off by a frame or two on
/// a decoder variant, at negligible cost: 12 extra latent frames of recompute
/// per window is ~23 ms of audio, far below the ~1.5% wall-clock overhead
/// budget for this whole scheme.
pub(crate) const CONTEXT_FRAMES: usize = 32;

const _: () = assert!(
    CONTEXT_FRAMES >= DERIVED_MIN_CONTEXT_FRAMES,
    "CONTEXT_FRAMES must be at least the derived minimum left context or \
     windowed decode will not match the whole-utterance decode"
);

/// Latent frames decoded per window (excluding [`CONTEXT_FRAMES`] of left
/// context). ~64 frames is ~2.56 s of audio per window. Peak activation
/// memory during one window's `AudioVaeDecoder::forward` scales with
/// `WINDOW_FRAMES + CONTEXT_FRAMES` (96 frames here), not `WINDOW_FRAMES`
/// alone — bounding it near 64 keeps peak memory roughly constant regardless
/// of total utterance length, at the cost of recomputing `CONTEXT_FRAMES`
/// latent frames' worth of activations on every window after the first.
pub(crate) const WINDOW_FRAMES: usize = 64;

/// Decode a latent `[1, feat_dim, T]` to a waveform `[1, 1, T * HOP_LENGTH]`,
/// windowing the decode so peak activation memory stays bounded by
/// `WINDOW_FRAMES + CONTEXT_FRAMES` latent frames rather than the full
/// utterance `T`.
///
/// For `T <= WINDOW_FRAMES` this issues exactly one `vae_decoder.forward`
/// call over the whole latent (the loop below's first iteration always has
/// `context_start == 0 == start`, so it trims nothing) — identical to the
/// non-windowed path. For longer `T`, every window after the first is
/// widened by [`CONTEXT_FRAMES`] real latent frames of left context, and the
/// corresponding `CONTEXT_FRAMES * HOP_LENGTH` samples of decoded output are
/// trimmed before concatenation, so the result is bit-for-bit identical to a
/// single whole-utterance `forward` (see this module's doc for why: no
/// global-over-time op exists in the decoder to break that equivalence).
pub(crate) fn decode_latent_windowed<R, C>(
    client: &C,
    vae_decoder: &AudioVaeDecoder<R>,
    latent: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: VoxCpmClient<R>,
    // `Tensor::cat` joins the per-window outputs, and it is the runtime's own
    // client that performs the concatenation, not the `client` argument.
    R::Client: numr::ops::ShapeOps<R>,
{
    let shape = latent.shape();
    if shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "latent",
            reason: format!("expected a 3D [B, feat_dim, T] latent, got {shape:?}"),
        });
    }
    let total_frames = shape[2];
    if total_frames == 0 {
        return Err(Error::InvalidArgument {
            arg: "latent",
            reason: "expected at least 1 latent frame, got 0".to_string(),
        });
    }

    let mut windows: Vec<Tensor<R>> = Vec::with_capacity(total_frames.div_ceil(WINDOW_FRAMES));
    let mut start = 0usize;
    while start < total_frames {
        let context_start = start.saturating_sub(CONTEXT_FRAMES);
        let window_end = (start + WINDOW_FRAMES).min(total_frames);
        let slice_len = window_end - context_start;

        let slice = latent.narrow(2, context_start, slice_len)?.contiguous()?;
        let decoded = vae_decoder.forward(client, &slice)?;

        let trim_frames = start - context_start;
        let kept = if trim_frames == 0 {
            // First window only: context_start == start == 0, mirroring the
            // whole-utterance path's own zero-left-pad at the true start.
            decoded
        } else {
            let trim_samples = trim_frames * HOP_LENGTH;
            let decoded_len = decoded.shape()[2];
            let keep_len =
                decoded_len
                    .checked_sub(trim_samples)
                    .ok_or_else(|| Error::InvalidArgument {
                        arg: "latent",
                        reason: format!(
                            "decoded window length {decoded_len} is shorter than the \
                             {trim_samples}-sample context trim ({trim_frames} frames * \
                             HOP_LENGTH); CONTEXT_FRAMES or WINDOW_FRAMES is misconfigured"
                        ),
                    })?;
            decoded.narrow(2, trim_samples, keep_len)?.contiguous()?
        };
        windows.push(kept);

        start = window_end;
    }

    let refs: Vec<&Tensor<R>> = windows.iter().collect();
    Tensor::cat(&refs, 2).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::voxcpm::vae::causal_conv1d::CausalConv1d;
    use crate::model::audio::voxcpm::vae::causal_transpose_conv1d::CausalTransposeConv1d;
    use crate::model::audio::voxcpm::vae::decoder::{
        AudioVaeDecoderWeights, FINAL_CHANNELS, FRONT_HIDDEN, INPUT_CHANNELS,
    };
    use crate::model::audio::voxcpm::vae::decoder_block::{DecoderBlock, DecoderBlockWeights};
    use crate::model::audio::voxcpm::vae::res_unit::ResUnit;
    use crate::model::audio::voxcpm::vae::snake::Snake;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn snake(c: usize, device: &<CpuRuntime as Runtime>::Device) -> Snake<CpuRuntime> {
        let alpha =
            Tensor::<CpuRuntime>::from_slice(&vec![0.2f32; c], &[1, c, 1], device).expect("alpha");
        Snake::new(alpha).expect("snake")
    }

    fn depthwise(
        c: usize,
        k: usize,
        dilation: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> CausalConv1d<CpuRuntime> {
        let weight = Tensor::<CpuRuntime>::from_slice(
            &(0..c * k)
                .map(|i| 0.01 * ((i % 5) as f32 - 2.0))
                .collect::<Vec<f32>>(),
            &[c, 1, k],
            device,
        )
        .expect("weight");
        let bias = Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c], &[c], device).expect("bias");
        CausalConv1d::new(weight, Some(bias), k, dilation, c).expect("depthwise")
    }

    fn pointwise(
        c_in: usize,
        c_out: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> CausalConv1d<CpuRuntime> {
        let weight = Tensor::<CpuRuntime>::from_slice(
            &(0..c_out * c_in)
                .map(|i| 0.005 * ((i % 7) as f32 - 3.0))
                .collect::<Vec<f32>>(),
            &[c_out, c_in, 1],
            device,
        )
        .expect("weight");
        let bias =
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c_out], &[c_out], device).expect("bias");
        CausalConv1d::new(weight, Some(bias), 1, 1, 1).expect("pointwise")
    }

    fn res_unit(
        c: usize,
        dilation: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> ResUnit<CpuRuntime> {
        ResUnit::new(
            snake(c, device),
            depthwise(c, 7, dilation, device),
            snake(c, device),
            pointwise(c, c, device),
        )
    }

    fn decoder_block(
        input_dim: usize,
        output_dim: usize,
        stride: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> DecoderBlock<CpuRuntime> {
        let k = 2 * stride;
        let up_weight = Tensor::<CpuRuntime>::from_slice(
            &(0..input_dim * output_dim * k)
                .map(|i| 0.01 * ((i % 5) as f32 - 2.0))
                .collect::<Vec<f32>>(),
            &[input_dim, output_dim, k],
            device,
        )
        .expect("up_weight");
        let up_bias =
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; output_dim], &[output_dim], device)
                .expect("up_bias");
        let num_buckets = 4;
        let scale_embed = Tensor::<CpuRuntime>::from_slice(
            &vec![1.0f32; num_buckets * input_dim],
            &[num_buckets, input_dim],
            device,
        )
        .expect("scale_embed");
        let bias_embed = Tensor::<CpuRuntime>::from_slice(
            &vec![0.0f32; num_buckets * input_dim],
            &[num_buckets, input_dim],
            device,
        )
        .expect("bias_embed");
        DecoderBlock::new(DecoderBlockWeights {
            snake: snake(input_dim, device),
            upsample: CausalTransposeConv1d::new(up_weight, Some(up_bias), stride)
                .expect("upsample"),
            res1: res_unit(output_dim, 1, device),
            res3: res_unit(output_dim, 3, device),
            res9: res_unit(output_dim, 9, device),
            scale_embed,
            bias_embed,
        })
        .expect("decoder_block")
    }

    /// A real (small-weight, non-degenerate) `AudioVaeDecoder` at full
    /// architecture scale (`STRIDES`, `CAUSAL_KERNEL`, dilations all match
    /// production), so `CONTEXT_FRAMES` is exercised exactly as derived.
    fn build_decoder(device: &<CpuRuntime as Runtime>::Device) -> AudioVaeDecoder<CpuRuntime> {
        let dims = [
            (FRONT_HIDDEN, FRONT_HIDDEN / 2),
            (FRONT_HIDDEN / 2, FRONT_HIDDEN / 4),
            (FRONT_HIDDEN / 4, FRONT_HIDDEN / 8),
            (FRONT_HIDDEN / 8, FRONT_HIDDEN / 16),
            (FRONT_HIDDEN / 16, FRONT_HIDDEN / 32),
            (FRONT_HIDDEN / 32, FINAL_CHANNELS),
        ];
        let blocks =
            std::array::from_fn(|i| decoder_block(dims[i].0, dims[i].1, STRIDES[i], device));

        AudioVaeDecoder::new(AudioVaeDecoderWeights {
            front_dw: depthwise(INPUT_CHANNELS, 7, 1, device),
            front_pw: pointwise(INPUT_CHANNELS, FRONT_HIDDEN, device),
            blocks,
            final_snake: snake(FINAL_CHANNELS, device),
            final_conv: {
                let weight = Tensor::<CpuRuntime>::from_slice(
                    &(0..FINAL_CHANNELS * 7)
                        .map(|i| 0.001 * ((i % 5) as f32 - 2.0))
                        .collect::<Vec<f32>>(),
                    &[1, FINAL_CHANNELS, 7],
                    device,
                )
                .expect("final_conv weight");
                let bias = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], device)
                    .expect("final_conv bias");
                CausalConv1d::new(weight, Some(bias), 7, 1, 1).expect("final_conv")
            },
        })
    }

    fn latent(frames: usize, device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
        let data: Vec<f32> = (0..INPUT_CHANNELS * frames)
            .map(|i| (i as f32 * 0.013).sin())
            .collect();
        Tensor::<CpuRuntime>::from_slice(&data, &[1, INPUT_CHANNELS, frames], device)
            .expect("latent")
    }

    /// The critical property: windowed decode across MULTIPLE windows must
    /// reproduce the whole-utterance decode bit-for-bit (up to float
    /// reassociation). `3 * WINDOW_FRAMES + 5` guarantees at least 3 full
    /// windows plus a partial final one.
    #[test]
    fn windowed_decode_matches_whole_utterance_across_multiple_windows() {
        let (client, device) = cpu_setup();
        let decoder = build_decoder(&device);
        let frames = 3 * WINDOW_FRAMES + 5;
        let x = latent(frames, &device);

        let whole = decoder
            .forward(&client, &x)
            .expect("whole-utterance forward");
        let windowed = decode_latent_windowed(&client, &decoder, &x).expect("windowed forward");

        assert_eq!(whole.shape(), windowed.shape());
        assert_eq!(whole.shape(), &[1, 1, frames * HOP_LENGTH]);

        let want: Vec<f32> = whole.contiguous().expect("contig").to_vec();
        let got: Vec<f32> = windowed.contiguous().expect("contig").to_vec();
        assert_eq!(want.len(), got.len());

        // MEASURED: the max difference is exactly 0.0 — bit-for-bit, not
        // merely close. There is no float reassociation to absorb, because a
        // window runs the same layers over the same values in the same order
        // as the whole-utterance pass; the only difference is where the
        // buffer starts, and every kept sample's receptive field lies wholly
        // inside the window once CONTEXT_FRAMES >= DERIVED_MIN_CONTEXT_FRAMES.
        //
        // So the assertion is exact equality rather than a tolerance. A
        // tolerance here would silently absorb the failure this test exists
        // to catch: too little left context does NOT produce a small error,
        // it produces a wrong boundary region, and an ULP-scale bound is the
        // honest pin on a property that currently holds exactly. If a future
        // numr kernel legitimately reassociates and this starts failing by
        // one ULP, that is worth a deliberate decision, not a pre-widened
        // bound hiding it.
        let mut max_diff = 0.0f32;
        for (w, g) in want.iter().zip(got.iter()) {
            max_diff = max_diff.max((w - g).abs());
        }
        assert_eq!(
            max_diff, 0.0,
            "windowed decode diverged from whole-utterance decode by {max_diff}; \
             CONTEXT_FRAMES ({CONTEXT_FRAMES}) is likely too small"
        );
    }

    /// Degenerate case: a latent shorter than one window is exactly one
    /// `vae_decoder.forward` call, identical to the non-windowed path.
    #[test]
    fn latent_shorter_than_one_window_matches_whole_utterance_path() {
        let (client, device) = cpu_setup();
        let decoder = build_decoder(&device);
        let frames = WINDOW_FRAMES - 1;
        let x = latent(frames, &device);

        let whole = decoder
            .forward(&client, &x)
            .expect("whole-utterance forward");
        let windowed = decode_latent_windowed(&client, &decoder, &x).expect("windowed forward");

        let want: Vec<f32> = whole.contiguous().expect("contig").to_vec();
        let got: Vec<f32> = windowed.contiguous().expect("contig").to_vec();
        assert_eq!(want, got);
    }

    #[test]
    fn sample_count_matches_frames_times_hop_length_for_exact_multiple() {
        let (client, device) = cpu_setup();
        let decoder = build_decoder(&device);
        let frames = 2 * WINDOW_FRAMES;
        let x = latent(frames, &device);

        let windowed = decode_latent_windowed(&client, &decoder, &x).expect("windowed forward");
        assert_eq!(windowed.shape(), &[1, 1, frames * HOP_LENGTH]);
    }

    #[test]
    fn sample_count_matches_frames_times_hop_length_for_non_multiple() {
        let (client, device) = cpu_setup();
        let decoder = build_decoder(&device);
        let frames = 2 * WINDOW_FRAMES + 17;
        let x = latent(frames, &device);

        let windowed = decode_latent_windowed(&client, &decoder, &x).expect("windowed forward");
        assert_eq!(windowed.shape(), &[1, 1, frames * HOP_LENGTH]);
    }
}
