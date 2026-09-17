//! The windowed decode itself: [`decode_latent_windowed`] for a whole
//! latent and [`decode_latent_windowed_from`] for a suffix of one.
//!
//! Both walk the same window grid, anchored at absolute latent frame 0:
//! window `k` covers frames `[k * WINDOW_FRAMES, (k + 1) * WINDOW_FRAMES)`
//! and is decoded with [`CONTEXT_FRAMES`] of real left context. Anchoring
//! the grid at the utterance start rather than at the caller's slice means
//! a suffix decode issues, for every complete window, the exact same
//! `AudioVaeDecoder::forward` call the whole-utterance decode issues — same
//! input slice, same shape — so their kept samples agree bit for bit. Only a
//! window that is still partial at the time of a suffix decode differs, and
//! it differs by float reassociation at most (see the module doc of
//! [`super::context`] for why a shorter input cannot change the receptive
//! field of any kept sample).

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::client::VoxCpmClient;
use crate::model::audio::voxcpm::model::chunked_decode::context::{CONTEXT_FRAMES, WINDOW_FRAMES};
use crate::model::audio::voxcpm::vae::decoder::{AudioVaeDecoder, HOP_LENGTH};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Decode a latent `[1, feat_dim, T]` to a waveform `[1, 1, T * HOP_LENGTH]`,
/// windowing the decode so peak activation memory stays bounded by
/// `WINDOW_FRAMES + CONTEXT_FRAMES` latent frames rather than the full
/// utterance `T`.
///
/// For `T <= WINDOW_FRAMES` this issues exactly one `vae_decoder.forward`
/// call over the whole latent — identical to the non-windowed path. For
/// longer `T`, every window after the first is widened by
/// [`CONTEXT_FRAMES`] real latent frames of left context, and the
/// corresponding `CONTEXT_FRAMES * HOP_LENGTH` samples of decoded output are
/// trimmed before concatenation, so the result matches a single
/// whole-utterance `forward` (see [`super::context`] for why: no
/// global-over-time op exists in the decoder to break that equivalence).
pub(crate) fn decode_latent_windowed<R, C>(
    client: &C,
    vae_decoder: &AudioVaeDecoder<R>,
    latent: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: VoxCpmClient<R>,
    R::Client: numr::ops::ShapeOps<R>,
{
    decode_latent_windowed_from(client, vae_decoder, latent, 0, 0)
}

/// Decode the suffix of an utterance's latent from absolute frame `from`.
///
/// `latent` is `[1, feat_dim, T]` and holds absolute frames `[origin, origin
/// + T)` of the utterance; the caller has already dropped everything before
/// `origin`. Returns `[1, 1, (origin + T - from) * HOP_LENGTH]`: the decoded
/// waveform for frames `[from, origin + T)` only.
///
/// Windows follow the grid anchored at absolute frame 0 (see the module
/// doc), starting at the window that contains `from`. `origin` must be at or
/// before that window's context start, `window_start - CONTEXT_FRAMES`, or
/// the first window would decode with less left context than the
/// whole-utterance path — an error, never a silently different waveform.
/// `from` must lie inside `[origin, origin + T)`.
pub(crate) fn decode_latent_windowed_from<R, C>(
    client: &C,
    vae_decoder: &AudioVaeDecoder<R>,
    latent: &Tensor<R>,
    origin: usize,
    from: usize,
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
    let end = origin + total_frames;
    if from < origin || from >= end {
        return Err(Error::InvalidArgument {
            arg: "from",
            reason: format!(
                "frame {from} is outside the supplied latent, which holds absolute frames \
                 [{origin}, {end})"
            ),
        });
    }

    let mut windows: Vec<Tensor<R>> = Vec::with_capacity((end - from).div_ceil(WINDOW_FRAMES) + 1);
    let mut window_start = (from / WINDOW_FRAMES) * WINDOW_FRAMES;
    while window_start < end {
        let context_start = window_start.saturating_sub(CONTEXT_FRAMES);
        if context_start < origin {
            return Err(Error::InvalidArgument {
                arg: "origin",
                reason: format!(
                    "the window starting at frame {window_start} needs left context from \
                     frame {context_start}, but the supplied latent starts at {origin}"
                ),
            });
        }
        let window_end = (window_start + WINDOW_FRAMES).min(end);

        let slice = latent
            .narrow(2, context_start - origin, window_end - context_start)?
            .contiguous()?;
        let decoded = vae_decoder.forward(client, &slice)?;

        // Drop the context, and on the first window also everything before
        // `from` when `from` sits mid-window.
        let trim_frames = window_start.max(from) - context_start;
        let kept = if trim_frames == 0 {
            // Only when context_start == window_start == from == 0, which
            // mirrors the whole-utterance path's own zero-left-pad at the
            // true start.
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
                             {trim_samples}-sample trim ({trim_frames} frames * \
                             HOP_LENGTH); CONTEXT_FRAMES or WINDOW_FRAMES is misconfigured"
                        ),
                    })?;
            decoded.narrow(2, trim_samples, keep_len)?.contiguous()?
        };
        windows.push(kept);

        window_start = window_end;
    }

    let refs: Vec<&Tensor<R>> = windows.iter().collect();
    Tensor::cat(&refs, 2).map_err(Error::Numr)
}
#[cfg(test)]
mod tests {
    use super::super::test_decoder::{build_decoder, latent};
    use super::*;
    use crate::test_utils::cpu_setup;

    /// The critical property: windowed decode across MULTIPLE windows must
    /// reproduce the whole-utterance decode to within float reassociation.
    /// `3 * WINDOW_FRAMES + 5` guarantees at least 3 full windows plus a
    /// partial final one.
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

        // A window runs the same layers over the same values in the same
        // order as the whole-utterance pass; the only difference is where the
        // buffer starts, and every kept sample's receptive field lies wholly
        // inside the window once CONTEXT_FRAMES >= DERIVED_MIN_CONTEXT_FRAMES.
        // So the two agree to within float reassociation, and the bound is an
        // ULP-scale pin rather than a correctness tolerance.
        //
        // The bound must stay far below the failure this test exists to
        // catch. Too little left context does NOT produce a small error: it
        // produces a wrong boundary region, which is orders of magnitude
        // larger than any reassociation. A bound wide enough to absorb that
        // would defeat the test.
        //
        // MEASURED: the max difference sits near one ULP of the near-silent
        // samples where it occurs, deterministic across repeated runs. It is
        // a reassociation in the layers underneath, not a boundary defect.
        // Widen this only against a re-measured value, never to make a red
        // suite green.
        const MAX_REASSOCIATION: f32 = 1e-9;

        let mut max_diff = 0.0f32;
        for (w, g) in want.iter().zip(got.iter()) {
            max_diff = max_diff.max((w - g).abs());
        }
        assert!(
            max_diff <= MAX_REASSOCIATION,
            "windowed decode diverged from whole-utterance decode by {max_diff}, \
             past the {MAX_REASSOCIATION} reassociation bound; \
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

    /// Decode `[from, available)` the way a streaming caller does: hand over
    /// only the frames from the containing window's context start onward.
    fn suffix(
        client: &numr::runtime::cpu::CpuClient,
        decoder: &AudioVaeDecoder<numr::runtime::cpu::CpuRuntime>,
        x: &Tensor<numr::runtime::cpu::CpuRuntime>,
        from: usize,
        available: usize,
    ) -> Vec<f32> {
        let origin = ((from / WINDOW_FRAMES) * WINDOW_FRAMES).saturating_sub(CONTEXT_FRAMES);
        let part = x.narrow(2, origin, available - origin).expect("narrow");
        decode_latent_windowed_from(client, decoder, &part, origin, from)
            .expect("suffix decode")
            .contiguous()
            .expect("contig")
            .to_vec()
    }

    /// Suffix decodes cut at window boundaries issue the very same decoder
    /// calls as the whole decode, so their concatenation is bit-identical.
    #[test]
    fn window_aligned_suffix_decodes_concatenate_to_the_whole_decode_exactly() {
        let (client, device) = cpu_setup();
        let decoder = build_decoder(&device);
        let frames = 3 * WINDOW_FRAMES + 5;
        let x = latent(frames, &device);
        let want: Vec<f32> = decode_latent_windowed(&client, &decoder, &x)
            .expect("whole")
            .contiguous()
            .expect("contig")
            .to_vec();

        let mut got = Vec::with_capacity(want.len());
        let mut from = 0;
        while from < frames {
            let available = (from + WINDOW_FRAMES).min(frames);
            got.extend(suffix(&client, &decoder, &x, from, available));
            from = available;
        }
        assert_eq!(got.len(), want.len());
        assert!(
            got.iter()
                .zip(&want)
                .all(|(g, w)| g.to_bits() == w.to_bits()),
            "window-aligned suffix decodes must be bit-identical to the whole decode"
        );
    }

    /// Suffix decodes cut mid-window (the streaming chunk size is a few
    /// patches, not a window) still reproduce the whole decode to within
    /// float reassociation, and cover every sample exactly once.
    #[test]
    fn mid_window_suffix_decodes_match_the_whole_decode() {
        let (client, device) = cpu_setup();
        let decoder = build_decoder(&device);
        let frames = 2 * WINDOW_FRAMES + 9;
        let x = latent(frames, &device);
        let want: Vec<f32> = decode_latent_windowed(&client, &decoder, &x)
            .expect("whole")
            .contiguous()
            .expect("contig")
            .to_vec();

        let step = 12;
        let mut got = Vec::with_capacity(want.len());
        let mut from = 0;
        while from < frames {
            let available = (from + step).min(frames);
            got.extend(suffix(&client, &decoder, &x, from, available));
            from = available;
        }
        assert_eq!(got.len(), want.len());
        let max_diff = got
            .iter()
            .zip(&want)
            .fold(0.0f32, |m, (g, w)| m.max((g - w).abs()));
        assert!(max_diff <= 1e-9, "suffix decodes diverged by {max_diff}");
    }

    #[test]
    fn suffix_decode_rejects_a_from_outside_the_latent_and_a_late_origin() {
        let (client, device) = cpu_setup();
        let decoder = build_decoder(&device);
        let x = latent(WINDOW_FRAMES + 8, &device);
        let part = x.narrow(2, 8, WINDOW_FRAMES).expect("narrow");

        // `from` before `origin`.
        assert!(decode_latent_windowed_from(&client, &decoder, &part, 8, 4).is_err());
        // `from` past the end.
        assert!(
            decode_latent_windowed_from(&client, &decoder, &part, 8, WINDOW_FRAMES + 8).is_err()
        );
        // Window 1 needs context from frame WINDOW_FRAMES - CONTEXT_FRAMES,
        // but the latent starts later than that when origin is 40.
        let late = x.narrow(2, 40, WINDOW_FRAMES - 32).expect("narrow");
        let err = decode_latent_windowed_from(&client, &decoder, &late, 40, WINDOW_FRAMES)
            .expect_err("origin past the context start must error");
        assert!(err.to_string().contains("left context"), "{err}");
    }
}
