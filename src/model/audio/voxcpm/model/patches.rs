//! Waveform padding and the AudioVAE patch fold — the two pure reshaping
//! steps between raw reference audio and `feat_encoder`'s input.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::borrow::Cow;

/// Right-pad `wav` with zeros to a multiple of `multiple`.
///
/// Borrows when `wav` is already a multiple, so the common path copies
/// nothing. `multiple` must be non-zero.
pub fn pad_to_multiple(wav: &[f32], multiple: usize) -> Result<Cow<'_, [f32]>> {
    if multiple == 0 {
        return Err(Error::InvalidArgument {
            arg: "multiple",
            reason: "expected at least 1, got 0".to_string(),
        });
    }
    let remainder = wav.len() % multiple;
    if remainder == 0 {
        return Ok(Cow::Borrowed(wav));
    }
    let mut padded = vec![0.0f32; wav.len() + (multiple - remainder)];
    padded[..wav.len()].copy_from_slice(wav);
    Ok(Cow::Owned(padded))
}

/// Fold an AudioVAE latent `[1, feat_dim, frames]` into per-patch features
/// `[frames / patch_size, patch_size, feat_dim]`.
///
/// This is the reference's `feat.view(64, -1, 4).permute(1, 2, 0)` exactly.
/// EVERY other permutation of those three axes is shape-valid for the square
/// case and scrambles time, so the axis order here is load-bearing:
/// `feat[c][t]` must land at `folded[t / patch_size][t % patch_size][c]`.
///
/// `frames` must already be a multiple of `patch_size` — that is what
/// [`super::config::VoxCpm2Config::ref_pad_multiple`] guarantees upstream.
pub fn fold_patches<R: Runtime<DType = DType>>(
    latent: &Tensor<R>,
    patch_size: usize,
    feat_dim: usize,
) -> Result<Tensor<R>> {
    let shape = latent.shape().to_vec();
    if shape.len() != 3 || shape[0] != 1 || shape[1] != feat_dim {
        return Err(Error::InvalidArgument {
            arg: "latent",
            reason: format!("expected [1, {feat_dim}, frames], got {shape:?}"),
        });
    }
    let frames = shape[2];
    if patch_size == 0 || !frames.is_multiple_of(patch_size) {
        return Err(Error::InvalidArgument {
            arg: "latent",
            reason: format!(
                "expected a frame count divisible by patch_size ({patch_size}), got {frames}"
            ),
        });
    }
    let t_ref = frames / patch_size;
    Ok(latent
        .contiguous()?
        .reshape(&[feat_dim, t_ref, patch_size])?
        .permute(&[1, 2, 0])?
        .contiguous()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn padding_leaves_an_exact_multiple_untouched() {
        let wav = vec![0.5f32; 2560 * 3];
        let padded = pad_to_multiple(&wav, 2560).expect("pad");
        assert_eq!(padded.len(), wav.len());
        assert!(
            matches!(padded, Cow::Borrowed(_)),
            "an exact multiple must not be copied"
        );
    }

    #[test]
    fn padding_extends_to_the_next_multiple_with_zeros() {
        let wav = vec![0.5f32; 2560 + 1];
        let padded = pad_to_multiple(&wav, 2560).expect("pad");
        assert_eq!(padded.len(), 2560 * 2);
        assert_eq!(&padded[..wav.len()], wav.as_slice());
        assert!(
            padded[wav.len()..].iter().all(|&v| v == 0.0),
            "the tail must be zeros"
        );
    }

    /// The VAE's own 640 modulus is NOT the one this path needs: 640*5 is a
    /// valid VAE length but leaves 5 frames, which `patch_size = 4` cannot
    /// fold.
    #[test]
    fn padding_multiple_is_the_patch_multiple_not_the_vae_hop() {
        let wav = vec![0.5f32; 640 * 5];
        assert_eq!(pad_to_multiple(&wav, 640).expect("pad").len(), 640 * 5);
        assert_eq!(pad_to_multiple(&wav, 2560).expect("pad").len(), 2560 * 2);
    }

    #[test]
    fn padding_rejects_zero_multiple() {
        assert!(pad_to_multiple(&[0.0f32], 0).is_err());
    }

    /// Every value encodes its own `(channel, frame)` index, so a wrong
    /// permutation cannot pass: `latent[c][t]` must land at
    /// `folded[t / P][t % P][c]`.
    #[test]
    fn fold_places_each_frame_at_its_own_patch_slot() {
        let (_client, device) = cpu_setup();
        let (feat_dim, patch_size, t_ref) = (3usize, 2usize, 4usize);
        let frames = patch_size * t_ref;

        let data: Vec<f32> = (0..feat_dim * frames)
            .map(|i| {
                let (c, t) = (i / frames, i % frames);
                (c * 100 + t) as f32
            })
            .collect();
        let latent =
            Tensor::<CpuRuntime>::from_slice(&data, &[1, feat_dim, frames], &device).expect("in");

        let folded = fold_patches(&latent, patch_size, feat_dim).expect("fold");
        assert_eq!(folded.shape(), &[t_ref, patch_size, feat_dim]);

        let got: Vec<f32> = folded.contiguous().expect("contig").to_vec();
        for t in 0..t_ref {
            for p in 0..patch_size {
                for c in 0..feat_dim {
                    let idx = (t * patch_size + p) * feat_dim + c;
                    let want = (c * 100 + t * patch_size + p) as f32;
                    assert_eq!(
                        got[idx],
                        want,
                        "folded[{t}][{p}][{c}] must be latent[{c}][{}]",
                        t * patch_size + p
                    );
                }
            }
        }
    }

    #[test]
    fn fold_rejects_a_frame_count_that_is_not_a_whole_number_of_patches() {
        let (_client, device) = cpu_setup();
        let data = vec![0.0f32; 3 * 5];
        let latent = Tensor::<CpuRuntime>::from_slice(&data, &[1, 3, 5], &device).expect("in");
        assert!(fold_patches(&latent, 2, 3).is_err());
    }

    #[test]
    fn fold_rejects_a_wrong_feature_width() {
        let (_client, device) = cpu_setup();
        let data = vec![0.0f32; 3 * 4];
        let latent = Tensor::<CpuRuntime>::from_slice(&data, &[1, 3, 4], &device).expect("in");
        assert!(fold_patches(&latent, 2, 64).is_err());
    }
}
