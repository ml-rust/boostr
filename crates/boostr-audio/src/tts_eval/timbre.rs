//! Timbre drift proxy: AudioVAE latent similarity plus F0 deltas.
//!
//! This is NOT a speaker-verification score. The AudioVAE encoder is a
//! reconstruction codec: its latent carries everything the decoder needs to
//! rebuild the waveform, phonetic content and prosody included, and it was
//! never trained to separate speakers. Mean-pooling it over time and taking a
//! cosine against the reference clip's pooled latent gives a number that
//! moves when the voice drifts, which is enough for an A/B on the SAME
//! speaker and the same prompt set: base against adapter, one kernel
//! against another. It says nothing about whether two clips are the same
//! person in the sense a verifier means, and a render of different text
//! shifts it for content reasons alone. Read it as a relative drift track,
//! never as an absolute identity score.
//!
//! The pooled vector is the same latent [`VoxCpm2Model::encode_reference`]
//! feeds the transformer stack, before the patch fold; pooling over time
//! makes the fold irrelevant. A caller that already holds a loaded model can
//! pool with [`embed_with`] on its `vae_encoder` instead of loading the VAE a
//! second time.
//!
//! The follow-up is a real speaker-embedding model of the ECAPA-TDNN or
//! WavLM-SV class. It plugs in at [`embed_with`]: the cosine and the F0
//! deltas stay, only the vector's source changes, and `latent_cosine` then
//! becomes a speaker-similarity score.
//!
//! [`VoxCpm2Model::encode_reference`]: boostr::model::audio::voxcpm::VoxCpm2Model::encode_reference

use std::path::Path;

use crate::error::{Error, Result};
use boostr::model::audio::voxcpm::VoxCpmClient;
use boostr::model::audio::voxcpm::vae::AudioVaeEncoder;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, TensorOps, TypeConversionOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Sample rate the AudioVAE encoder is defined at.
pub const ENCODER_RATE: u32 = 16_000;

/// Drift of a render's voice against its reference clip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimbreProxy {
    /// Cosine between the render's and the reference's time-pooled AudioVAE
    /// latents, in `[-1, 1]`. Higher is closer; see the module docs for what
    /// it does and does not measure.
    pub latent_cosine: f64,
    /// `render F0 mean - reference F0 mean`, Hz. `None` when either clip has
    /// no voiced frames.
    pub f0_mean_delta_hz: Option<f64>,
    /// `render F0 std / reference F0 std`. `None` when either is missing or
    /// the reference std is zero.
    pub f0_std_ratio: Option<f64>,
}

/// F0 statistics of one clip, as `signal_stats` reports them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F0Stats {
    pub mean_hz: Option<f64>,
    pub std_hz: Option<f64>,
}

/// One loaded AudioVAE encoder, reused across a batch.
pub struct TimbreScorer<R: Runtime> {
    encoder: AudioVaeEncoder<R>,
}

impl<R: Runtime<DType = DType>> TimbreScorer<R> {
    /// Load the encoder from `audiovae.pth`, its safetensors conversion, or a
    /// directory holding either.
    pub fn from_checkpoint(path: &Path, device: &R::Device) -> Result<Self>
    where
        R::Client: TypeConversionOps<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
    {
        Ok(Self::new(AudioVaeEncoder::<R>::from_checkpoint(
            path, device,
        )?))
    }

    /// Wrap an already-loaded encoder.
    pub fn new(encoder: AudioVaeEncoder<R>) -> Self {
        Self { encoder }
    }

    /// Time-pooled latent of `samples_16k`. See [`embed_with`].
    pub fn embed<C: VoxCpmClient<R>>(&self, client: &C, samples_16k: &[f32]) -> Result<Vec<f32>> {
        embed_with(&self.encoder, client, samples_16k)
    }
}

/// Mean over frames of `encoder`'s latent `[1, feat_dim, frames]` for
/// `samples_16k` (mono, 16 kHz): one `feat_dim`-long vector.
pub fn embed_with<R, C>(
    encoder: &AudioVaeEncoder<R>,
    client: &C,
    samples_16k: &[f32],
) -> Result<Vec<f32>>
where
    R: Runtime<DType = DType>,
    C: VoxCpmClient<R>,
{
    if samples_16k.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "samples_16k",
            reason: "expected at least 1 sample, got 0".to_string(),
        });
    }
    let wave = Tensor::<R>::from_slice(samples_16k, &[1, 1, samples_16k.len()], client.device())?;
    let latent = encoder.forward(client, &wave)?;
    let shape = latent.shape().to_vec();
    if shape.len() != 3 || shape[0] != 1 || shape[2] == 0 {
        return Err(Error::DataError {
            reason: format!(
                "AudioVAE latent has shape {shape:?}, expected [1, feat_dim, frames > 0]"
            ),
        });
    }
    let (feat_dim, frames) = (shape[1], shape[2]);
    let values: Vec<f32> = latent.contiguous()?.try_to_vec()?;
    Ok(mean_over_frames(&values, feat_dim, frames))
}

/// Mean of a row-major `[feat_dim, frames]` block along `frames`.
fn mean_over_frames(values: &[f32], feat_dim: usize, frames: usize) -> Vec<f32> {
    (0..feat_dim)
        .map(|c| {
            let row = &values[c * frames..(c + 1) * frames];
            row.iter().map(|&v| f64::from(v)).sum::<f64>() / frames as f64
        })
        .map(|m| m as f32)
        .collect()
}

/// Cosine similarity of two equal-length vectors. Zero when either has no
/// magnitude.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Combine pooled latents and F0 statistics into a [`TimbreProxy`].
///
/// Returns an error when the two embeddings differ in length: that means two
/// different encoders produced them and the cosine is meaningless.
pub fn timbre_proxy(
    render_embed: &[f32],
    reference_embed: &[f32],
    render_f0: F0Stats,
    reference_f0: F0Stats,
) -> Result<TimbreProxy> {
    if render_embed.len() != reference_embed.len() {
        return Err(Error::InvalidArgument {
            arg: "render_embed",
            reason: format!(
                "render embedding has {} values, reference has {}",
                render_embed.len(),
                reference_embed.len()
            ),
        });
    }
    let f0_mean_delta_hz = match (render_f0.mean_hz, reference_f0.mean_hz) {
        (Some(r), Some(f)) => Some(r - f),
        _ => None,
    };
    let f0_std_ratio = match (render_f0.std_hz, reference_f0.std_hz) {
        (Some(r), Some(f)) if f > 0.0 => Some(r / f),
        _ => None,
    };
    Ok(TimbreProxy {
        latent_cosine: cosine(render_embed, reference_embed),
        f0_mean_delta_hz,
        f0_std_ratio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_of_parallel_and_orthogonal_vectors() {
        assert!((cosine(&[1.0, 2.0], &[2.0, 4.0]) - 1.0).abs() < 1e-9);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-9);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn pooling_averages_each_channel_row() {
        // [2 channels, 3 frames], row-major.
        let pooled = mean_over_frames(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0], 2, 3);
        assert_eq!(pooled, vec![2.0, 20.0]);
    }

    #[test]
    fn proxy_reports_f0_deltas_and_rejects_length_mismatch() {
        let r = F0Stats {
            mean_hz: Some(150.0),
            std_hz: Some(20.0),
        };
        let f = F0Stats {
            mean_hz: Some(140.0),
            std_hz: Some(10.0),
        };
        let p = timbre_proxy(&[1.0, 1.0], &[1.0, 1.0], r, f).expect("proxy");
        assert!((p.latent_cosine - 1.0).abs() < 1e-9);
        assert_eq!(p.f0_mean_delta_hz, Some(10.0));
        assert_eq!(p.f0_std_ratio, Some(2.0));

        let none = F0Stats {
            mean_hz: None,
            std_hz: None,
        };
        let p = timbre_proxy(&[1.0], &[1.0], none, f).expect("proxy");
        assert_eq!(p.f0_mean_delta_hz, None);
        assert_eq!(p.f0_std_ratio, None);

        assert!(timbre_proxy(&[1.0], &[1.0, 2.0], r, f).is_err());
    }
}
