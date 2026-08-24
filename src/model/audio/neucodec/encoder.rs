//! [`NeuCodecEncoder`] — the full "16 kHz waveform in, FSQ code indices out"
//! half of NeuCodec, wiring the already-ported pieces into upstream's
//! `NeuCodec.encode_code`.
//!
//! ```text
//! samples [T] @ 16 kHz
//!   -> right zero-pad to a multiple of 320          [1, 1, Tp]
//!   -> semantic: fbank -> SemanticEncoder -> ᵀ -> SemanticAdapter   [1, 1024, Ts]
//!   -> acoustic: AcousticEncoder                                    [1, 1024, Ta]
//!   -> truncate BOTH to min(Ts, Ta) = T
//!   -> cat([semantic, acoustic], dim = 1)           [1, 2048, T]
//!   -> fc_prior (applied on [1, T, 2048])           [1, 2048, T]
//!   -> ResidualFsq::encode -> indices [1, T, 1] -> permute  [1, 1, T]
//! ```
//!
//! Two steps below look like bugs and are not; both are documented at their
//! call sites and verified against upstream: the padding ALWAYS fires (even
//! when the length is already a multiple of 320), and the two branches produce
//! different frame counts that are TRUNCATED, never interpolated or aligned.

use crate::error::{Error, Result};
use crate::model::audio::neucodec::acoustic_encoder::{AcousticEncoder, encoder_hop_length};
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::model::audio::neucodec::fbank::{SAMPLE_RATE, STACKED_DIM, seamless_fbank};
use crate::model::audio::neucodec::loader;
use crate::model::audio::neucodec::semantic_adapter::{SEMANTIC_ADAPTER_CHANNELS, SemanticAdapter};
use crate::model::audio::neucodec::semantic_encoder::SemanticEncoder;
use crate::nn::Linear;
use crate::nn::fsq::ResidualFsq;
use crate::nn::var_contiguous;
use numr::autograd::{Var, var_cat, var_narrow, var_permute};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::path::Path;

/// Width of the concatenated semantic+acoustic prior: one 1024-wide branch
/// each, and also the FSQ quantizer's `dim`.
pub const PRIOR_DIM: usize = 2 * SEMANTIC_ADAPTER_CHANNELS;

/// Longest input [`NeuCodecEncoder::encode`] accepts, in samples: 60 s at
/// 16 kHz.
///
/// The semantic branch's self-attention (16 heads, 16 layers) builds a dense
/// `[1, heads, T, T]` score tensor per layer, so cost grows with the SQUARE
/// of the input length. At this limit `T` is ~3,000 frames (50 Hz), and one
/// layer's score tensor alone is `16 * 3,000 * 3,000 * 4` bytes ~= 576 MB. A
/// 26-minute clip (`T` ~ 80,000) would need on the order of a terabyte per
/// layer and cannot succeed — `encode` refuses it up front instead of dying
/// deep inside a matmul with no indication that input length was the cause.
pub const MAX_ENCODE_SAMPLES: usize = 60 * SAMPLE_RATE;

/// Check a waveform length against the encode limit before anything is
/// allocated or uploaded.
///
/// Split out as a free function so the refusal can be tested without a
/// checkpoint: it is the whole point of [`MAX_ENCODE_SAMPLES`], and a guard
/// whose only test skips when no model is present is not a tested guard.
pub fn check_encode_len(len: usize, max_samples: usize) -> Result<()> {
    if len == 0 {
        return Err(Error::InvalidArgument {
            arg: "samples",
            reason: "expected a non-empty 16 kHz mono waveform".to_string(),
        });
    }
    if len > max_samples {
        let seconds = len as f64 / SAMPLE_RATE as f64;
        let limit_seconds = max_samples as f64 / SAMPLE_RATE as f64;
        return Err(Error::InvalidArgument {
            arg: "samples",
            reason: format!(
                "{len} samples ({seconds:.1} s) exceeds the {max_samples}-sample \
                 ({limit_seconds:.1} s) encode limit: the semantic branch's attention \
                 cost is quadratic in input length; split the audio into shorter \
                 utterance clips and encode each separately"
            ),
        });
    }
    Ok(())
}

/// Number of samples the waveform must be a multiple of before encoding: the
/// acoustic encoder's total stride (product of `ENCODER_STRIDES`, 320), i.e.
/// the 16 kHz -> 50 Hz ratio.
pub fn encode_alignment() -> usize {
    encoder_hop_length()
}

/// Samples of right zero-padding upstream appends before encoding.
///
/// **This always returns a non-zero count.** Upstream computes
/// `pad = 320 - (T % 320)` unconditionally, so a length that is already a
/// multiple of 320 gets a FULL extra 320 samples appended (8000 -> 8320,
/// 8320 -> 8640). Do not "optimize" the exact-multiple case away — it changes
/// the frame count and therefore every emitted code index.
pub fn encode_padding(len: usize) -> usize {
    let stride = encode_alignment();
    stride - (len % stride)
}

/// Intermediates of [`NeuCodecEncoder::encode_stages`], exposed so a parity
/// test can localize a failure to one branch instead of only seeing wrong
/// indices at the end.
pub struct EncodeStages<R: Runtime> {
    /// The right zero-padded waveform, `[1, 1, Tp]`.
    pub padded: Tensor<R>,
    /// Semantic branch after the adapter, `[1, 1024, Ts]` — PRE-truncation, so
    /// a padding bug shows up directly as a wrong `Ts`.
    pub semantic: Tensor<R>,
    /// Acoustic branch, `[1, 1024, Ta]` — PRE-truncation.
    pub acoustic: Tensor<R>,
    /// Post-`fc_prior` prior, `[1, 2048, T]` with `T = min(Ts, Ta)`.
    pub prior: Tensor<R>,
    /// FSQ code indices, `[1, 1, T]`, `DType::I32`.
    pub indices: Tensor<R>,
}

/// Already-built parts for [`NeuCodecEncoder`], following the `*Weights`
/// convention used across this module.
pub struct NeuCodecEncoderWeights<R: Runtime> {
    pub acoustic_encoder: AcousticEncoder<R>,
    pub semantic_encoder: SemanticEncoder<R>,
    pub semantic_adapter: SemanticAdapter<R>,
    /// Upstream `fc_prior`, stored as `fc_encoder.*` in the checkpoint.
    pub fc_prior: Linear<R>,
    pub quantizer: ResidualFsq<R>,
}

/// The full NeuCodec encoder: both branches, the prior projection, and the
/// residual FSQ quantizer.
pub struct NeuCodecEncoder<R: Runtime> {
    acoustic_encoder: AcousticEncoder<R>,
    semantic_encoder: SemanticEncoder<R>,
    semantic_adapter: SemanticAdapter<R>,
    fc_prior: Linear<R>,
    quantizer: ResidualFsq<R>,
}

impl<R: Runtime<DType = DType>> NeuCodecEncoder<R> {
    /// Assemble from already-built parts, validating that the prior projection
    /// and the quantizer agree on [`PRIOR_DIM`].
    pub fn new(weights: NeuCodecEncoderWeights<R>) -> Result<Self> {
        check_fc_prior(&weights.fc_prior)?;

        let dim = weights.quantizer.config().dim;
        if dim != PRIOR_DIM {
            return Err(Error::ModelError {
                reason: format!("quantizer dim {dim} does not match prior width {PRIOR_DIM}"),
            });
        }

        Ok(Self {
            acoustic_encoder: weights.acoustic_encoder,
            semantic_encoder: weights.semantic_encoder,
            semantic_adapter: weights.semantic_adapter,
            fc_prior: weights.fc_prior,
            quantizer: weights.quantizer,
        })
    }

    /// Load every part from a `neuphonic/neucodec` checkpoint (file, or the
    /// directory containing `model.safetensors`).
    pub fn from_safetensors<P: AsRef<Path>>(path: P, device: &R::Device) -> Result<Self> {
        let path = path.as_ref();
        Self::new(NeuCodecEncoderWeights {
            acoustic_encoder: loader::load_acoustic_encoder::<R, _>(path, device)?,
            semantic_encoder: loader::load_semantic_encoder::<R, _>(path, device)?,
            semantic_adapter: loader::load_semantic_adapter::<R, _>(path, device)?,
            fc_prior: loader::load_fc_prior::<R, _>(path, device)?,
            quantizer: loader::load_residual_fsq::<R, _>(path, device)?,
        })
    }

    pub fn acoustic_encoder(&self) -> &AcousticEncoder<R> {
        &self.acoustic_encoder
    }

    pub fn semantic_encoder(&self) -> &SemanticEncoder<R> {
        &self.semantic_encoder
    }

    pub fn semantic_adapter(&self) -> &SemanticAdapter<R> {
        &self.semantic_adapter
    }

    pub fn quantizer(&self) -> &ResidualFsq<R> {
        &self.quantizer
    }

    /// Encode 16 kHz mono `samples` into FSQ code indices `[1, 1, T]` (I32).
    ///
    /// `samples` must already be 16 kHz: upstream never resamples a tensor
    /// input, so neither does this. Refuses inputs longer than
    /// [`MAX_ENCODE_SAMPLES`] — see [`Self::encode_with_limit`] to override.
    pub fn encode<C>(&self, client: &C, samples: &[f32], device: &R::Device) -> Result<Tensor<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        self.encode_with_limit(client, samples, device, MAX_ENCODE_SAMPLES)
    }

    /// [`Self::encode`], with the [`MAX_ENCODE_SAMPLES`] refusal threshold
    /// replaced by `max_samples`.
    ///
    /// For a caller whose device can afford the quadratic attention cost of
    /// a longer clip. Everything else about `encode` is unchanged.
    pub fn encode_with_limit<C>(
        &self,
        client: &C,
        samples: &[f32],
        device: &R::Device,
        max_samples: usize,
    ) -> Result<Tensor<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        Ok(self
            .encode_stages_with_limit(client, samples, device, max_samples)?
            .indices)
    }

    /// [`Self::encode`], keeping the per-branch intermediates.
    pub fn encode_stages<C>(
        &self,
        client: &C,
        samples: &[f32],
        device: &R::Device,
    ) -> Result<EncodeStages<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        self.encode_stages_with_limit(client, samples, device, MAX_ENCODE_SAMPLES)
    }

    /// Single choke point for the length guard: both [`Self::encode`] (via
    /// [`Self::encode_with_limit`]) and [`Self::encode_stages`] call this.
    fn encode_stages_with_limit<C>(
        &self,
        client: &C,
        samples: &[f32],
        device: &R::Device,
        max_samples: usize,
    ) -> Result<EncodeStages<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        check_encode_len(samples.len(), max_samples)?;

        // Always-fires right zero-pad — see `encode_padding`.
        let mut padded = Vec::with_capacity(samples.len() + encode_alignment());
        padded.extend_from_slice(samples);
        padded.resize(samples.len() + encode_padding(samples.len()), 0.0);

        let waveform = Tensor::<R>::try_from_slice(&padded, &[1, 1, padded.len()], device)
            .map_err(Error::Numr)?;

        let semantic = self.semantic_branch(client, &padded, device)?;
        let acoustic = self
            .acoustic_encoder
            .forward(client, &Var::new(waveform.clone(), false))?;

        // The branches DISAGREE on frame count (8320 samples -> Ta = 26,
        // Ts = 25). Upstream neither interpolates nor aligns: it keeps the
        // earliest `min(Ta, Ts)` frames of both and drops the tail.
        let min_len = min_time(&semantic, &acoustic)?;
        let semantic_cut = narrow_time(&semantic, min_len)?;
        let acoustic_cut = narrow_time(&acoustic, min_len)?;

        // SEMANTIC FIRST on the channel axis: channels [0, 1024) are semantic,
        // [1024, 2048) acoustic. The reverse order is shape-identical and
        // silently wrong.
        let joined = var_cat(&[&semantic_cut, &acoustic_cut], 1, client).map_err(Error::Numr)?;

        // `fc_prior` acts on the channel axis, so it runs on [B, T, 2048].
        let joined_tl = to_time_last(&joined)?;
        let prior_tl = self.fc_prior.forward(client, &joined_tl)?;
        let prior = to_time_last(&prior_tl)?;

        // `prior_tl` is already channels-last [B, T, 2048] — reuse it instead
        // of permuting `prior` (channels-first) back, which would be an
        // identity round trip through two needless permute+contiguous calls.
        let (_codes, indices) = self.quantizer.encode(client, &prior_tl)?;
        // indices: [B, T, num_quantizers = 1] -> [B, 1, T], by axis permute.
        let indices = indices
            .permute(&[0, 2, 1])
            .map_err(Error::Numr)?
            .contiguous()
            .map_err(Error::Numr)?;

        Ok(EncodeStages {
            padded: waveform,
            semantic: semantic.tensor().clone(),
            acoustic: acoustic.tensor().clone(),
            prior: prior.tensor().clone(),
            indices,
        })
    }

    /// Semantic branch: Kaldi fbank features -> Wav2Vec2-BERT conformer ->
    /// channels-first -> adapter. Returns `[1, 1024, Ts]`.
    fn semantic_branch<C>(&self, client: &C, padded: &[f32], device: &R::Device) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let features = seamless_fbank::<R>(padded, device)?;
        let frames = features
            .shape()
            .first()
            .copied()
            .ok_or_else(|| Error::ModelError {
                reason: format!("fbank returned a rank-0 tensor: {:?}", features.shape()),
            })?;
        let features = features
            .reshape(&[1, frames, STACKED_DIM])
            .map_err(Error::Numr)?;

        // SemanticEncoder emits [B, Ts, 1024]; the adapter wants channels-first.
        let hidden = self
            .semantic_encoder
            .forward(client, &Var::new(features, false))?;
        let hidden = to_time_last(&hidden)?;
        self.semantic_adapter.forward(client, &hidden)
    }
}

/// `fc_prior` must be `Linear(2048 -> 2048)` WITH bias.
fn check_fc_prior<R: Runtime>(linear: &Linear<R>) -> Result<()> {
    let shape = linear.weight().tensor().shape();
    if shape != [PRIOR_DIM, PRIOR_DIM].as_slice() {
        return Err(Error::ModelError {
            reason: format!("fc_prior weight shape {shape:?} != [{PRIOR_DIM}, {PRIOR_DIM}]"),
        });
    }
    if linear.bias().is_none() {
        return Err(Error::ModelError {
            reason: "fc_prior must have a bias".to_string(),
        });
    }
    Ok(())
}

/// Swap the last two axes of a rank-3 `Var`, materializing the result so the
/// downstream reshape/conv sees a contiguous buffer.
///
/// `[B, X, Y] <-> [B, Y, X]` — this is how the pipeline moves between the
/// channels-first layout the convs need and the channels-last layout `Linear`
/// and the quantizer need.
fn to_time_last<R: Runtime<DType = DType>>(x: &Var<R>) -> Result<Var<R>> {
    if x.shape().len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("expected a rank-3 tensor, got {:?}", x.shape()),
        });
    }
    var_contiguous(&var_permute(x, &[0, 2, 1]).map_err(Error::Numr)?)
}

/// Length of the trailing time axis of a channels-first `[B, C, T]` `Var`.
fn time_len<R: Runtime>(x: &Var<R>, arg: &'static str) -> Result<usize> {
    let shape = x.shape();
    match (shape.len(), shape.get(2).copied()) {
        (3, Some(t)) => Ok(t),
        _ => Err(Error::InvalidArgument {
            arg,
            reason: format!("expected channels-first [B, C, T], got {shape:?}"),
        }),
    }
}

/// `min(Ts, Ta)` over the two branches.
fn min_time<R: Runtime>(semantic: &Var<R>, acoustic: &Var<R>) -> Result<usize> {
    let ts = time_len(semantic, "semantic")?;
    let ta = time_len(acoustic, "acoustic")?;
    Ok(ts.min(ta))
}

/// Keep the EARLIEST `len` frames of a channels-first `[B, C, T]` `Var`.
fn narrow_time<R: Runtime<DType = DType>>(x: &Var<R>, len: usize) -> Result<Var<R>>
where
    R::Client: NeuCodecClient<R>,
{
    if time_len(x, "x")? == len {
        return Ok(x.alias());
    }
    var_contiguous(&var_narrow(x, 2, 0, len).map_err(Error::Numr)?)
}

mod frames;

#[cfg(test)]
mod tests;
