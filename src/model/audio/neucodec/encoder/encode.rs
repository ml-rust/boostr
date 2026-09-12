//! The encode pipeline: pad, run both branches, truncate, join, project,
//! quantize.

use super::axes::{min_time, narrow_time, to_time_last};
use super::limits::{MAX_ENCODE_SAMPLES, check_encode_len, encode_alignment, encode_padding};
use super::model::NeuCodecEncoder;
use crate::error::{Error, Result};
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::model::audio::neucodec::fbank::{STACKED_DIM, seamless_fbank};
use numr::autograd::{Var, var_cat};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

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

impl<R: Runtime<DType = DType>> NeuCodecEncoder<R> {
    /// Encode 16 kHz mono `samples` into FSQ code indices `[1, 1, T]` (I32).
    ///
    /// `samples` must already be 16 kHz: the reference implementation never resamples a tensor
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

        let waveform =
            Tensor::<R>::from_slice(&padded, &[1, 1, padded.len()], device).map_err(Error::Numr)?;

        let semantic = self.semantic_branch(client, &padded, device)?;
        let acoustic = self
            .acoustic_encoder
            .forward(client, &Var::new(waveform.clone(), false))?;

        // The branches DISAGREE on frame count (8320 samples -> Ta = 26,
        // Ts = 25). The reference implementation neither interpolates nor aligns: it keeps the
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

#[cfg(test)]
mod tests {
    use super::super::axes::tests::var;
    use super::*;
    use crate::test_utils::cpu_setup;

    /// The channel-axis join is SEMANTIC FIRST: `[0, C)` semantic, `[C, 2C)`
    /// acoustic. The reverse order is shape-identical and silently wrong.
    #[test]
    fn concat_puts_semantic_in_the_low_channels() {
        let (client, device) = cpu_setup();

        let semantic = var(&[1.0, 2.0], &[1, 1, 2], &device);
        let acoustic = var(&[3.0, 4.0], &[1, 1, 2], &device);
        let joined = var_cat(&[&semantic, &acoustic], 1, &client).expect("cat");

        assert_eq!(joined.shape(), &[1, 2, 2]);
        assert_eq!(
            joined
                .tensor()
                .contiguous()
                .expect("contiguous")
                .to_vec::<f32>(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }
}
