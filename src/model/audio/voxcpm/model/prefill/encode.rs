//! [`VoxCpm2Model::encode_reference`] and [`VoxCpm2Model::encode_prompt`]:
//! a reference or continuation-prompt waveform to per-patch features.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::client::VoxCpmClient;
use crate::model::audio::voxcpm::model::loader::VoxCpm2Model;
use crate::model::audio::voxcpm::model::patches::{PadSide, fold_patches, pad_to_multiple};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> VoxCpm2Model<R> {
    /// Encode the reference waveform into per-patch features `[T_ref,
    /// patch_size, feat_dim]`.
    ///
    /// `ref_wav_16k` is mono 16 kHz PCM. It is right-padded with zeros to a
    /// multiple of `patch_size * 640` BEFORE the VAE encode — see
    /// [`VoxCpm2Config::ref_pad_multiple`](super::super::config::VoxCpm2Config::ref_pad_multiple)
    /// for why the VAE's own 640 modulus is not enough.
    ///
    /// The reference prefix uses the reference AUDIO only. Its transcript
    /// never enters the sequence.
    pub fn encode_reference<C>(&self, client: &C, ref_wav_16k: &[f32]) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
    {
        self.encode_padded(client, ref_wav_16k, PadSide::Right, "ref_wav_16k")
    }

    /// Encode a continuation prompt waveform into per-patch features
    /// `[T_prompt, patch_size, feat_dim]`, for
    /// [`PrefillRow::prompt_feat`](super::PrefillRow::prompt_feat).
    ///
    /// `prompt_wav_16k` is mono 16 kHz PCM. It is LEFT-padded with zeros to
    /// the same `patch_size * 640` multiple [`encode_reference`](Self::encode_reference)
    /// uses, so the prompt's real audio ends at its last patch. Its
    /// transcript is the caller's: tokenize `prompt_text ++ target_text` as
    /// the row's `text_token_ids`.
    pub fn encode_prompt<C>(&self, client: &C, prompt_wav_16k: &[f32]) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
    {
        self.encode_padded(client, prompt_wav_16k, PadSide::Left, "prompt_wav_16k")
    }

    /// Pad `wav_16k` on `side`, VAE-encode it and fold the latent into
    /// patches. `arg` names the waveform in the error.
    fn encode_padded<C>(
        &self,
        client: &C,
        wav_16k: &[f32],
        side: PadSide,
        arg: &'static str,
    ) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
    {
        if wav_16k.is_empty() {
            return Err(Error::InvalidArgument {
                arg,
                reason: "expected at least 1 sample of 16 kHz mono audio, got 0; \
                         pass a non-empty waveform"
                    .to_string(),
            });
        }
        let padded = pad_to_multiple(wav_16k, self.config.ref_pad_multiple(), side)?;
        let wave = Tensor::<R>::from_slice(padded.as_ref(), &[1, 1, padded.len()], self.device()?)?;
        // No dtype cast here: the encoder always loads and runs at F32 (see
        // `AudioVaeEncoder::from_checkpoint`'s docs for why it has no dtype
        // option), matching the waveform's own F32 samples exactly.
        let latent = self.vae_encoder.forward(client, &wave)?;
        fold_patches(&latent, self.config.patch_size, self.config.feat_dim)
    }
}
