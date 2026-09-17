//! [`VoxCpm2Model::encode_reference`]: reference waveform to per-patch
//! features.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::client::VoxCpmClient;
use crate::model::audio::voxcpm::model::loader::VoxCpm2Model;
use crate::model::audio::voxcpm::model::patches::{fold_patches, pad_to_multiple};
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
    /// Voice-clone mode uses the reference AUDIO only. There is no reference
    /// transcript on this path.
    pub fn encode_reference<C>(&self, client: &C, ref_wav_16k: &[f32]) -> Result<Tensor<R>>
    where
        C: VoxCpmClient<R>,
    {
        if ref_wav_16k.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "ref_wav_16k",
                reason: "expected at least 1 sample, got 0".to_string(),
            });
        }
        let padded = pad_to_multiple(ref_wav_16k, self.config.ref_pad_multiple())?;
        let wave = Tensor::<R>::from_slice(padded.as_ref(), &[1, 1, padded.len()], self.device()?)?;
        // No dtype cast here: the encoder always loads and runs at F32 (see
        // `AudioVaeEncoder::from_checkpoint`'s docs for why it has no dtype
        // option), matching `ref_wav_16k`'s own F32 samples exactly.
        let latent = self.vae_encoder.forward(client, &wave)?;
        fold_patches(&latent, self.config.patch_size, self.config.feat_dim)
    }
}
