//! [`ClapModel`]: both CLAP towers and their projections. Audio features
//! and token ids in, unit-length embeddings out, as transformers'
//! `ClapModel.get_audio_features` and `get_text_features` give them.

use super::audio_encoder::ClapAudioEncoder;
use super::client::ClapClient;
use super::config::ClapConfig;
use super::projection::ClapProjection;
use super::text::ClapTextEncoder;
use crate::error::Result;
use crate::nn::VarBuilder;
use crate::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::ops::{IndexingOps, ScalarOps, TensorOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A CLAP checkpoint: the HTSAT audio tower, the RoBERTa text tower, and a
/// projection for each into the shared space.
pub struct ClapModel<R: Runtime> {
    audio: ClapAudioEncoder<R>,
    audio_projection: ClapProjection<R>,
    text: ClapTextEncoder<R>,
    text_projection: ClapProjection<R>,
    config: ClapConfig,
}

impl<R: Runtime<DType = DType>> ClapModel<R> {
    /// Reads `audio_model.audio_encoder.*`, `audio_projection.*`,
    /// `text_model.*` and `text_projection.*` under `vb`.
    pub fn from_varbuilder<C>(
        client: &C,
        vb: &mut VarBuilder<'_, R>,
        config: &ClapConfig,
    ) -> Result<Self>
    where
        C: ClapClient<R>,
    {
        let mut a = vb.pp("audio_model.audio_encoder");
        let audio = ClapAudioEncoder::from_varbuilder(client, &mut a, config)?;
        drop(a);
        let mut ap = vb.pp("audio_projection");
        let audio_projection = ClapProjection::from_varbuilder(&mut ap, config.projection_act)?;
        drop(ap);
        let mut t = vb.pp("text_model");
        let text = ClapTextEncoder::from_varbuilder(&mut t, config)?;
        drop(t);
        let mut tp = vb.pp("text_projection");
        let text_projection = ClapProjection::from_varbuilder(&mut tp, config.projection_act)?;
        drop(tp);
        Ok(Self {
            audio,
            audio_projection,
            text,
            text_projection,
            config: config.clone(),
        })
    }

    pub fn config(&self) -> &ClapConfig {
        &self.config
    }

    /// Unit-length audio embeddings, `[batch, projection_dim]`, of
    /// `features` (`[batch, 1, frames, mels]`).
    pub fn audio_embeddings<C>(&self, client: &C, features: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: ClapClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let pooled = self.audio.forward(client, features)?;
        self.audio_projection.embed(client, &pooled)
    }

    /// Unit-length text embeddings, `[batch, projection_dim]`, of one row of
    /// token ids per text, special tokens included.
    pub fn text_embeddings<C>(
        &self,
        client: &C,
        ids: &[Vec<u32>],
        device: &R::Device,
    ) -> Result<Tensor<R>>
    where
        C: ClapClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R> + DequantOps<R>,
    {
        let pooled = self.text.forward(client, ids, device)?;
        self.text_projection.embed(client, &pooled)
    }
}
