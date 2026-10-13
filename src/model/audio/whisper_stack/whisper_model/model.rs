//! `WhisperModel` struct, construction, and the encoder entry point.

use crate::error::Result;
use crate::model::audio::whisper::WhisperEncoder;
use crate::model::audio::whisper_decoder::WhisperDecoder;
use crate::model::config::AudioConfig;
use crate::nn::VarBuilder;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, MatmulOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps,
    TensorOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Full Whisper model (encoder + decoder).
pub struct WhisperModel<R: Runtime> {
    pub encoder: WhisperEncoder<R>,
    pub decoder: WhisperDecoder<R>,
    config: AudioConfig,
}

impl<R: Runtime<DType = DType>> WhisperModel<R> {
    /// Load from a VarBuilder rooted at the model top level. Expects `model.encoder.*`
    /// and `model.decoder.*` underneath.
    pub fn from_varbuilder(vb: &mut VarBuilder<'_, R>, config: &AudioConfig) -> Result<Self> {
        let mut model_vb = vb.pp("model");
        let mut enc_vb = model_vb.pp("encoder");
        let encoder = WhisperEncoder::from_varbuilder(&mut enc_vb, config)?;
        drop(enc_vb);
        let mut dec_vb = model_vb.pp("decoder");
        let decoder = WhisperDecoder::from_varbuilder(&mut dec_vb, config)?;
        Ok(Self {
            encoder,
            decoder,
            config: config.clone(),
        })
    }

    pub fn config(&self) -> &AudioConfig {
        &self.config
    }

    /// Run the encoder on a mel spectrogram tensor `[B, num_mel_bins, audio_len]`.
    ///
    /// Returns the encoder hidden state `[B, S, hidden]` that both a greedy
    /// decode and any custom decode loop consume.
    pub fn encode<C>(&self, client: &C, mel: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ActivationOps<R>
            + NormalizationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + ShapeOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + ConvOps<R> + ReduceOps<R> + BinaryOps<R>,
    {
        self.encoder.forward_inference(client, mel)
    }
}
