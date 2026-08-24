//! [`NeuCodec`] — the FSQ quantizer plus the acoustic decoder, i.e. the
//! complete "audio tokens in, waveform out" half of NeuCodec.
//!
//! This is the pure-Rust listening path: given the 50 Hz FSQ code indices that
//! a model predicts, it reconstructs 24 kHz audio with no Python in the loop.
//! The encoder (BigCodec acoustic + Wav2Vec2-BERT semantic branches) lives in
//! [`super::encoder`] — both directions of NeuCodec are pure Rust.
//!
//! Pipeline, matching upstream `NeuCodec.decode_code`:
//!
//! ```text
//! indices [B, T]  (i32, 0..65_536)
//!   -> FSQ project_out            -> [B, T, 2048]
//!   -> NeuCodecDecoder            -> waveform [B, T * 480]
//! ```
//!
//! Upstream applies a separate `fc_post_a` Linear between the quantizer and the
//! decoder backbone. In this checkpoint that layer is exported as
//! `acoustic_decoder.fc`, so it is already the decoder's first stage and is NOT
//! duplicated here.

use crate::error::Result;
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::model::audio::neucodec::config::NeuCodecDecoderConfig;
use crate::model::audio::neucodec::decoder::NeuCodecDecoder;
use crate::nn::fsq::Fsq;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::path::Path;

/// FSQ quantizer + acoustic decoder.
pub struct NeuCodec<R: Runtime<DType = DType>> {
    quantizer: Fsq<R>,
    decoder: NeuCodecDecoder<R>,
}

impl<R: Runtime<DType = DType>> NeuCodec<R> {
    /// Assemble from an already-built quantizer and decoder.
    pub fn new(quantizer: Fsq<R>, decoder: NeuCodecDecoder<R>) -> Self {
        Self { quantizer, decoder }
    }

    /// Load both halves from a `neuphonic/neucodec` checkpoint (file or the
    /// directory containing `model.safetensors`).
    pub fn from_safetensors<P: AsRef<Path>>(path: P, device: &R::Device) -> Result<Self> {
        let path = path.as_ref();
        let quantizer = super::loader::load_fsq_quantizer::<R, _>(path, device)?;
        let decoder = NeuCodecDecoder::from_safetensors(path, device)?;
        Ok(Self::new(quantizer, decoder))
    }

    pub fn config(&self) -> &NeuCodecDecoderConfig {
        self.decoder.config()
    }

    pub fn decoder(&self) -> &NeuCodecDecoder<R> {
        &self.decoder
    }

    pub fn quantizer(&self) -> &Fsq<R> {
        &self.quantizer
    }

    /// Dequantize code indices into the decoder's input features.
    ///
    /// `indices`: `[B, T]`, integer dtype -> `[B, T, fc_in_dim]`.
    pub fn indices_to_features<C>(&self, client: &C, indices: &Tensor<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        self.quantizer.indices_to_codes(client, indices)
    }
}

impl NeuCodec<numr::runtime::cpu::CpuRuntime> {
    /// Full decode: code indices `[B, T]` -> waveform `[B, T * hop_length]`.
    ///
    /// CPU-only because the ISTFT tail is (see
    /// [`crate::model::audio::kokoro::istft`]).
    pub fn decode(
        &self,
        client: &numr::runtime::cpu::CpuClient,
        indices: &Tensor<numr::runtime::cpu::CpuRuntime>,
    ) -> Result<Tensor<numr::runtime::cpu::CpuRuntime>> {
        let features = self.indices_to_features(client, indices)?;
        self.decoder.forward(client, &features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{cpu_setup, neucodec_checkpoint};
    use numr::runtime::cpu::CpuRuntime;

    fn checkpoint() -> Option<std::path::PathBuf> {
        neucodec_checkpoint()
    }

    #[test]
    fn decodes_indices_to_waveform() {
        let Some(path) = checkpoint() else { return };
        let (client, device) = cpu_setup();
        let codec = NeuCodec::<CpuRuntime>::from_safetensors(path, &device).expect("load codec");
        let cfg = *codec.config();

        let frames = 6;
        let idx: Vec<i32> = vec![0, 1, 4095, 32768, 65535, 7];
        let indices = Tensor::<CpuRuntime>::try_from_slice(&idx, &[1, frames], &device).unwrap();

        let features = codec
            .indices_to_features(&client, &indices)
            .expect("dequantize");
        assert_eq!(features.shape(), &[1, frames, cfg.fc_in_dim]);

        let waveform = codec.decode(&client, &indices).expect("decode");
        assert_eq!(waveform.shape(), &[1, frames * cfg.hop_length]);
        for v in waveform.contiguous().unwrap().to_vec::<f32>() {
            assert!(v.is_finite(), "sample not finite: {v}");
        }
    }
}
