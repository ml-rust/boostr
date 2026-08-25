//! SafeTensors loading for Silero VAD.
//!
//! The checkpoint is 15 tensors and 309 633 parameters, so it is read whole
//! rather than streamed. Tensor names are taken verbatim — no HuggingFace name
//! normalization applies to this checkpoint.

use std::path::Path;

use crate::error::Result;
use crate::model::audio::vad::config::{ENCODER_STRIDES, VadConfig};
use crate::model::audio::vad::model::{SileroVad, SileroVadWeights};
use crate::nn::VarMap;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> SileroVad<R> {
    /// Load the 16 kHz checkpoint (`silero_vad_16k.safetensors`).
    pub fn from_safetensors<P: AsRef<Path>>(path: P, device: &R::Device) -> Result<Self> {
        Self::from_safetensors_with(path, device, VadConfig::silero_16k())
    }

    /// Load a checkpoint whose geometry is described by `config` — either
    /// [`VadConfig::silero_16k`] or [`VadConfig::silero_8k`]. Every shape is
    /// checked against `config` by [`SileroVad::new`], so loading the 8 kHz
    /// weights as 16 kHz fails rather than producing nonsense.
    pub fn from_safetensors_with<P: AsRef<Path>>(
        path: P,
        device: &R::Device,
        config: VadConfig,
    ) -> Result<Self> {
        let map = VarMap::<R>::from_safetensors(path, device)?;
        let mut encoder = Vec::with_capacity(ENCODER_STRIDES.len());
        for i in 0..ENCODER_STRIDES.len() {
            encoder.push((
                fetch(&map, &format!("encoder.{i}.reparam_conv.weight"))?,
                fetch(&map, &format!("encoder.{i}.reparam_conv.bias"))?,
            ));
        }
        let weights = SileroVadWeights {
            stft_basis: fetch(&map, "stft.forward_basis_buffer")?,
            encoder,
            rnn_weight_ih: fetch(&map, "decoder.rnn.weight_ih")?,
            rnn_weight_hh: fetch(&map, "decoder.rnn.weight_hh")?,
            rnn_bias_ih: fetch(&map, "decoder.rnn.bias_ih")?,
            rnn_bias_hh: fetch(&map, "decoder.rnn.bias_hh")?,
            head_weight: fetch(&map, "decoder.decoder.2.weight")?,
            head_bias: fetch(&map, "decoder.decoder.2.bias")?,
        };
        Self::new(config, weights)
    }
}

/// Read `name` out of a loaded checkpoint. Shapes are checked later, in
/// [`SileroVad::new`], so one path validates them for both entry points.
fn fetch<R: Runtime<DType = DType>>(map: &VarMap<R>, name: &str) -> Result<Tensor<R>> {
    map.get_tensor(name).cloned()
}
