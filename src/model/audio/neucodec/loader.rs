//! Weight loading for [`NeuCodecDecoder`] from a `neuphonic/neucodec`
//! SafeTensors checkpoint.
//!
//! Only the `acoustic_decoder.*` tensors are read (136 of the checkpoint's
//! 811), so loading the decoder does NOT materialize the 2.4 GB encoder or
//! the Wav2Vec2-BERT semantic branch.
//!
//! Verified key layout (dumped from the real `model.safetensors` header):
//!
//! ```text
//! acoustic_decoder.fc.{weight[1024,2048],bias[1024]}
//! acoustic_decoder.embed.{weight[1024,1024,7],bias[1024]}
//! acoustic_decoder.prior_net.{0,1}.norm{1,2}.{weight,bias}[1024]
//! acoustic_decoder.prior_net.{0,1}.conv{1,2}.{weight[1024,1024,3],bias[1024]}
//! acoustic_decoder.layers.{0..11}.input_layernorm.weight[1024]          (no bias -> RMSNorm)
//! acoustic_decoder.layers.{0..11}.self_attn.{q,k,v,o}_proj.weight[1024,1024]  (no biases)
//! acoustic_decoder.layers.{0..11}.post_attention_layernorm.weight[1024]
//! acoustic_decoder.layers.{0..11}.mlp.fc1.weight[4096,1024]
//! acoustic_decoder.layers.{0..11}.mlp.fc2.weight[1024,4096]
//! acoustic_decoder.norm.{weight,bias}[1024]                             (has bias -> LayerNorm)
//! acoustic_decoder.post_net.{0,1}.…                                     (as prior_net)
//! acoustic_decoder.head.linear.{weight[1922,1024],bias[1922]}
//! ```
//!
//! Note the asymmetry that pins the norm families: the per-layer norms and the
//! attention/MLP projections have NO bias tensors, while `norm`, the resnet
//! norms, and both conv/linear endpoints DO.

use crate::error::{Error, Result};
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::model::audio::neucodec::config::NeuCodecDecoderConfig;
use crate::model::audio::neucodec::decoder::{NeuCodecDecoder, NeuCodecDecoderWeights};
use crate::model::audio::neucodec::istft_head::{IstftHead, IstftHeadWeights};
use crate::model::audio::neucodec::resnet_block::{ResnetBlock, ResnetBlockWeights};
use crate::model::audio::neucodec::transformer_block::{TransformerBlock, TransformerBlockWeights};
use crate::nn::{Conv1d, GroupNorm, LayerNorm, Linear, RmsNorm};
use numr::dtype::DType;
use numr::ops::PaddingMode;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::path::Path;

/// Default top-level prefix for the decoder's tensors in the checkpoint.
pub const DEFAULT_DECODER_PREFIX: &str = "acoustic_decoder";

/// Reads `acoustic_decoder.*` tensors and assembles a [`NeuCodecDecoder`].
struct DecoderLoader<'a, R: Runtime<DType = DType>> {
    loader: &'a mut SafeTensorsLoader,
    device: &'a R::Device,
    prefix: String,
    config: NeuCodecDecoderConfig,
}

impl<R: Runtime<DType = DType>> DecoderLoader<'_, R> {
    fn tensor(&mut self, name: &str, expected: &[usize]) -> Result<Tensor<R>> {
        let full = format!("{}.{}", self.prefix, name);
        let t = self.loader.load_tensor::<R>(&full, self.device)?;
        if t.shape() != expected {
            return Err(Error::ModelError {
                reason: format!(
                    "{full}: expected shape {expected:?}, checkpoint has {:?}",
                    t.shape()
                ),
            });
        }
        Ok(t)
    }

    fn linear(&mut self, name: &str, out_f: usize, in_f: usize, bias: bool) -> Result<Linear<R>> {
        let weight = self.tensor(&format!("{name}.weight"), &[out_f, in_f])?;
        let bias = if bias {
            Some(self.tensor(&format!("{name}.bias"), &[out_f])?)
        } else {
            None
        };
        Ok(Linear::new(weight, bias, false))
    }

    /// Same-padding Conv1d with bias — the only conv shape this decoder uses.
    fn conv1d(&mut self, name: &str, channels: usize, kernel: usize) -> Result<Conv1d<R>> {
        let weight = self.tensor(&format!("{name}.weight"), &[channels, channels, kernel])?;
        let bias = self.tensor(&format!("{name}.bias"), &[channels])?;
        Ok(Conv1d::new(
            weight,
            Some(bias),
            1,
            PaddingMode::Same,
            1,
            1,
            false,
        ))
    }

    fn group_norm(&mut self, name: &str, channels: usize) -> Result<GroupNorm<R>> {
        let weight = self.tensor(&format!("{name}.weight"), &[channels])?;
        let bias = self.tensor(&format!("{name}.bias"), &[channels])?;
        Ok(GroupNorm::new(
            weight,
            bias,
            self.config.resnet_norm_groups,
            self.config.resnet_norm_eps,
            false,
        ))
    }

    fn resnet_block(&mut self, name: &str) -> Result<ResnetBlock<R>> {
        let hidden = self.config.hidden_size;
        let k = self.config.resnet_kernel_size;
        Ok(ResnetBlock::new(ResnetBlockWeights {
            norm1: self.group_norm(&format!("{name}.norm1"), hidden)?,
            conv1: self.conv1d(&format!("{name}.conv1"), hidden, k)?,
            norm2: self.group_norm(&format!("{name}.norm2"), hidden)?,
            conv2: self.conv1d(&format!("{name}.conv2"), hidden, k)?,
        }))
    }

    fn transformer_block(&mut self, idx: usize) -> Result<TransformerBlock<R>> {
        let hidden = self.config.hidden_size;
        let mlp = self.config.mlp_intermediate_size;
        let eps = self.config.rms_norm_eps;
        let p = format!("layers.{idx}");

        // Per-layer norms have weight only (no bias) => RMSNorm.
        let input_layernorm = RmsNorm::new(
            self.tensor(&format!("{p}.input_layernorm.weight"), &[hidden])?,
            eps,
            false,
        );
        let post_attention_layernorm = RmsNorm::new(
            self.tensor(&format!("{p}.post_attention_layernorm.weight"), &[hidden])?,
            eps,
            false,
        );

        // Attention projections carry no biases (`attention_bias: false`).
        let weights = TransformerBlockWeights {
            input_layernorm,
            q_proj: self.linear(&format!("{p}.self_attn.q_proj"), hidden, hidden, false)?,
            k_proj: self.linear(&format!("{p}.self_attn.k_proj"), hidden, hidden, false)?,
            v_proj: self.linear(&format!("{p}.self_attn.v_proj"), hidden, hidden, false)?,
            o_proj: self.linear(&format!("{p}.self_attn.o_proj"), hidden, hidden, false)?,
            post_attention_layernorm,
            mlp_fc1: self.linear(&format!("{p}.mlp.fc1"), mlp, hidden, false)?,
            mlp_fc2: self.linear(&format!("{p}.mlp.fc2"), hidden, mlp, false)?,
        };
        TransformerBlock::new(weights, self.config.num_heads, self.config.head_dim)
    }

    fn build(&mut self) -> Result<NeuCodecDecoderWeights<R>> {
        let hidden = self.config.hidden_size;

        let fc = self.linear("fc", hidden, self.config.fc_in_dim, true)?;
        let embed = self.conv1d("embed", hidden, self.config.embed_kernel_size)?;

        let mut prior_net = Vec::with_capacity(self.config.num_prior_resnet_blocks);
        for i in 0..self.config.num_prior_resnet_blocks {
            prior_net.push(self.resnet_block(&format!("prior_net.{i}"))?);
        }

        let mut layers = Vec::with_capacity(self.config.num_transformer_layers);
        for i in 0..self.config.num_transformer_layers {
            layers.push(self.transformer_block(i)?);
        }

        // The final norm HAS a bias => LayerNorm, unlike the per-layer norms.
        let norm = LayerNorm::new(
            self.tensor("norm.weight", &[hidden])?,
            self.tensor("norm.bias", &[hidden])?,
            self.config.layer_norm_eps,
            false,
        );

        let mut post_net = Vec::with_capacity(self.config.num_post_resnet_blocks);
        for i in 0..self.config.num_post_resnet_blocks {
            post_net.push(self.resnet_block(&format!("post_net.{i}"))?);
        }

        let head = IstftHead::new(
            IstftHeadWeights {
                linear: self.linear("head.linear", self.config.head_out_dim(), hidden, true)?,
            },
            self.config.n_fft,
            self.config.mag_clamp_max,
        )?;

        Ok(NeuCodecDecoderWeights {
            fc,
            embed,
            prior_net,
            layers,
            norm,
            post_net,
            head,
        })
    }
}

impl<R: Runtime<DType = DType>> NeuCodecDecoder<R> {
    /// Load the acoustic decoder from a `neuphonic/neucodec` checkpoint.
    ///
    /// `path` may be either the `model.safetensors` file or the directory
    /// containing it. Uses [`NeuCodecDecoderConfig::default`] — the verified
    /// geometry of the released checkpoint.
    pub fn from_safetensors<P: AsRef<Path>>(path: P, device: &R::Device) -> Result<Self> {
        Self::from_safetensors_with(
            path,
            NeuCodecDecoderConfig::default(),
            DEFAULT_DECODER_PREFIX,
            device,
        )
    }

    /// Load with an explicit config and checkpoint prefix.
    ///
    /// Every tensor is shape-checked against `config`, so a config that
    /// disagrees with the checkpoint fails loudly at the first mismatched
    /// tensor rather than silently building a wrong model.
    pub fn from_safetensors_with<P: AsRef<Path>>(
        path: P,
        config: NeuCodecDecoderConfig,
        prefix: &str,
        device: &R::Device,
    ) -> Result<Self> {
        config.validate()?;
        let mut loader = SafeTensorsLoader::open(path)?;
        let weights = DecoderLoader::<R> {
            loader: &mut loader,
            device,
            prefix: prefix.to_string(),
            config,
        }
        .build()?;
        Self::new(config, weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    /// The real checkpoint, if present. Loader tests are skipped when the
    /// weights are not downloaded (they live outside the repo).
    const CHECKPOINT: &str = "/home/farhan/Projects/models/neucodec/model.safetensors";

    fn checkpoint() -> Option<&'static Path> {
        let p = Path::new(CHECKPOINT);
        p.exists().then_some(p)
    }

    #[test]
    fn rejects_missing_file() {
        let device = <CpuRuntime as Runtime>::default_device();
        assert!(
            NeuCodecDecoder::<CpuRuntime>::from_safetensors(
                "/nonexistent/model.safetensors",
                &device
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_wrong_prefix() {
        let Some(path) = checkpoint() else { return };
        let device = <CpuRuntime as Runtime>::default_device();
        assert!(
            NeuCodecDecoder::<CpuRuntime>::from_safetensors_with(
                path,
                NeuCodecDecoderConfig::default(),
                "not_a_real_prefix",
                &device,
            )
            .is_err()
        );
    }

    /// A config that disagrees with the checkpoint must fail on shape, not
    /// silently build a mis-shaped model.
    #[test]
    fn rejects_config_disagreeing_with_checkpoint() {
        let Some(path) = checkpoint() else { return };
        let device = <CpuRuntime as Runtime>::default_device();
        let cfg = NeuCodecDecoderConfig {
            mlp_intermediate_size: 2048, // real checkpoint is 4096
            ..Default::default()
        };
        let err = NeuCodecDecoder::<CpuRuntime>::from_safetensors_with(
            path,
            cfg,
            DEFAULT_DECODER_PREFIX,
            &device,
        );
        assert!(err.is_err(), "wrong mlp width must be rejected");
    }

    #[test]
    fn loads_real_checkpoint_and_generates_waveform() {
        let Some(path) = checkpoint() else { return };
        let (client, device) = cpu_setup();
        let decoder =
            NeuCodecDecoder::<CpuRuntime>::from_safetensors(path, &device).expect("load decoder");

        assert!(
            !decoder.is_training_mode(),
            "a loaded pretrained decoder must be in eval mode"
        );

        let cfg = *decoder.config();
        let frames = 8;
        let x_data: Vec<f32> = (0..(frames * cfg.fc_in_dim))
            .map(|i| ((i as f32) * 0.001).sin() * 0.5)
            .collect();
        let x = numr::autograd::Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_data, &[1, frames, cfg.fc_in_dim], &device),
            false,
        );

        let waveform = decoder.forward(&client, &x).expect("decode");
        assert_eq!(waveform.shape(), &[1, frames * cfg.hop_length]);
        for v in waveform.contiguous().unwrap().to_vec::<f32>() {
            assert!(v.is_finite(), "decoded sample is not finite: {v}");
        }
    }
}
