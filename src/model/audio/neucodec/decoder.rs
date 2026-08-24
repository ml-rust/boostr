//! `NeuCodecDecoder` — the NeuCodec acoustic decoder top-level assembly.
//!
//! Architecture (VERIFIED from `acoustic_decoder.*` in the real
//! `neuphonic/neucodec` `model.safetensors` — NOT `config.json` or the
//! GitHub source, both of which are partly wrong about this decoder):
//!
//! ```text
//! input [B, T, 2048]                      (FSQ project_out features)
//!   -> fc            Linear[1024, 2048]+bias          -> [B, T, 1024]
//!   -> embed         Conv1d(1024->1024, k=7)+bias      -> [B, 1024, T]  (channels-first)
//!   -> prior_net     2x ResnetBlock                    -> [B, 1024, T]
//!   -> (permute)                                       -> [B, T, 1024]  (channels-last)
//!   -> 12x TransformerBlock (RMSNorm/attn/RMSNorm/MLP) -> [B, T, 1024]
//!   -> (permute)                                       -> [B, 1024, T]
//!   -> post_net      2x ResnetBlock                    -> [B, 1024, T]
//!   -> (permute)                                       -> [B, T, 1024]
//!   -> norm          LayerNorm(eps=1e-6, weight+bias)  -> [B, T, 1024]
//!   -> head.linear   Linear[1922, 1024]+bias           -> [B, T, 1922]
//!   -> split/activate -> (mag [B, 961, T], phase [B, 961, T])
//!   -> istft (n_fft=1920, hop=480)                     -> waveform [B, samples]
//! ```
//!
//! `samples == T * hop_length` under Vocos `padding="same"` framing — one hop
//! per latent frame (see [`NeuCodecDecoder::forward`] tests for the exact
//! derivation).
//!
//! This module is architecture-only: no weight loading. Construct with
//! synthetic weights via [`NeuCodecDecoderWeights`]; a loader is a separate
//! unit.

use crate::error::{Error, Result};
use crate::model::audio::kokoro::{IStftOptions, IStftPadding, hann_window, istft};
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::model::audio::neucodec::config::NeuCodecDecoderConfig;
use crate::model::audio::neucodec::istft_head::IstftHead;
use crate::model::audio::neucodec::resnet_block::ResnetBlock;
use crate::model::audio::neucodec::transformer_block::TransformerBlock;
use crate::nn::{Conv1d, LayerNorm, Linear, TrainMode, var_contiguous};
use numr::autograd::{Var, var_permute};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Bundled, already-built weights for the full acoustic decoder.
pub struct NeuCodecDecoderWeights<R: Runtime> {
    pub fc: Linear<R>,
    pub embed: Conv1d<R>,
    pub prior_net: Vec<ResnetBlock<R>>,
    pub layers: Vec<TransformerBlock<R>>,
    pub norm: LayerNorm<R>,
    pub post_net: Vec<ResnetBlock<R>>,
    pub head: IstftHead<R>,
}

/// NeuCodec acoustic decoder: FSQ features -> waveform.
pub struct NeuCodecDecoder<R: Runtime> {
    config: NeuCodecDecoderConfig,
    fc: Linear<R>,
    embed: Conv1d<R>,
    prior_net: Vec<ResnetBlock<R>>,
    layers: Vec<TransformerBlock<R>>,
    norm: LayerNorm<R>,
    post_net: Vec<ResnetBlock<R>>,
    head: IstftHead<R>,
}

impl<R: Runtime<DType = DType>> NeuCodecDecoder<R> {
    /// Build the decoder from a validated config and already-built weights.
    pub fn new(config: NeuCodecDecoderConfig, weights: NeuCodecDecoderWeights<R>) -> Result<Self> {
        config.validate()?;

        if weights.prior_net.len() != config.num_prior_resnet_blocks {
            return Err(Error::InvalidArgument {
                arg: "weights.prior_net",
                reason: format!(
                    "expected {} blocks, got {}",
                    config.num_prior_resnet_blocks,
                    weights.prior_net.len()
                ),
            });
        }
        if weights.post_net.len() != config.num_post_resnet_blocks {
            return Err(Error::InvalidArgument {
                arg: "weights.post_net",
                reason: format!(
                    "expected {} blocks, got {}",
                    config.num_post_resnet_blocks,
                    weights.post_net.len()
                ),
            });
        }
        if weights.layers.len() != config.num_transformer_layers {
            return Err(Error::InvalidArgument {
                arg: "weights.layers",
                reason: format!(
                    "expected {} transformer layers, got {}",
                    config.num_transformer_layers,
                    weights.layers.len()
                ),
            });
        }

        Ok(Self {
            config,
            fc: weights.fc,
            embed: weights.embed,
            prior_net: weights.prior_net,
            layers: weights.layers,
            norm: weights.norm,
            post_net: weights.post_net,
            head: weights.head,
        })
    }

    pub fn config(&self) -> &NeuCodecDecoderConfig {
        &self.config
    }

    /// Propagate training/eval mode to every `ResnetBlock` (the only stochastic
    /// layers in this decoder — each holds an upstream `dropout=0.1`).
    ///
    /// Inherent method rather than a [`TrainMode`] impl so it stays available
    /// without importing the trait; a `TrainMode` impl delegates to it.
    pub fn set_training_mode(&mut self, training: bool) {
        for block in self.prior_net.iter_mut().chain(self.post_net.iter_mut()) {
            block.set_training(training);
        }
    }

    /// Whether this decoder's `ResnetBlock` dropouts are active.
    pub fn is_training_mode(&self) -> bool {
        self.prior_net
            .first()
            .or_else(|| self.post_net.first())
            .is_some_and(|b| b.is_training())
    }

    /// Forward through everything up to (and including) the ISTFT head:
    /// `x [B, T, fc_in_dim] -> (mag [B, F, T], phase [B, F, T])`.
    ///
    /// Runtime-generic (no CPU requirement) — the final ISTFT step is
    /// CPU-only and lives in [`NeuCodecDecoder::forward`] (implemented for
    /// `CpuRuntime` only), matching the pattern used by
    /// `crate::model::audio::kokoro::generator::IStftNetGenerator`.
    #[allow(clippy::type_complexity)]
    pub fn forward_features<C>(&self, client: &C, x: &Var<R>) -> Result<(Var<R>, Var<R>)>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let shape = x.shape();
        if shape.len() != 3 || shape[2] != self.config.fc_in_dim {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!(
                    "expected [B, T, {}], got {:?}",
                    self.config.fc_in_dim, shape
                ),
            });
        }

        // fc: [B, T, fc_in_dim] -> [B, T, hidden]
        let h = self.fc.forward(client, x)?;

        // channels-last -> channels-first for embed + prior_net
        let h = var_permute(&h, &[0, 2, 1]).map_err(Error::Numr)?;
        let h = var_contiguous(&h)?;
        let mut h = self.embed.forward(client, &h)?;
        for block in &self.prior_net {
            h = block.forward(client, &h)?;
        }

        // channels-first -> channels-last for the transformer stack
        let h = var_permute(&h, &[0, 2, 1]).map_err(Error::Numr)?;
        let mut h = var_contiguous(&h)?;
        for layer in &self.layers {
            h = layer.forward(client, &h)?;
        }

        // channels-last -> channels-first for post_net
        let h = var_permute(&h, &[0, 2, 1]).map_err(Error::Numr)?;
        let mut h = var_contiguous(&h)?;
        for block in &self.post_net {
            h = block.forward(client, &h)?;
        }

        // channels-first -> channels-last, then the FINAL norm.
        //
        // `norm` runs AFTER `post_net`, not between the transformer stack and
        // `post_net` — upstream `VocosBackbone.forward` is
        // `embed -> prior_net -> transformers -> post_net -> final_layer_norm`.
        // The checkpoint cannot reveal this (it only records that `norm` has a
        // bias); only the source ordering does.
        let h = var_permute(&h, &[0, 2, 1]).map_err(Error::Numr)?;
        let h = var_contiguous(&h)?;
        let h = self.norm.forward(client, &h)?;

        self.head.forward(client, &h)
    }
}

impl<R: Runtime<DType = DType>> TrainMode for NeuCodecDecoder<R> {
    fn set_training(&mut self, training: bool) {
        self.set_training_mode(training);
    }

    fn is_training(&self) -> bool {
        self.is_training_mode()
    }
}

impl NeuCodecDecoder<numr::runtime::cpu::CpuRuntime> {
    /// Full forward: `x [B, T, fc_in_dim] -> waveform [B, T * hop_length]`.
    pub fn forward(
        &self,
        client: &numr::runtime::cpu::CpuClient,
        x: &Var<numr::runtime::cpu::CpuRuntime>,
    ) -> Result<numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>> {
        let (mag, phase) = self.forward_features(client, x)?;
        let window = hann_window(self.config.n_fft, x.tensor().device())?;
        istft(
            client,
            mag.tensor(),
            phase.tensor(),
            &window,
            IStftOptions {
                hop_length: self.config.hop_length,
                // Vocos `padding="same"`, NOT torch's `center=True`: upstream
                // trims `(n_fft - hop)/2 = 720` per end, not `n_fft/2 = 960`.
                // This sets both the output length (`T*hop`, one hop per input
                // frame) and the alignment, so the two are not interchangeable.
                padding: IStftPadding::Same,
                eps: 1e-8,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::neucodec::istft_head::IstftHeadWeights;
    use crate::model::audio::neucodec::resnet_block::ResnetBlockWeights;
    use crate::model::audio::neucodec::transformer_block::TransformerBlockWeights;
    use crate::nn::{GroupNorm, RmsNorm};
    use crate::test_utils::cpu_setup;
    use numr::ops::PaddingMode;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    fn linear(out_f: usize, in_f: usize, val: f32, device: &CpuDevice) -> Linear<CpuRuntime> {
        Linear::new(
            Tensor::<CpuRuntime>::from_slice(&vec![val; out_f * in_f], &[out_f, in_f], device)
                .unwrap(),
            Some(Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; out_f], &[out_f], device).unwrap()),
            false,
        )
    }

    fn layer_norm(c: usize, device: &CpuDevice) -> LayerNorm<CpuRuntime> {
        LayerNorm::new(
            Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; c], &[c], device).unwrap(),
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c], &[c], device).unwrap(),
            1e-5,
            false,
        )
    }

    fn rms_norm(c: usize, device: &CpuDevice) -> RmsNorm<CpuRuntime> {
        RmsNorm::new(
            Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; c], &[c], device).unwrap(),
            1e-6,
            false,
        )
    }

    fn conv(c: usize, k: usize, val: f32, device: &CpuDevice) -> Conv1d<CpuRuntime> {
        let n = c * c * k;
        Conv1d::new(
            Tensor::<CpuRuntime>::from_slice(&vec![val; n], &[c, c, k], device).unwrap(),
            Some(Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c], &[c], device).unwrap()),
            1,
            PaddingMode::Same,
            1,
            1,
            false,
        )
    }

    /// GroupNorm with a test-scale group count (production uses 32; the
    /// synthetic decoder here is only 8 channels wide).
    fn group_norm(c: usize, groups: usize, device: &CpuDevice) -> GroupNorm<CpuRuntime> {
        GroupNorm::new(
            Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; c], &[c], device).unwrap(),
            Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c], &[c], device).unwrap(),
            groups,
            1e-6,
            false,
        )
    }

    fn resnet_block(c: usize, k: usize, val: f32, device: &CpuDevice) -> ResnetBlock<CpuRuntime> {
        ResnetBlock::new(ResnetBlockWeights {
            norm1: group_norm(c, 2, device),
            conv1: conv(c, k, val, device),
            norm2: group_norm(c, 2, device),
            conv2: conv(c, k, val, device),
        })
    }

    fn transformer_block(
        hidden: usize,
        heads: usize,
        head_dim: usize,
        mlp: usize,
        val: f32,
        device: &CpuDevice,
    ) -> TransformerBlock<CpuRuntime> {
        TransformerBlock::new(
            TransformerBlockWeights {
                input_layernorm: rms_norm(hidden, device),
                q_proj: linear(hidden, hidden, val, device),
                k_proj: linear(hidden, hidden, val, device),
                v_proj: linear(hidden, hidden, val, device),
                o_proj: linear(hidden, hidden, val, device),
                post_attention_layernorm: rms_norm(hidden, device),
                mlp_fc1: linear(mlp, hidden, val, device),
                mlp_fc2: linear(hidden, mlp, val, device),
            },
            heads,
            head_dim,
        )
        .unwrap()
    }

    /// A small synthetic decoder: hidden=8, heads=2, head_dim=4, mlp=16,
    /// fc_in_dim=6, n_fft=8 (F=5), hop=3. Weights are tiny nonzero values so
    /// outputs are non-degenerate but numerically small (finite-output check).
    fn make_decoder(
        val: f32,
    ) -> (
        NeuCodecDecoder<CpuRuntime>,
        numr::runtime::cpu::CpuClient,
        CpuDevice,
        NeuCodecDecoderConfig,
    ) {
        let (client, device) = cpu_setup();
        let hidden = 8;
        let heads = 2;
        let head_dim = 4;
        let mlp = 16;
        let fc_in = 6;
        let n_fft = 8;
        // `n_fft - hop` must be EVEN for the `samples == frames * hop` identity
        // to hold exactly (the Vocos trim is `(n_fft - hop) / 2`, floored).
        // The real config satisfies this: 1920 - 480 = 1440.
        let hop = 4;
        let config = NeuCodecDecoderConfig {
            hidden_size: hidden,
            fc_in_dim: fc_in,
            embed_kernel_size: 3,
            resnet_kernel_size: 3,
            num_prior_resnet_blocks: 2,
            num_post_resnet_blocks: 2,
            num_transformer_layers: 2,
            num_heads: heads,
            head_dim,
            mlp_intermediate_size: mlp,
            rms_norm_eps: 1e-6,
            resnet_norm_groups: 2,
            resnet_norm_eps: 1e-6,
            layer_norm_eps: 1e-6,
            n_fft,
            hop_length: hop,
            mag_clamp_max: 1e2,
        };

        let weights = NeuCodecDecoderWeights {
            fc: linear(hidden, fc_in, val, &device),
            embed: conv(hidden, config.embed_kernel_size, val, &device),
            prior_net: (0..config.num_prior_resnet_blocks)
                .map(|_| resnet_block(hidden, config.resnet_kernel_size, val, &device))
                .collect(),
            layers: (0..config.num_transformer_layers)
                .map(|_| transformer_block(hidden, heads, head_dim, mlp, val, &device))
                .collect(),
            norm: layer_norm(hidden, &device),
            post_net: (0..config.num_post_resnet_blocks)
                .map(|_| resnet_block(hidden, config.resnet_kernel_size, val, &device))
                .collect(),
            head: IstftHead::new(
                IstftHeadWeights {
                    linear: linear(config.head_out_dim(), hidden, val, &device),
                },
                n_fft,
                config.mag_clamp_max,
            )
            .unwrap(),
        };

        let decoder = NeuCodecDecoder::new(config, weights).unwrap();
        (decoder, client, device, config)
    }

    #[test]
    fn forward_features_shapes() {
        let (decoder, client, device, config) = make_decoder(0.01);
        let batch = 2;
        let frames = 5;
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.1f32; batch * frames * config.fc_in_dim],
                &[batch, frames, config.fc_in_dim],
                &device,
            )
            .unwrap(),
            false,
        );
        let (mag, phase) = decoder.forward_features(&client, &x).unwrap();
        let f = config.n_freq_bins();
        assert_eq!(mag.shape(), &[batch, f, frames]);
        assert_eq!(phase.shape(), &[batch, f, frames]);
    }

    /// Full decoder: input `[batch, frames, fc_in_dim]` -> waveform
    /// `[batch, samples]`.
    ///
    /// Derivation (see `crate::model::audio::kokoro::istft`): overlap-add
    /// builds `raw_len = (frames-1)*hop + n_fft`, then Vocos `padding="same"`
    /// trims `(n_fft - hop)/2` from each end, leaving
    /// `raw_len - (n_fft - hop) = frames * hop` samples.
    ///
    /// So one input frame yields exactly `hop_length` output samples, which is
    /// what makes the 50 Hz latent rate line up with 24 kHz audio
    /// (`50 * 480 = 24000`). An earlier version of this port used
    /// `torch.istft`-style `center=true` trimming of `n_fft/2` per end and got
    /// `(frames-1)*hop` — one hop short, and misaligned by 240 samples.
    #[test]
    fn forward_waveform_sample_count_matches_vocos_same_padding() {
        let (decoder, client, device, config) = make_decoder(0.01);
        let batch = 2;
        let frames = 7;
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.05f32; batch * frames * config.fc_in_dim],
                &[batch, frames, config.fc_in_dim],
                &device,
            )
            .unwrap(),
            false,
        );
        let waveform = decoder.forward(&client, &x).unwrap();
        let expected_samples = frames * config.hop_length;
        assert_eq!(waveform.shape(), &[batch, expected_samples]);
    }

    #[test]
    fn forward_waveform_is_finite() {
        let (decoder, client, device, config) = make_decoder(0.02);
        let batch = 1;
        let frames = 6;
        let x_data: Vec<f32> = (0..(frames * config.fc_in_dim))
            .map(|i| (i as f32 * 0.017).sin())
            .collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_data, &[batch, frames, config.fc_in_dim], &device)
                .unwrap(),
            false,
        );
        let waveform = decoder.forward(&client, &x).unwrap();
        for v in waveform.contiguous().unwrap().to_vec::<f32>() {
            assert!(v.is_finite(), "waveform sample is not finite: {v}");
        }
    }

    #[test]
    fn new_rejects_wrong_prior_net_block_count() {
        let (_decoder, _client, device, config) = make_decoder(0.01);
        let hidden = config.hidden_size;
        let weights = NeuCodecDecoderWeights {
            fc: linear(hidden, config.fc_in_dim, 0.01, &device),
            embed: conv(hidden, config.embed_kernel_size, 0.01, &device),
            prior_net: vec![resnet_block(
                hidden,
                config.resnet_kernel_size,
                0.01,
                &device,
            )], // wrong count (1 vs 2)
            layers: (0..config.num_transformer_layers)
                .map(|_| {
                    transformer_block(
                        hidden,
                        config.num_heads,
                        config.head_dim,
                        config.mlp_intermediate_size,
                        0.01,
                        &device,
                    )
                })
                .collect(),
            norm: layer_norm(hidden, &device),
            post_net: (0..config.num_post_resnet_blocks)
                .map(|_| resnet_block(hidden, config.resnet_kernel_size, 0.01, &device))
                .collect(),
            head: IstftHead::new(
                IstftHeadWeights {
                    linear: linear(config.head_out_dim(), hidden, 0.01, &device),
                },
                config.n_fft,
                config.mag_clamp_max,
            )
            .unwrap(),
        };
        assert!(NeuCodecDecoder::new(config, weights).is_err());
    }

    #[test]
    fn forward_features_rejects_wrong_input_width() {
        let (decoder, client, device, config) = make_decoder(0.01);
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.0f32; 4 * (config.fc_in_dim + 1)],
                &[1, 4, config.fc_in_dim + 1],
                &device,
            )
            .unwrap(),
            false,
        );
        assert!(decoder.forward_features(&client, &x).is_err());
    }
}
