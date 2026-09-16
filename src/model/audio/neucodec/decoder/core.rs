use super::{NeuCodecDecoder, NeuCodecDecoderWeights};
use crate::error::{Error, Result};
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::model::audio::neucodec::config::NeuCodecDecoderConfig;
use crate::nn::{TrainMode, var_contiguous};
use numr::autograd::{Var, var_permute};
use numr::dtype::DType;
use numr::runtime::Runtime;

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
    /// layers in this decoder — each holds a `dropout=0.1` in the reference
    /// implementation).
    ///
    /// Inherent method rather than a [`crate::nn::TrainMode`] impl so it stays available
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
    /// CPU-only and lives in [`NeuCodecDecoder::forward`](super::NeuCodecDecoder::forward) (implemented for
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
        // `post_net` — the reference implementation's `VocosBackbone.forward` is
        // `embed -> prior_net -> transformers -> post_net -> final_layer_norm`.
        // The checkpoint cannot reveal this (it only records that `norm` has a
        // bias); only the source ordering does.
        let h = var_permute(&h, &[0, 2, 1]).map_err(Error::Numr)?;
        let h = var_contiguous(&h)?;
        let h = self.norm.forward(client, &h)?;

        self.head.forward(client, &h)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;
    use crate::model::audio::neucodec::istft_head::{IstftHead, IstftHeadWeights};
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

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
