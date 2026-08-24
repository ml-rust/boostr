//! `SemanticEncoder` — NeuCodec's Wav2Vec2-BERT semantic branch end to end.
//!
//! ```text
//! stacked filterbank features [B, T, 160]
//!   -> FeatureProjection            -> [B, T, 1024]
//!   -> 16 x SemanticEncoderLayer    -> [B, T, 1024]
//! ```
//!
//! The value NeuCodec consumes is the output AFTER the final layer — each
//! layer already ends in its own `final_layer_norm`, so there is no extra
//! encoder-level norm to apply on top (and the checkpoint has no tensor for
//! one).
//!
//! Layerdrop is absent by construction: it is a training-time regularizer and
//! an exact no-op at inference, so every layer always runs, in order.

use crate::error::{Error, Result};
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::model::audio::neucodec::semantic_encoder::attention::relative_distance_index_tensor;
use crate::model::audio::neucodec::semantic_encoder::config::SemanticEncoderConfig;
use crate::model::audio::neucodec::semantic_encoder::feature_projection::FeatureProjection;
use crate::model::audio::neucodec::semantic_encoder::layer::SemanticEncoderLayer;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Already-built parts of a [`SemanticEncoder`].
pub struct SemanticEncoderWeights<R: Runtime> {
    pub feature_projection: FeatureProjection<R>,
    pub layers: Vec<SemanticEncoderLayer<R>>,
}

/// The 16-layer conformer encoder plus its feature projection.
pub struct SemanticEncoder<R: Runtime> {
    feature_projection: FeatureProjection<R>,
    layers: Vec<SemanticEncoderLayer<R>>,
    config: SemanticEncoderConfig,
}

impl<R: Runtime> SemanticEncoder<R> {
    /// Assemble from already-loaded parts.
    ///
    /// Errors if the layer count disagrees with `config`, so a partially
    /// loaded checkpoint cannot silently produce a shallower encoder.
    pub fn new(weights: SemanticEncoderWeights<R>, config: SemanticEncoderConfig) -> Result<Self> {
        config.validate()?;
        if weights.layers.len() != config.num_layers {
            return Err(Error::ModelError {
                reason: format!(
                    "expected {} encoder layers, got {}",
                    config.num_layers,
                    weights.layers.len()
                ),
            });
        }
        Ok(Self {
            feature_projection: weights.feature_projection,
            layers: weights.layers,
            config,
        })
    }

    /// The geometry this encoder was built with.
    pub fn config(&self) -> &SemanticEncoderConfig {
        &self.config
    }

    /// Number of conformer layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// The feature projection, for callers that want the pre-conformer latent.
    pub fn feature_projection(&self) -> &FeatureProjection<R> {
        &self.feature_projection
    }

    /// The conformer layers in order, for callers that want intermediate
    /// hidden states rather than only the final one.
    pub fn layers(&self) -> &[SemanticEncoderLayer<R>] {
        &self.layers
    }
}

impl<R: Runtime<DType = DType>> SemanticEncoder<R> {
    /// Forward: `x [B, T, 160] -> [B, T, 1024]`, length-preserving.
    ///
    /// Every layer shares the same `seq_len` and the same
    /// `left_max_position_embeddings`/`right_max_position_embeddings` window
    /// (both come from the shared [`SemanticEncoderConfig`]), so the
    /// relative-distance index table is bit-identical across layers. It is
    /// built once here, right after the feature projection fixes `seq_len`,
    /// and reused by every layer via
    /// [`SemanticEncoderLayer::forward_with_indices`] instead of each layer
    /// rebuilding (and re-uploading, on device backends) its own copy.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let hidden = self.feature_projection.forward(client, x)?;
        let seq_len = hidden
            .shape()
            .get(1)
            .copied()
            .ok_or_else(|| Error::InvalidArgument {
                arg: "x",
                reason: format!(
                    "expected feature projection output with a sequence axis, got {:?}",
                    hidden.shape()
                ),
            })?;
        let index_tensor = relative_distance_index_tensor::<R>(
            seq_len,
            self.config.left_max_position_embeddings,
            self.config.right_max_position_embeddings,
            hidden.tensor().device(),
        )?;
        let mut hidden = hidden;
        for layer in &self.layers {
            hidden = layer.forward_with_indices(client, &hidden, &index_tensor)?;
        }
        Ok(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::neucodec::semantic_encoder::feature_projection::FeatureProjectionWeights;
    use crate::model::audio::neucodec::semantic_encoder::layer::tests::{
        layer_norm, linear, test_config, test_layer,
    };
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    fn encoder(cfg: SemanticEncoderConfig, device: &CpuDevice) -> SemanticEncoder<CpuRuntime> {
        let in_dim = cfg.feature_projection_input_dim;
        let feature_projection = FeatureProjection::new(
            FeatureProjectionWeights {
                layer_norm: layer_norm(in_dim, device),
                projection: linear(cfg.hidden_size, in_dim, device),
            },
            in_dim,
        )
        .expect("feature projection");

        let layers = (0..cfg.num_layers)
            .map(|_| test_layer(cfg, device))
            .collect();

        SemanticEncoder::new(
            SemanticEncoderWeights {
                feature_projection,
                layers,
            },
            cfg,
        )
        .expect("encoder")
    }

    #[test]
    fn forward_projects_and_preserves_length() {
        let (client, device) = cpu_setup();
        let cfg = test_config();
        let enc = encoder(cfg, &device);
        assert_eq!(enc.num_layers(), cfg.num_layers);

        let t = 7;
        let in_dim = cfg.feature_projection_input_dim;
        let data: Vec<f32> = (0..(t * in_dim))
            .map(|i| (i as f32 * 0.031).cos())
            .collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&data, &[1, t, in_dim], &device).unwrap(),
            false,
        );

        let y = enc.forward(&client, &x).expect("forward");
        assert_eq!(y.shape(), &[1, t, cfg.hidden_size]);
        for v in y.tensor().contiguous().expect("contiguous").to_vec::<f32>() {
            assert!(v.is_finite(), "encoder output is not finite: {v}");
        }
    }

    #[test]
    fn rejects_layer_count_mismatch() {
        let (_client, device) = cpu_setup();
        let cfg = test_config();
        let in_dim = cfg.feature_projection_input_dim;
        let feature_projection = FeatureProjection::new(
            FeatureProjectionWeights {
                layer_norm: layer_norm(in_dim, &device),
                projection: linear(cfg.hidden_size, in_dim, &device),
            },
            in_dim,
        )
        .expect("feature projection");

        let weights = SemanticEncoderWeights {
            feature_projection,
            layers: vec![test_layer(cfg, &device)], // cfg.num_layers is 2
        };
        assert!(SemanticEncoder::new(weights, cfg).is_err());
    }

    #[test]
    fn rejects_wrong_feature_width() {
        let (client, device) = cpu_setup();
        let cfg = test_config();
        let enc = encoder(cfg, &device);
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 5 * 3], &[1, 5, 3], &device).unwrap(),
            false,
        );
        assert!(enc.forward(&client, &x).is_err());
    }
}
