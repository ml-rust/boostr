//! Loads `semantic_encoder.*` and assembles a [`SemanticEncoder`].

use super::support::checked_tensor;
use crate::error::Result;
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::model::audio::neucodec::semantic_encoder::attention::{
    SemanticSelfAttention, SemanticSelfAttentionWeights,
};
use crate::model::audio::neucodec::semantic_encoder::config::SemanticEncoderConfig;
use crate::model::audio::neucodec::semantic_encoder::conv_module::{
    ConvolutionModule, ConvolutionModuleWeights, causal_padding,
};
use crate::model::audio::neucodec::semantic_encoder::encoder::{
    SemanticEncoder, SemanticEncoderWeights,
};
use crate::model::audio::neucodec::semantic_encoder::feature_projection::{
    FeatureProjection, FeatureProjectionWeights,
};
use crate::model::audio::neucodec::semantic_encoder::layer::{
    SemanticEncoderLayer, SemanticEncoderLayerWeights, SemanticFeedForward,
};
use crate::nn::{Conv1d, Embedding, LayerNorm, Linear};
use numr::dtype::DType;
use numr::ops::PaddingMode;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::path::Path;

/// Top-level prefix for the Wav2Vec2-BERT semantic branch.
pub const DEFAULT_SEMANTIC_ENCODER_PREFIX: &str = "semantic_encoder";

/// Reads `semantic_encoder.*` and assembles a [`SemanticEncoder`].
///
/// Checkpoint layout (all shapes checked; a mismatch fails the load):
///
/// ```text
/// feature_projection.layer_norm.{weight,bias}          [160]
/// feature_projection.projection.{weight,bias}          [1024,160] / [1024]
/// encoder.layers.{0..15}.
///   ffn1_layer_norm.{weight,bias}                      [1024]
///   ffn1.intermediate_dense.{weight,bias}              [4096,1024] / [4096]
///   ffn1.output_dense.{weight,bias}                    [1024,4096] / [1024]
///   self_attn_layer_norm.{weight,bias}                 [1024]
///   self_attn.linear_{q,k,v,out}.{weight,bias}         [1024,1024] / [1024]
///   self_attn.distance_embedding.weight                [73,64]
///   conv_module.layer_norm.{weight,bias}               [1024]
///   conv_module.pointwise_conv1.weight                 [2048,1024,1]   (no bias)
///   conv_module.depthwise_conv.weight                  [1024,1,31]     (no bias)
///   conv_module.depthwise_layer_norm.{weight,bias}     [1024]
///   conv_module.pointwise_conv2.weight                 [1024,1024,1]   (no bias)
///   ffn2_layer_norm.{weight,bias}                      [1024]
///   ffn2.intermediate_dense.{weight,bias}              [4096,1024] / [4096]
///   ffn2.output_dense.{weight,bias}                    [1024,4096] / [1024]
///   final_layer_norm.{weight,bias}                     [1024]
/// ```
///
/// `masked_spec_embed` is present in the checkpoint but is a training-only
/// masking token; inference never reads it, so it is deliberately not loaded.
struct SemanticEncoderLoader<'a, R: Runtime<DType = DType>> {
    loader: &'a mut SafeTensorsLoader,
    device: &'a R::Device,
    prefix: String,
    config: SemanticEncoderConfig,
}

impl<R: Runtime<DType = DType>> SemanticEncoderLoader<'_, R> {
    fn tensor(&mut self, name: &str, expected: &[usize]) -> Result<Tensor<R>> {
        checked_tensor::<R>(self.loader, self.device, &self.prefix, name, expected)
    }

    fn layer_norm(&mut self, name: &str, dim: usize) -> Result<LayerNorm<R>> {
        let weight = self.tensor(&format!("{name}.weight"), &[dim])?;
        let bias = self.tensor(&format!("{name}.bias"), &[dim])?;
        Ok(LayerNorm::new(
            weight,
            bias,
            self.config.layer_norm_eps,
            false,
        ))
    }

    fn linear(&mut self, name: &str, out_f: usize, in_f: usize) -> Result<Linear<R>> {
        let weight = self.tensor(&format!("{name}.weight"), &[out_f, in_f])?;
        let bias = self.tensor(&format!("{name}.bias"), &[out_f])?;
        Ok(Linear::new(weight, Some(bias), false))
    }

    /// Bias-free Conv1d — every conv in the convolution module is `bias=False`.
    fn conv(
        &mut self,
        name: &str,
        out_ch: usize,
        in_ch: usize,
        kernel: usize,
        groups: usize,
        padding: PaddingMode,
    ) -> Result<Conv1d<R>> {
        let weight = self.tensor(&format!("{name}.weight"), &[out_ch, in_ch, kernel])?;
        Ok(Conv1d::new(weight, None, 1, padding, 1, groups, false))
    }

    fn feed_forward(&mut self, name: &str) -> Result<SemanticFeedForward<R>> {
        let hidden = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        Ok(SemanticFeedForward::new(
            self.linear(&format!("{name}.intermediate_dense"), inter, hidden)?,
            self.linear(&format!("{name}.output_dense"), hidden, inter)?,
        ))
    }

    fn self_attn(&mut self, name: &str) -> Result<SemanticSelfAttention<R>> {
        let hidden = self.config.hidden_size;
        let rows = self.config.distance_embedding_len();
        let head_dim = self.config.head_dim;
        let weights = SemanticSelfAttentionWeights {
            linear_q: self.linear(&format!("{name}.linear_q"), hidden, hidden)?,
            linear_k: self.linear(&format!("{name}.linear_k"), hidden, hidden)?,
            linear_v: self.linear(&format!("{name}.linear_v"), hidden, hidden)?,
            linear_out: self.linear(&format!("{name}.linear_out"), hidden, hidden)?,
            distance_embedding: Embedding::new(
                self.tensor(
                    &format!("{name}.distance_embedding.weight"),
                    &[rows, head_dim],
                )?,
                false,
            ),
        };
        SemanticSelfAttention::new(weights, self.config)
    }

    fn conv_module(&mut self, name: &str) -> Result<ConvolutionModule<R>> {
        let hidden = self.config.hidden_size;
        let kernel = self.config.conv_depthwise_kernel_size;
        // Pointwise convs are k=1, so they need no padding at all.
        let no_padding = PaddingMode::conv1d(0, 0);
        let weights = ConvolutionModuleWeights {
            layer_norm: self.layer_norm(&format!("{name}.layer_norm"), hidden)?,
            pointwise_conv1: self.conv(
                &format!("{name}.pointwise_conv1"),
                2 * hidden,
                hidden,
                1,
                1,
                no_padding,
            )?,
            // Depthwise: one filter per channel, causal left-only padding.
            depthwise_conv: self.conv(
                &format!("{name}.depthwise_conv"),
                hidden,
                1,
                kernel,
                hidden,
                causal_padding(kernel),
            )?,
            depthwise_layer_norm: self
                .layer_norm(&format!("{name}.depthwise_layer_norm"), hidden)?,
            pointwise_conv2: self.conv(
                &format!("{name}.pointwise_conv2"),
                hidden,
                hidden,
                1,
                1,
                no_padding,
            )?,
        };
        ConvolutionModule::new(weights, hidden)
    }

    fn layer(&mut self, idx: usize) -> Result<SemanticEncoderLayer<R>> {
        let hidden = self.config.hidden_size;
        let p = format!("encoder.layers.{idx}");
        Ok(SemanticEncoderLayer::new(SemanticEncoderLayerWeights {
            ffn1_layer_norm: self.layer_norm(&format!("{p}.ffn1_layer_norm"), hidden)?,
            ffn1: self.feed_forward(&format!("{p}.ffn1"))?,
            self_attn_layer_norm: self.layer_norm(&format!("{p}.self_attn_layer_norm"), hidden)?,
            self_attn: self.self_attn(&format!("{p}.self_attn"))?,
            conv_module: self.conv_module(&format!("{p}.conv_module"))?,
            ffn2_layer_norm: self.layer_norm(&format!("{p}.ffn2_layer_norm"), hidden)?,
            ffn2: self.feed_forward(&format!("{p}.ffn2"))?,
            final_layer_norm: self.layer_norm(&format!("{p}.final_layer_norm"), hidden)?,
        }))
    }

    fn build(&mut self) -> Result<SemanticEncoderWeights<R>> {
        let in_dim = self.config.feature_projection_input_dim;
        let hidden = self.config.hidden_size;

        // The norm is over the 160-wide INPUT, before the projection.
        let feature_projection = FeatureProjection::new(
            FeatureProjectionWeights {
                layer_norm: self.layer_norm("feature_projection.layer_norm", in_dim)?,
                projection: self.linear("feature_projection.projection", hidden, in_dim)?,
            },
            in_dim,
        )?;

        let mut layers = Vec::with_capacity(self.config.num_layers);
        for i in 0..self.config.num_layers {
            layers.push(self.layer(i)?);
        }

        Ok(SemanticEncoderWeights {
            feature_projection,
            layers,
        })
    }
}

/// Load the Wav2Vec2-BERT semantic encoder from a `neuphonic/neucodec`
/// checkpoint, using the verified default geometry.
pub fn load_semantic_encoder<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    device: &R::Device,
) -> Result<SemanticEncoder<R>> {
    load_semantic_encoder_with(
        path,
        SemanticEncoderConfig::default(),
        DEFAULT_SEMANTIC_ENCODER_PREFIX,
        device,
    )
}

/// Load the semantic encoder with an explicit config and checkpoint prefix.
///
/// Every tensor is shape-checked against `config`, so a config that disagrees
/// with the checkpoint fails at the first mismatched tensor rather than
/// silently building a wrong model.
pub fn load_semantic_encoder_with<R: Runtime<DType = DType>, P: AsRef<Path>>(
    path: P,
    config: SemanticEncoderConfig,
    prefix: &str,
    device: &R::Device,
) -> Result<SemanticEncoder<R>> {
    config.validate()?;
    let mut loader = SafeTensorsLoader::open(path)?;
    let weights = SemanticEncoderLoader::<R> {
        loader: &mut loader,
        device,
        prefix: prefix.to_string(),
        config,
    }
    .build()?;
    SemanticEncoder::new(weights, config)
}
