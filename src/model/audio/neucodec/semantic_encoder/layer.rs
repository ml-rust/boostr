//! `Wav2Vec2BertEncoderLayer` — one conformer layer of NeuCodec's semantic
//! branch, in the "macaron" arrangement: two half-weighted feed-forward
//! modules sandwiching self-attention and a convolution module.
//!
//! ```text
//! residual = x;  x = residual + 0.5 * ffn1(ffn1_layer_norm(x))
//! residual = x;  x = residual + self_attn(self_attn_layer_norm(x))
//! residual = x;  x = residual + conv_module(x)          # NO external pre-norm
//! residual = x;  x = residual + 0.5 * ffn2(ffn2_layer_norm(x))
//! x = final_layer_norm(x)                               # trailing POST-norm
//! ```
//!
//! ## The 0.5 scales the FFN OUTPUT, and only for the two FFNs
//!
//! `x = residual + 0.5 * h`, not `0.5 * (residual + h)` and not
//! `0.5 * residual + h`. Halving the residual too shrinks the whole stream
//! geometrically across 16 layers; halving nothing doubles the FFN
//! contribution. Both alternatives are shape-identical and both are wrong.
//!
//! The factor belongs to the macaron FFNs alone. Applying it to the attention
//! or convolution residual — the tidy-looking "scale every branch the same
//! way" reading — is not what the weights were trained with.
//!
//! ## The convolution branch takes the RAW residual, not a normalized one
//!
//! Every other sub-block here is pre-normed by the layer. The convolution
//! module is not: it owns an internal `layer_norm` as its first operation, so
//! adding a `conv_layer_norm` in the layer would normalize twice. There is no
//! such tensor in the checkpoint — the only give-away, since a second norm
//! would run fine on any weights you handed it.
//!
//! ## `final_layer_norm` is a trailing POST-norm, not a pre-norm
//!
//! It runs after the last residual add, so the value handed to the next layer
//! is already normalized. Reading it as the next block's pre-norm (the usual
//! pre-norm-transformer habit) would drop it from the last layer's output —
//! and NeuCodec consumes exactly that output.

use crate::error::{Error, Result};
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::model::audio::neucodec::semantic_encoder::attention::SemanticSelfAttention;
use crate::model::audio::neucodec::semantic_encoder::conv_module::ConvolutionModule;
use crate::nn::{LayerNorm, Linear};
use numr::autograd::{Var, var_add, var_mul_scalar, var_silu};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Half-weight applied to each macaron feed-forward module's output.
pub const FFN_RESIDUAL_SCALE: f64 = 0.5;

/// A `Wav2Vec2BertFeedForward`: `output_dense(silu(intermediate_dense(x)))`.
///
/// The activation is `swish`, which is SiLU — `x * sigmoid(x)` — not the
/// ReLU/GELU a "standard transformer FFN" reading would supply.
pub struct SemanticFeedForward<R: Runtime> {
    intermediate_dense: Linear<R>,
    output_dense: Linear<R>,
}

impl<R: Runtime> SemanticFeedForward<R> {
    /// Assemble from already-loaded projections.
    pub fn new(intermediate_dense: Linear<R>, output_dense: Linear<R>) -> Self {
        Self {
            intermediate_dense,
            output_dense,
        }
    }
}

impl<R: Runtime<DType = DType>> SemanticFeedForward<R> {
    /// Forward: `[B, T, hidden] -> [B, T, hidden]`.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let h = self.intermediate_dense.forward(client, x)?;
        let h = var_silu(&h, client).map_err(Error::Numr)?;
        self.output_dense.forward(client, &h)
    }
}

/// Already-built weights for one [`SemanticEncoderLayer`].
pub struct SemanticEncoderLayerWeights<R: Runtime> {
    pub ffn1_layer_norm: LayerNorm<R>,
    pub ffn1: SemanticFeedForward<R>,
    pub self_attn_layer_norm: LayerNorm<R>,
    pub self_attn: SemanticSelfAttention<R>,
    pub conv_module: ConvolutionModule<R>,
    pub ffn2_layer_norm: LayerNorm<R>,
    pub ffn2: SemanticFeedForward<R>,
    pub final_layer_norm: LayerNorm<R>,
}

/// One conformer layer: FFN/2 -> attention -> convolution -> FFN/2 -> post-norm.
pub struct SemanticEncoderLayer<R: Runtime> {
    ffn1_layer_norm: LayerNorm<R>,
    ffn1: SemanticFeedForward<R>,
    self_attn_layer_norm: LayerNorm<R>,
    self_attn: SemanticSelfAttention<R>,
    conv_module: ConvolutionModule<R>,
    ffn2_layer_norm: LayerNorm<R>,
    ffn2: SemanticFeedForward<R>,
    final_layer_norm: LayerNorm<R>,
}

impl<R: Runtime> SemanticEncoderLayer<R> {
    /// Assemble from already-loaded sub-modules.
    pub fn new(weights: SemanticEncoderLayerWeights<R>) -> Self {
        Self {
            ffn1_layer_norm: weights.ffn1_layer_norm,
            ffn1: weights.ffn1,
            self_attn_layer_norm: weights.self_attn_layer_norm,
            self_attn: weights.self_attn,
            conv_module: weights.conv_module,
            ffn2_layer_norm: weights.ffn2_layer_norm,
            ffn2: weights.ffn2,
            final_layer_norm: weights.final_layer_norm,
        }
    }
}

impl<R: Runtime<DType = DType>> SemanticEncoderLayer<R> {
    /// Forward: `x [B, T, hidden] -> [B, T, hidden]`.
    ///
    /// The attention sub-block builds its own relative-distance index table
    /// for this call. Callers driving multiple layers over the same
    /// `seq_len` (e.g.
    /// [`crate::model::audio::neucodec::semantic_encoder::encoder::SemanticEncoder`])
    /// should build the table once and call [`Self::forward_with_indices`]
    /// on every layer instead, to avoid rebuilding and re-uploading a
    /// bit-identical table per layer.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        // 1. Macaron FFN, half-scaled output onto the UNSCALED residual.
        let h = self.ffn1_layer_norm.forward(client, x)?;
        let h = self.ffn1.forward(client, &h)?;
        let h = var_mul_scalar(&h, FFN_RESIDUAL_SCALE, client).map_err(Error::Numr)?;
        let x = var_add(x, &h, client).map_err(Error::Numr)?;

        // 2. Self-attention, full residual.
        let h = self.self_attn_layer_norm.forward(client, &x)?;
        let h = self.self_attn.forward(client, &h)?;
        let x = var_add(&x, &h, client).map_err(Error::Numr)?;

        // 3. Convolution on the RAW residual — the module norms internally.
        let h = self.conv_module.forward(client, &x)?;
        let x = var_add(&x, &h, client).map_err(Error::Numr)?;

        // 4. Second macaron FFN, half-scaled again.
        let h = self.ffn2_layer_norm.forward(client, &x)?;
        let h = self.ffn2.forward(client, &h)?;
        let h = var_mul_scalar(&h, FFN_RESIDUAL_SCALE, client).map_err(Error::Numr)?;
        let x = var_add(&x, &h, client).map_err(Error::Numr)?;

        // 5. Trailing post-norm.
        self.final_layer_norm.forward(client, &x)
    }

    /// Forward with a caller-supplied relative-distance index tensor,
    /// threaded straight through to the attention sub-block.
    ///
    /// `index_tensor` must be the table built by
    /// [`crate::model::audio::neucodec::semantic_encoder::attention::relative_distance_index_tensor`]
    /// for this call's `seq_len` and this layer's attention window — see
    /// [`SemanticSelfAttention::forward_with_indices`].
    pub fn forward_with_indices<C>(
        &self,
        client: &C,
        x: &Var<R>,
        index_tensor: &Tensor<R>,
    ) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        // 1. Macaron FFN, half-scaled output onto the UNSCALED residual.
        let h = self.ffn1_layer_norm.forward(client, x)?;
        let h = self.ffn1.forward(client, &h)?;
        let h = var_mul_scalar(&h, FFN_RESIDUAL_SCALE, client).map_err(Error::Numr)?;
        let x = var_add(x, &h, client).map_err(Error::Numr)?;

        // 2. Self-attention, full residual.
        let h = self.self_attn_layer_norm.forward(client, &x)?;
        let h = self
            .self_attn
            .forward_with_indices(client, &h, index_tensor)?;
        let x = var_add(&x, &h, client).map_err(Error::Numr)?;

        // 3. Convolution on the RAW residual — the module norms internally.
        let h = self.conv_module.forward(client, &x)?;
        let x = var_add(&x, &h, client).map_err(Error::Numr)?;

        // 4. Second macaron FFN, half-scaled again.
        let h = self.ffn2_layer_norm.forward(client, &x)?;
        let h = self.ffn2.forward(client, &h)?;
        let h = var_mul_scalar(&h, FFN_RESIDUAL_SCALE, client).map_err(Error::Numr)?;
        let x = var_add(&x, &h, client).map_err(Error::Numr)?;

        // 5. Trailing post-norm.
        self.final_layer_norm.forward(client, &x)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::model::audio::neucodec::semantic_encoder::attention::SemanticSelfAttentionWeights;
    use crate::model::audio::neucodec::semantic_encoder::config::SemanticEncoderConfig;
    use crate::model::audio::neucodec::semantic_encoder::conv_module::{
        ConvolutionModuleWeights, causal_padding,
    };
    use crate::nn::{Conv1d, Embedding};
    use crate::test_utils::cpu_setup;
    use numr::ops::PaddingMode;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    pub(crate) fn test_config() -> SemanticEncoderConfig {
        SemanticEncoderConfig {
            hidden_size: 8,
            num_heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            num_layers: 2,
            conv_depthwise_kernel_size: 5,
            feature_projection_input_dim: 6,
            left_max_position_embeddings: 3,
            right_max_position_embeddings: 1,
            ..Default::default()
        }
    }

    pub(crate) fn linear(out_f: usize, in_f: usize, device: &CpuDevice) -> Linear<CpuRuntime> {
        Linear::new(
            Tensor::<CpuRuntime>::try_from_slice(
                &vec![0.02f32; out_f * in_f],
                &[out_f, in_f],
                device,
            )
            .unwrap(),
            Some(
                Tensor::<CpuRuntime>::try_from_slice(&vec![0.0f32; out_f], &[out_f], device)
                    .unwrap(),
            ),
            false,
        )
    }

    pub(crate) fn layer_norm(dim: usize, device: &CpuDevice) -> LayerNorm<CpuRuntime> {
        LayerNorm::new(
            Tensor::<CpuRuntime>::try_from_slice(&vec![1.0f32; dim], &[dim], device).unwrap(),
            Tensor::<CpuRuntime>::try_from_slice(&vec![0.0f32; dim], &[dim], device).unwrap(),
            1e-5,
            false,
        )
    }

    fn conv(
        out_ch: usize,
        in_ch: usize,
        kernel: usize,
        groups: usize,
        padding: PaddingMode,
        device: &CpuDevice,
    ) -> Conv1d<CpuRuntime> {
        Conv1d::new(
            Tensor::<CpuRuntime>::try_from_slice(
                &vec![0.05f32; out_ch * in_ch * kernel],
                &[out_ch, in_ch, kernel],
                device,
            )
            .unwrap(),
            None,
            1,
            padding,
            1,
            groups,
            false,
        )
    }

    pub(crate) fn test_layer(
        cfg: SemanticEncoderConfig,
        device: &CpuDevice,
    ) -> SemanticEncoderLayer<CpuRuntime> {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let k = cfg.conv_depthwise_kernel_size;
        let rows = cfg.distance_embedding_len();
        let table: Vec<f32> = (0..(rows * cfg.head_dim))
            .map(|i| (i as f32 * 0.017).sin() * 0.1)
            .collect();

        SemanticEncoderLayer::new(SemanticEncoderLayerWeights {
            ffn1_layer_norm: layer_norm(hidden, device),
            ffn1: SemanticFeedForward::new(
                linear(inter, hidden, device),
                linear(hidden, inter, device),
            ),
            self_attn_layer_norm: layer_norm(hidden, device),
            self_attn: SemanticSelfAttention::new(
                SemanticSelfAttentionWeights {
                    linear_q: linear(hidden, hidden, device),
                    linear_k: linear(hidden, hidden, device),
                    linear_v: linear(hidden, hidden, device),
                    linear_out: linear(hidden, hidden, device),
                    distance_embedding: Embedding::new(
                        Tensor::<CpuRuntime>::try_from_slice(&table, &[rows, cfg.head_dim], device)
                            .unwrap(),
                        false,
                    ),
                },
                cfg,
            )
            .expect("attention"),
            conv_module: ConvolutionModule::new(
                ConvolutionModuleWeights {
                    layer_norm: layer_norm(hidden, device),
                    pointwise_conv1: conv(
                        2 * hidden,
                        hidden,
                        1,
                        1,
                        PaddingMode::conv1d(0, 0),
                        device,
                    ),
                    depthwise_conv: conv(hidden, 1, k, hidden, causal_padding(k), device),
                    depthwise_layer_norm: layer_norm(hidden, device),
                    pointwise_conv2: conv(hidden, hidden, 1, 1, PaddingMode::conv1d(0, 0), device),
                },
                hidden,
            )
            .expect("conv module"),
            ffn2_layer_norm: layer_norm(hidden, device),
            ffn2: SemanticFeedForward::new(
                linear(inter, hidden, device),
                linear(hidden, inter, device),
            ),
            final_layer_norm: layer_norm(hidden, device),
        })
    }

    #[test]
    fn forward_preserves_shape() {
        let (client, device) = cpu_setup();
        let cfg = test_config();
        let layer = test_layer(cfg, &device);

        let t = 9;
        let data: Vec<f32> = (0..(t * cfg.hidden_size))
            .map(|i| (i as f32 * 0.05).sin())
            .collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&data, &[1, t, cfg.hidden_size], &device).unwrap(),
            false,
        );
        let y = layer.forward(&client, &x).expect("forward");
        assert_eq!(y.shape(), &[1, t, cfg.hidden_size]);
        for v in y.tensor().contiguous().expect("contiguous").to_vec::<f32>() {
            assert!(v.is_finite(), "layer output is not finite: {v}");
        }
    }

    #[test]
    fn rejects_wrong_hidden_width() {
        let (client, device) = cpu_setup();
        let layer = test_layer(test_config(), &device);
        let x = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[0.0f32; 4 * 3], &[1, 4, 3], &device).unwrap(),
            false,
        );
        assert!(layer.forward(&client, &x).is_err());
    }
}
