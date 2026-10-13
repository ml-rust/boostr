//! Full Whisper decoder: embeddings → N layers → final LN → tied logits head.
//! Holds the decoder state, its loader, and the KV-cached forward.

use super::cache::{DecoderCache, DecoderLayerCache};
use super::capture::CrossAttentionCapture;
use super::helpers::load_layernorm;
use super::layer::WhisperDecoderLayer;
use crate::error::{Error, Result};
use crate::model::config::AudioConfig;
use crate::nn::{Embedding, LayerNorm, VarBuilder};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConditionalOps, MatmulOps, NormalizationOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Whisper decoder: token+positional embeddings → N layers with cross-attn → final LN → tied `proj_out`.
pub struct WhisperDecoder<R: Runtime> {
    pub(super) embed_tokens: Embedding<R>,
    /// Shape `[max_target_positions, hidden]`, added to token embeddings at the
    /// current decoding position(s).
    pub(super) embed_positions: Tensor<R>,
    pub(super) layers: Vec<WhisperDecoderLayer<R>>,
    pub(super) layer_norm: LayerNorm<R>,
    /// Tied to `embed_tokens.weight` (shape `[vocab, hidden]`). We keep an owned
    /// handle here and use matmul with transposed embedding matrix as the logits head.
    pub(super) tied_out_weight: Tensor<R>,
    vocab_size: usize,
    pub(super) hidden_size: usize,
}

impl<R: Runtime<DType = DType>> WhisperDecoder<R> {
    /// Load a Whisper decoder from a VarBuilder rooted at the decoder's prefix
    /// (typically `vb.pp("model").pp("decoder")` for HF checkpoints).
    ///
    /// `proj_out.weight` is assumed tied to `embed_tokens.weight`.
    pub fn from_varbuilder(vb: &mut VarBuilder<'_, R>, config: &AudioConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        let num_heads = config.num_heads;
        let num_layers = config.decoder_layer_count();

        let mut tok_vb = vb.pp("embed_tokens");
        let tok_weight = tok_vb.take_tensor("weight")?;
        drop(tok_vb);
        let tied_out_weight = tok_weight.clone();
        let embed_tokens = Embedding::new(tok_weight, false);

        let mut pos_vb = vb.pp("embed_positions");
        let embed_positions = pos_vb.take_tensor("weight")?;
        drop(pos_vb);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let mut layer_vb = vb.pp(&format!("layers.{i}"));
            layers.push(WhisperDecoderLayer::from_varbuilder(
                &mut layer_vb,
                hidden,
                num_heads,
            )?);
        }

        let layer_norm = load_layernorm(vb, "layer_norm")?;

        Ok(Self {
            embed_tokens,
            embed_positions,
            layers,
            layer_norm,
            tied_out_weight,
            vocab_size: config.vocab_size,
            hidden_size: hidden,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Number of decoder layers (for allocating a matching `DecoderCache`).
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Allocate an empty KV cache sized for this decoder.
    pub fn new_cache(&self) -> DecoderCache<R> {
        DecoderCache {
            layers: (0..self.layers.len())
                .map(|_| DecoderLayerCache::default())
                .collect(),
        }
    }

    /// Forward pass using a KV cache. Suitable for both prefill (multi-token
    /// `input_ids`) and incremental decoding (single-token `input_ids`).
    ///
    /// Returns logits for the **current** input window only: `[B, T, vocab]`.
    ///
    /// Delegates to `Self::forward_with_cache_capturing` with no capture
    /// sink, so this path pays no extra cost for the capture feature.
    pub fn forward_with_cache<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        encoder_out: &Tensor<R>,
        position_offset: usize,
        cache: &mut DecoderCache<R>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ActivationOps<R>
            + NormalizationOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + UnaryOps<R>
            + ConditionalOps<R>
            + numr::ops::IndexingOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        self.forward_with_cache_capturing(
            client,
            input_ids,
            encoder_out,
            position_offset,
            cache,
            None,
        )
    }

    /// Same as [`Self::forward_with_cache`], with an optional
    /// [`CrossAttentionCapture`] that collects each requested layer's
    /// cross-attention probabilities for this call.
    pub(crate) fn forward_with_cache_capturing<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        encoder_out: &Tensor<R>,
        position_offset: usize,
        cache: &mut DecoderCache<R>,
        mut capture: Option<&mut CrossAttentionCapture<R>>,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ActivationOps<R>
            + NormalizationOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + UnaryOps<R>
            + ConditionalOps<R>
            + numr::ops::IndexingOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        assert_eq!(
            cache.layers.len(),
            self.layers.len(),
            "decoder cache layer count mismatch"
        );

        let shape = input_ids.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        let tok = self.embed_tokens.forward(client, input_ids)?;

        let pos = self
            .embed_positions
            .narrow(0, position_offset, seq_len)
            .map_err(Error::Numr)?
            .reshape(&[1, seq_len, self.hidden_size])
            .map_err(Error::Numr)?
            .broadcast_to(&[batch, seq_len, self.hidden_size])
            .map_err(Error::Numr)?
            .contiguous()?;

        let mut hidden = client.add(tok.tensor(), &pos).map_err(Error::Numr)?;

        for (i, (layer, layer_cache)) in self.layers.iter().zip(cache.layers.iter_mut()).enumerate()
        {
            let sink = match capture.as_mut() {
                Some(cap) => cap.slots.get_mut(i).and_then(|slot| slot.as_mut()),
                None => None,
            };
            hidden = layer.forward_with_cache_capturing(
                client,
                &hidden,
                encoder_out,
                layer_cache,
                sink,
            )?;
        }

        let normed = self.layer_norm.forward(client, &Var::new(hidden, false))?;
        let hidden = normed.tensor().clone();

        // w_t stays a view so the matmul reads the weight transposed without a copy.
        let w_t = self.tied_out_weight.transpose(0, 1).map_err(Error::Numr)?;
        let logits = client.matmul(&hidden, &w_t).map_err(Error::Numr)?;
        Ok(logits)
    }
}
