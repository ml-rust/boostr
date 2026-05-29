//! Full Whisper decoder: embeddings → N layers → final LN → tied logits head.

use super::cache::{DecoderCache, DecoderLayerCache};
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
    embed_tokens: Embedding<R>,
    /// Shape `[max_target_positions, hidden]`, added to token embeddings at the
    /// current decoding position(s).
    embed_positions: Tensor<R>,
    layers: Vec<WhisperDecoderLayer<R>>,
    layer_norm: LayerNorm<R>,
    /// Tied to `embed_tokens.weight` (shape `[vocab, hidden]`). We keep an owned
    /// handle here and use matmul with transposed embedding matrix as the logits head.
    tied_out_weight: Tensor<R>,
    vocab_size: usize,
    hidden_size: usize,
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

        for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
            hidden = layer.forward_with_cache(client, &hidden, encoder_out, layer_cache)?;
        }

        let normed = self.layer_norm.forward(client, &Var::new(hidden, false))?;
        let hidden = normed.tensor().clone();

        let w_t = self
            .tied_out_weight
            .transpose(0, 1)
            .map_err(Error::Numr)?
            .contiguous()?;
        let logits = client.matmul(&hidden, &w_t).map_err(Error::Numr)?;
        Ok(logits)
    }

    /// Forward pass over a full decoder sequence, producing logits `[B, T, vocab]`.
    ///
    /// - `input_ids`: `[B, T]` token ids
    /// - `encoder_out`: `[B, S, hidden]` from the encoder
    /// - `position_offset`: index of the first token in `input_ids` relative to
    ///   the decoder start (0 for the first call). Used to slice the learned
    ///   positional embedding correctly for incremental decoding without a KV cache.
    pub fn forward_inference<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        encoder_out: &Tensor<R>,
        position_offset: usize,
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
        let shape = input_ids.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Token embeddings: [B, T, hidden]
        let tok = self.embed_tokens.forward(client, input_ids)?;

        // Positional slice: [seq_len, hidden] from [max_target_positions, hidden]
        let pos = self
            .embed_positions
            .narrow(0, position_offset, seq_len)
            .map_err(Error::Numr)?;
        let pos = pos
            .reshape(&[1, seq_len, self.hidden_size])
            .map_err(Error::Numr)?;
        let pos_b = pos
            .broadcast_to(&[batch, seq_len, self.hidden_size])
            .map_err(Error::Numr)?
            .contiguous()?;

        let mut hidden = client.add(tok.tensor(), &pos_b).map_err(Error::Numr)?;

        // Stack of decoder layers. Causal only matters for prefill (seq_len > 1);
        // for single-token incremental steps it's a no-op.
        for layer in &self.layers {
            hidden = layer.forward_inference(client, &hidden, encoder_out, true)?;
        }

        // Final LN
        let normed = self.layer_norm.forward(client, &Var::new(hidden, false))?;
        let hidden = normed.tensor().clone();

        // Logits = hidden @ embed_tokens.weight.T: [B, T, hidden] @ [hidden, vocab]
        let w_t = self
            .tied_out_weight
            .transpose(0, 1)
            .map_err(Error::Numr)?
            .contiguous()?;
        let logits = client.matmul(&hidden, &w_t).map_err(Error::Numr)?;
        Ok(logits)
    }
}
