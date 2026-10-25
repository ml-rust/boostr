//! Pipeline-parallel forward pieces (embed, layer ranges, head) and the paged
//! KV cache forward pass.

use super::super::Llama;
use super::super::lm_head::final_logits;
use super::super::parts::run_layers;
use crate::error::{Error, Result};
use crate::inference::LayeredKvCache;
use crate::inference::kv_cache::LayeredPagedKvCache;
use crate::model::traits::ModelClient;
use crate::ops::traits::{KvCacheOps, KvSeqLen, PagedAttentionOps, RoPEPackedOps};
use crate::quant::traits::DequantOps;
use numr::autograd::var_add;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = numr::dtype::DType>> Llama<R> {
    /// Contextualized hidden states: `[B, S] -> [B, S, hidden]` after all
    /// transformer layers and the final norm, before `lm_head`.
    ///
    /// Used for embedding extraction (pool over `S`, then normalize).
    pub fn forward_hidden<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
    ) -> Result<numr::autograd::Var<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        let mut hidden = self.embed(client, input_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(client, &hidden, &self.rope)?;
        }
        hidden = self.norm.forward(client, &hidden)?;
        Ok(hidden)
    }

    /// Return a reference to the model's RoPE module (for cos/sin cache access).
    pub fn rope(&self) -> &crate::nn::RoPE<R> {
        &self.rope
    }

    /// Embed token IDs into a hidden state tensor.
    ///
    /// # Arguments
    /// * `client` - Runtime client
    /// * `input_ids` - Token IDs `[B, S]`
    ///
    /// # Returns
    /// Hidden state `[B, S, hidden_size]`
    pub fn forward_embed<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
    ) -> Result<numr::autograd::Var<R>>
    where
        C: ModelClient<R>,
        R::Client: IndexingOps<R> + ScalarOps<R> + TensorOps<R>,
    {
        self.embed(client, input_ids)
    }

    /// Run a range of transformer layers with KV cache.
    ///
    /// Used for pipeline parallelism: each worker owns a contiguous slice of layers
    /// and processes activations forwarded from the previous stage.
    ///
    /// # Arguments
    /// * `client` - Runtime client
    /// * `hidden` - Input hidden state `[B, S, hidden_size]`
    /// * `prev_mlp_out` - Deferred MLP residual from the previous layer (or `None` for first layer)
    /// * `kv_cache` - The range's cache: exactly `end_layer - start_layer` layers.
    ///   Cache slot `0` holds layer `start_layer`. A full-model cache fits the
    ///   range `0..num_layers`.
    /// * `start_layer` - First layer index to run (inclusive)
    /// * `end_layer` - Last layer index to run (exclusive)
    /// * `position` - RoPE position offset
    ///
    /// # Returns
    /// `(hidden, prev_mlp_out)` — the updated hidden state and deferred MLP output for the
    /// next stage (or for `forward_head`).
    ///
    /// # Errors
    ///
    /// A range outside `0..num_layers`, an empty range, or a cache whose
    /// layer count differs from the range length.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_layers_range<C>(
        &self,
        client: &C,
        hidden: numr::autograd::Var<R>,
        prev_mlp_out: Option<numr::autograd::Var<R>>,
        kv_cache: &mut LayeredKvCache<R>,
        start_layer: usize,
        end_layer: usize,
        position: usize,
    ) -> Result<(numr::autograd::Var<R>, Option<numr::autograd::Var<R>>)>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        let num_layers = self.layers.len();
        if start_layer >= end_layer || end_layer > num_layers {
            return Err(Error::ModelError {
                reason: format!(
                    "layer range {start_layer}..{end_layer} is not a non-empty range \
                     inside 0..{num_layers}"
                ),
            });
        }
        run_layers(
            client,
            &self.layers[start_layer..end_layer],
            start_layer,
            &self.rope,
            hidden,
            prev_mlp_out,
            kv_cache,
            position,
        )
    }

    /// Apply the final norm and LM head to produce logits.
    ///
    /// # Arguments
    /// * `client` - Runtime client
    /// * `hidden` - Hidden state `[B, S, hidden_size]`
    /// * `prev_mlp_out` - Deferred MLP residual from the last layer (if any)
    ///
    /// # Returns
    /// Logits `[B, S, vocab_size]`
    pub fn forward_head<C>(
        &self,
        client: &C,
        hidden: numr::autograd::Var<R>,
        prev_mlp_out: Option<numr::autograd::Var<R>>,
    ) -> Result<Tensor<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        final_logits(
            client,
            &self.norm,
            &self.lm_head,
            &self.config,
            hidden,
            prev_mlp_out,
        )
    }

    /// Forward pass for inference with paged KV cache.
    ///
    /// Uses PagedAttention with block table indirection instead of contiguous KV cache.
    ///
    /// # Arguments
    /// * `client` - Runtime client
    /// * `input_ids` - Token IDs `[B, S]`
    /// * `paged_cache` - Layered paged KV cache
    /// * `slot_mapping` - Slot mapping tensor `[B*S]` (I32)
    /// * `block_table` - Block table tensor `[B, max_num_blocks]` (I32)
    /// * `seq_len_k` - Total KV sequence length (including new tokens)
    /// * `position` - RoPE position offset
    /// * `kv_seq_len` - Per-row key end, narrowing `seq_len_k` for a batch of
    ///   mixed-length sequences. `None` keeps every row at `seq_len_k`.
    /// * `rope_positions` - Per-row RoPE position, replacing the shared
    ///   `position` scalar for a batch of mixed cache lengths. `None` keeps
    ///   every row at the shared `position..position+seq_len` window.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_paged_kv_cache<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        paged_cache: &LayeredPagedKvCache<R>,
        slot_mapping: &Tensor<R>,
        block_table: &Tensor<R>,
        seq_len_k: usize,
        position: usize,
        kv_seq_len: Option<KvSeqLen<'_, R>>,
        rope_positions: Option<&Tensor<R>>,
    ) -> Result<Tensor<R>>
    where
        C: ModelClient<R> + RoPEPackedOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + KvCacheOps<R>
            + PagedAttentionOps<R>
            + DequantOps<R>,
    {
        // Embed tokens: [B, S] -> [B, S, hidden]
        let mut hidden = self.embed(client, input_ids)?;

        // Transformer blocks with paged KV cache — deferred residual add fusion
        let mut prev_mlp_out: Option<numr::autograd::Var<R>> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let (h, mlp_out) = layer.forward_with_paged_kv_cache(
                client,
                &hidden,
                prev_mlp_out.as_ref(),
                &self.rope,
                paged_cache,
                i,
                slot_mapping,
                block_table,
                seq_len_k,
                position,
                kv_seq_len,
                rope_positions,
            )?;
            hidden = h;
            prev_mlp_out = Some(mlp_out);
        }

        // Final residual add (deferred from last layer) + norm
        if let Some(last_mlp) = prev_mlp_out {
            hidden = var_add(&hidden, &last_mlp, client).map_err(Error::Numr)?;
        }
        hidden = self.norm.forward(client, &hidden)?;

        // LM head: [B, S, hidden] -> [B, S, vocab]
        let logits = self.head(client, &hidden)?;

        Ok(logits.tensor().clone())
    }
}
