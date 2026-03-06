//! LLaMA inference forward passes (KV cache and paged KV cache).

use super::Llama;
use crate::error::{Error, Result};
use crate::inference::LayeredKvCache;
use crate::inference::kv_cache::LayeredPagedKvCache;
use crate::model::traits::ModelClient;
use crate::ops::traits::{KvCacheOps, PagedAttentionOps};
use numr::autograd::var_add;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = numr::dtype::DType>> Llama<R> {
    /// Forward pass for inference with KV cache.
    ///
    /// Unlike `Model::forward`, this:
    /// - Accepts `Tensor<R>` input (no autograd overhead)
    /// - Uses a KV cache for efficient autoregressive decoding
    /// - Takes a position offset for RoPE
    ///
    /// # Arguments
    /// * `client` - Runtime client
    /// * `input_ids` - Token IDs `[B, S]`
    /// * `kv_cache` - Layered KV cache (one per transformer layer)
    /// * `position` - RoPE position offset (= number of previously decoded tokens)
    ///
    /// # Returns
    /// Logits `[B, S, vocab_size]`
    pub fn forward_with_kv_cache<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        position: usize,
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
            + ConditionalOps<R>,
    {
        let profile = std::env::var("BLAZR_PROFILE").is_ok();
        let device = input_ids.device();
        let rc = R::default_client(device);

        macro_rules! sync_log {
            ($t:expr, $msg:expr) => {
                if profile {
                    rc.synchronize();
                    eprintln!("[profile] {}: {:?}", $msg, $t.elapsed());
                }
            };
        }

        let t = std::time::Instant::now();

        // Embed tokens: [B, S] -> [B, S, hidden]
        let mut hidden = self.embed_tokens.forward(client, input_ids)?;
        sync_log!(t, "embed");

        // Transformer blocks with KV cache — deferred residual add fusion
        let mut prev_mlp_out: Option<numr::autograd::Var<R>> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let t_layer = std::time::Instant::now();
            let cache = kv_cache.layer_mut(i).ok_or_else(|| Error::ModelError {
                reason: format!("KV cache missing for layer {i}"),
            })?;
            let (h, mlp_out) = layer.forward_with_kv_cache(
                client,
                &hidden,
                prev_mlp_out.as_ref(),
                &self.rope,
                cache,
                position,
            )?;
            hidden = h;
            prev_mlp_out = Some(mlp_out);
            sync_log!(t_layer, format!("layer {i}"));
        }

        // Final residual add (deferred from last layer) + norm
        let t_norm = std::time::Instant::now();
        if let Some(last_mlp) = prev_mlp_out {
            hidden = var_add(&hidden, &last_mlp, client).map_err(Error::Numr)?;
        }
        hidden = self.norm.forward(client, &hidden)?;
        sync_log!(t_norm, "norm");

        // LM head: [B, S, hidden] -> [B, S, vocab]
        let t_lm = std::time::Instant::now();
        let logits = self.lm_head.forward(client, &hidden)?;
        sync_log!(t_lm, "lm_head");

        if profile {
            eprintln!("[profile] total forward: {:?}", t.elapsed());
        }

        Ok(logits.tensor().clone())
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
        R::Client: IndexingOps<R>,
    {
        self.embed_tokens.forward(client, input_ids)
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
    /// * `kv_cache` - Full layered KV cache (only slots `start_layer..end_layer` are accessed)
    /// * `start_layer` - First layer index to run (inclusive)
    /// * `end_layer` - Last layer index to run (exclusive)
    /// * `position` - RoPE position offset
    ///
    /// # Returns
    /// `(hidden, prev_mlp_out)` — the updated hidden state and deferred MLP output for the
    /// next stage (or for `forward_head`).
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
            + ConditionalOps<R>,
    {
        let end_layer = end_layer.min(self.layers.len());
        let mut hidden = hidden;
        let mut prev_mlp_out = prev_mlp_out;
        for i in start_layer..end_layer {
            let layer = self.layers.get(i).ok_or_else(|| Error::ModelError {
                reason: format!("Layer index {i} out of bounds"),
            })?;
            let cache = kv_cache.layer_mut(i).ok_or_else(|| Error::ModelError {
                reason: format!("KV cache missing for layer {i}"),
            })?;
            let (h, mlp_out) = layer.forward_with_kv_cache(
                client,
                &hidden,
                prev_mlp_out.as_ref(),
                &self.rope,
                cache,
                position,
            )?;
            hidden = h;
            prev_mlp_out = Some(mlp_out);
        }
        Ok((hidden, prev_mlp_out))
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
            + ConditionalOps<R>,
    {
        let mut hidden = hidden;
        if let Some(last_mlp) = prev_mlp_out {
            hidden = numr::autograd::var_add(&hidden, &last_mlp, client).map_err(Error::Numr)?;
        }
        hidden = self.norm.forward(client, &hidden)?;
        let logits = self.lm_head.forward(client, &hidden)?;
        Ok(logits.tensor().clone())
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
            + KvCacheOps<R>
            + PagedAttentionOps<R>,
    {
        // Embed tokens: [B, S] -> [B, S, hidden]
        let mut hidden = self.embed_tokens.forward(client, input_ids)?;

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
        let logits = self.lm_head.forward(client, &hidden)?;

        Ok(logits.tensor().clone())
    }
}
