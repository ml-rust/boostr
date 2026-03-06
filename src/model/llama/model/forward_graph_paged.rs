//! LLaMA CUDA graph-mode forward pass with paged KV cache.
//!
//! Like `forward_graph.rs` but uses `LayeredPagedKvCache` and the graph-mode
//! paged decode attention kernel that reads seq_len_k from device memory.

#[cfg(feature = "cuda")]
use super::Llama;
#[cfg(feature = "cuda")]
use crate::error::{Error, Result};
#[cfg(feature = "cuda")]
use crate::inference::LayeredPagedKvCache;
#[cfg(feature = "cuda")]
use numr::autograd::var_add;

#[cfg(feature = "cuda")]
impl Llama<numr::runtime::cuda::CudaRuntime> {
    /// Graph-mode forward pass with paged KV cache.
    ///
    /// Designed for CUDA graph capture+replay with paged attention.
    /// The caller must:
    /// 1. Pre-allocate `paged_cache` with enough blocks before capture.
    /// 2. Pre-allocate `slot_mapping` and `block_table` with stable device addresses.
    /// 3. Provide `PagedDeviceScalars` updated before each replay.
    /// 4. Provide `cos_slice`/`sin_slice` updated to the current position via D2D copy.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_graph_paged(
        &self,
        client: &numr::runtime::cuda::CudaClient,
        input_ids: &numr::tensor::Tensor<numr::runtime::cuda::CudaRuntime>,
        paged_cache: &LayeredPagedKvCache<numr::runtime::cuda::CudaRuntime>,
        slot_mapping: &numr::tensor::Tensor<numr::runtime::cuda::CudaRuntime>,
        block_table: &numr::tensor::Tensor<numr::runtime::cuda::CudaRuntime>,
        device_scalars: &crate::inference::decode_graph::DeviceScalars,
        cos_slice: &numr::autograd::Var<numr::runtime::cuda::CudaRuntime>,
        sin_slice: &numr::autograd::Var<numr::runtime::cuda::CudaRuntime>,
    ) -> Result<numr::tensor::Tensor<numr::runtime::cuda::CudaRuntime>>
    where
        numr::runtime::cuda::CudaClient: crate::model::traits::ModelClient<numr::runtime::cuda::CudaRuntime>
            + numr::ops::TensorOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::ScalarOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::ReduceOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::IndexingOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::ShapeOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::ActivationOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::BinaryOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::UnaryOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::CompareOps<numr::runtime::cuda::CudaRuntime>
            + numr::ops::ConditionalOps<numr::runtime::cuda::CudaRuntime>
            + crate::ops::KvCacheOps<numr::runtime::cuda::CudaRuntime>
            + crate::ops::PagedAttentionOps<numr::runtime::cuda::CudaRuntime>,
    {
        use crate::ops::cuda::attention::paged_decode::paged_decode_attention_fwd_graph;
        use numr::autograd::Var;

        // Embed tokens
        let mut hidden = self.embed_tokens.forward(client, input_ids)?;

        // Derive batch size from input shape so batched graph capture works.
        // Graph mode always decodes one token per sequence, so seq_len is always 1.
        let batch = input_ids.shape()[0];
        let seq_len = 1usize; // graph mode is always single-token decode per sequence
        let block_size = paged_cache.block_size();
        let max_num_blocks = block_table.shape()[block_table.shape().len() - 1];

        // Transformer layers
        let mut prev_mlp_out: Option<numr::autograd::Var<numr::runtime::cuda::CudaRuntime>> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            // Fused residual add from previous layer's MLP output
            if let Some(ref prev) = prev_mlp_out {
                hidden = var_add(&hidden, prev, client).map_err(Error::Numr)?;
            }

            // Pre-norm
            let normed = layer.input_layernorm.forward(client, &hidden)?;

            // Attention: Q/K/V projections
            let attn = &layer.self_attn;
            let qkv = crate::nn::MaybeQuantLinear::forward_batch(
                &[&attn.q_proj, &attn.k_proj, &attn.v_proj],
                client,
                &normed,
            )?;
            let (q, k, v) = (&qkv[0], &qkv[1], &qkv[2]);

            // Reshape and permute
            let q =
                numr::autograd::var_reshape(q, &[batch, seq_len, attn.num_heads, attn.head_dim])
                    .map_err(Error::Numr)?;
            let k =
                numr::autograd::var_reshape(k, &[batch, seq_len, attn.num_kv_heads, attn.head_dim])
                    .map_err(Error::Numr)?;
            let v =
                numr::autograd::var_reshape(v, &[batch, seq_len, attn.num_kv_heads, attn.head_dim])
                    .map_err(Error::Numr)?;

            let q = numr::autograd::var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;
            let k = numr::autograd::var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?;
            let v = numr::autograd::var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?;

            let q = numr::autograd::var_contiguous(&q);
            let k = numr::autograd::var_contiguous(&k);
            let v = numr::autograd::var_contiguous(&v);

            // Apply RoPE using stable cos/sin slices
            let q = client.apply_rope_interleaved(&q, cos_slice, sin_slice)?;
            let k = client.apply_rope_interleaved(&k, cos_slice, sin_slice)?;

            // Insert K/V into paged cache using slot_mapping
            let k_flat = k
                .tensor()
                .permute(&[0, 2, 1, 3])? // [B, S, H_kv, D]
                .contiguous()
                .reshape(&[batch * seq_len, attn.num_kv_heads, attn.head_dim])?;
            let v_flat = v.tensor().permute(&[0, 2, 1, 3])?.contiguous().reshape(&[
                batch * seq_len,
                attn.num_kv_heads,
                attn.head_dim,
            ])?;

            let layer_cache = paged_cache.layer(i);
            layer_cache.update(&k_flat, &v_flat, slot_mapping, client)?;

            // Pre-allocate output for graph stability
            let attn_output = numr::tensor::Tensor::<numr::runtime::cuda::CudaRuntime>::empty(
                &[batch, attn.num_heads, 1, attn.head_dim],
                numr::dtype::DType::F32,
                input_ids.device(),
            );

            // Paged decode attention with device-side seq_len_k
            paged_decode_attention_fwd_graph(
                client,
                q.tensor(),
                layer_cache.k_cache(),
                layer_cache.v_cache(),
                block_table,
                &attn_output,
                batch,
                attn.num_heads,
                attn.num_kv_heads,
                device_scalars.seq_len_k_ptr(),
                attn.head_dim,
                block_size,
                max_num_blocks,
            )?;

            // Reshape: [B, H, 1, D] -> [B, 1, H*D]
            let attn_out = Var::new(attn_output, false);
            let attn_out =
                numr::autograd::var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
            let attn_out = numr::autograd::var_contiguous(&attn_out);
            let attn_out = numr::autograd::var_reshape(
                &attn_out,
                &[batch, seq_len, attn.num_heads * attn.head_dim],
            )
            .map_err(Error::Numr)?;

            let attn_out = attn.o_proj.forward(client, &attn_out)?;

            // Post-attention norm + MLP
            let post_attn = var_add(&hidden, &attn_out, client).map_err(Error::Numr)?;
            let normed_post = layer.post_attention_layernorm.forward(client, &post_attn)?;
            let mlp_out = layer.mlp.forward(client, &normed_post)?;

            hidden = post_attn;
            prev_mlp_out = Some(mlp_out);
        }

        // Final residual + norm + lm_head
        if let Some(last_mlp) = prev_mlp_out {
            hidden = var_add(&hidden, &last_mlp, client).map_err(Error::Numr)?;
        }
        hidden = self.norm.forward(client, &hidden)?;
        let logits = self.lm_head.forward(client, &hidden)?;

        Ok(logits.tensor().clone())
    }
}
