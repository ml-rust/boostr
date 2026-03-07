//! Inference forward passes and expert weight management for [`LoadedModel`].

use crate::error::{Error, Result};
use crate::inference::kv_cache::LayeredPagedKvCache;
use crate::inference::{LayeredKvCache, LayeredSsmState};
use crate::model::llama::model::blocks::ExpertWeights;
use crate::model::multimodal::ModelInput;
use crate::model::registry::LoadedModel;
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, IndexingOps, NormalizationOps, ScalarOps, ShapeOps,
    TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> LoadedModel<R>
where
    R::Client: IndexingOps<R>,
{
    /// Forward pass with pre-allocated KV cache for transformer inference
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs, shape `[batch_size, seq_len]`
    /// * `kv_cache` - Mutable reference to pre-allocated KV cache
    /// * `position` - Starting position for positional encoding
    ///
    /// # Returns
    /// Logits tensor, shape `[batch_size, seq_len, vocab_size]`
    pub fn forward_with_kv_cache(
        &self,
        input_ids: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        position: usize,
    ) -> Result<Tensor<R>>
    where
        R::Client: ModelClient<R>,
    {
        let device = input_ids.device();
        let client = R::default_client(device);
        match self {
            LoadedModel::Llama(m) => {
                m.forward_with_kv_cache(&client, input_ids, kv_cache, position)
            }
            LoadedModel::LlamaTp(m) => {
                m.forward_with_kv_cache(&client, input_ids, kv_cache, position)
            }
            LoadedModel::Mamba2(_) => Err(Error::ModelError {
                reason: "Mamba2 does not use KV cache — use forward_with_ssm_state() instead"
                    .into(),
            }),
            LoadedModel::Hybrid(_) => Err(Error::ModelError {
                reason: "Hybrid model does not support forward_with_kv_cache — use forward_hybrid() instead"
                    .into(),
            }),
            LoadedModel::Multimodal(m) => {
                m.llm().forward_with_kv_cache(input_ids, kv_cache, position)
            }
        }
    }

    /// Forward pass with paged KV cache for transformer inference
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_paged_kv_cache(
        &self,
        input_ids: &Tensor<R>,
        paged_cache: &LayeredPagedKvCache<R>,
        slot_mapping: &Tensor<R>,
        block_table: &Tensor<R>,
        seq_len_k: usize,
        position: usize,
    ) -> Result<Tensor<R>>
    where
        R::Client: ModelClient<R>,
    {
        let device = input_ids.device();
        let client = R::default_client(device);
        match self {
            LoadedModel::Llama(m) => m.forward_with_paged_kv_cache(
                &client,
                input_ids,
                paged_cache,
                slot_mapping,
                block_table,
                seq_len_k,
                position,
            ),
            LoadedModel::LlamaTp(_) => Err(Error::ModelError {
                reason: "LlamaTp does not yet support paged KV cache".into(),
            }),
            LoadedModel::Mamba2(_) => Err(Error::ModelError {
                reason: "Mamba2 does not use KV cache — use forward_with_ssm_state() instead"
                    .into(),
            }),
            LoadedModel::Hybrid(_) => Err(Error::ModelError {
                reason: "Hybrid model does not yet support paged KV cache".into(),
            }),
            LoadedModel::Multimodal(m) => m.llm().forward_with_paged_kv_cache(
                input_ids,
                paged_cache,
                slot_mapping,
                block_table,
                seq_len_k,
                position,
            ),
        }
    }

    /// Forward pass with SSM state for Mamba2 inference
    pub fn forward_with_ssm_state(
        &self,
        input_ids: &Tensor<R>,
        ssm_state: &mut LayeredSsmState<R>,
    ) -> Result<Tensor<R>>
    where
        R::Client:
            ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
    {
        let device = input_ids.device();
        let client = R::default_client(device);
        match self {
            LoadedModel::Mamba2(m) => m.forward_with_ssm_state(&client, input_ids, ssm_state),
            LoadedModel::Llama(_) | LoadedModel::LlamaTp(_) => Err(Error::ModelError {
                reason: "Llama does not use SSM state — use forward_with_kv_cache() instead".into(),
            }),
            LoadedModel::Hybrid(_) => Err(Error::ModelError {
                reason: "Hybrid model does not support forward_with_ssm_state — use forward_hybrid() instead"
                    .into(),
            }),
            LoadedModel::Multimodal(m) => m.llm().forward_with_ssm_state(input_ids, ssm_state),
        }
    }

    /// Forward pass for hybrid model with both KV cache and SSM state
    pub fn forward_hybrid(
        &self,
        input_ids: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        ssm_state: &mut LayeredSsmState<R>,
        position: usize,
    ) -> Result<Tensor<R>>
    where
        R::Client: ModelClient<R>
            + ConvOps<R>
            + NormalizationOps<R>
            + UnaryOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + ShapeOps<R>
            + TensorOps<R>
            + ScalarOps<R>,
    {
        let device = input_ids.device();
        let client = R::default_client(device);
        match self {
            LoadedModel::Hybrid(m) => {
                m.forward_hybrid(&client, input_ids, kv_cache, ssm_state, position)
            }
            LoadedModel::Llama(_) | LoadedModel::LlamaTp(_) => Err(Error::ModelError {
                reason: "Llama model does not support forward_hybrid — use forward_with_kv_cache()"
                    .into(),
            }),
            LoadedModel::Mamba2(_) => Err(Error::ModelError {
                reason:
                    "Mamba2 model does not support forward_hybrid — use forward_with_ssm_state()"
                        .into(),
            }),
            LoadedModel::Multimodal(m) => m
                .llm()
                .forward_hybrid(input_ids, kv_cache, ssm_state, position),
        }
    }

    /// Embed token IDs into a hidden state tensor.
    ///
    /// Only supported for Llama-family models.
    pub fn forward_embed(&self, input_ids: &Tensor<R>) -> Result<numr::autograd::Var<R>>
    where
        R::Client: ModelClient<R>,
    {
        let device = input_ids.device();
        let client = R::default_client(device);
        match self {
            LoadedModel::Llama(m) => m.forward_embed(&client, input_ids),
            LoadedModel::LlamaTp(_) => Err(Error::ModelError {
                reason: "LlamaTp does not support pipeline-parallel forward_embed".into(),
            }),
            LoadedModel::Mamba2(_) => Err(Error::ModelError {
                reason: "Mamba2 does not support forward_embed — use forward_with_ssm_state()"
                    .into(),
            }),
            LoadedModel::Hybrid(_) => Err(Error::ModelError {
                reason: "Hybrid does not support forward_embed in pipeline mode".into(),
            }),
            LoadedModel::Multimodal(m) => m.llm().forward_embed(input_ids),
        }
    }

    /// Run a range of transformer layers with KV cache.
    ///
    /// Used for pipeline parallelism. Only supported for Llama-family models.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_layers_range(
        &self,
        hidden: numr::autograd::Var<R>,
        prev_mlp_out: Option<numr::autograd::Var<R>>,
        kv_cache: &mut crate::inference::LayeredKvCache<R>,
        start_layer: usize,
        end_layer: usize,
        position: usize,
    ) -> Result<(numr::autograd::Var<R>, Option<numr::autograd::Var<R>>)>
    where
        R::Client: ModelClient<R>,
    {
        let device = hidden.tensor().device();
        let client = R::default_client(device);
        match self {
            LoadedModel::Llama(m) => m.forward_layers_range(
                &client,
                hidden,
                prev_mlp_out,
                kv_cache,
                start_layer,
                end_layer,
                position,
            ),
            LoadedModel::LlamaTp(_) => Err(Error::ModelError {
                reason: "LlamaTp does not support pipeline-parallel forward_layers_range".into(),
            }),
            LoadedModel::Mamba2(_) => Err(Error::ModelError {
                reason: "Mamba2 does not support forward_layers_range".into(),
            }),
            LoadedModel::Hybrid(_) => Err(Error::ModelError {
                reason: "Hybrid does not support forward_layers_range in pipeline mode".into(),
            }),
            LoadedModel::Multimodal(m) => m.llm().forward_layers_range(
                hidden,
                prev_mlp_out,
                kv_cache,
                start_layer,
                end_layer,
                position,
            ),
        }
    }

    /// Apply the final norm and LM head to produce logits.
    ///
    /// Used for pipeline parallelism (last stage). Only supported for Llama-family models.
    pub fn forward_head(
        &self,
        hidden: numr::autograd::Var<R>,
        prev_mlp_out: Option<numr::autograd::Var<R>>,
    ) -> Result<Tensor<R>>
    where
        R::Client: ModelClient<R>,
    {
        let device = hidden.tensor().device();
        let client = R::default_client(device);
        match self {
            LoadedModel::Llama(m) => m.forward_head(&client, hidden, prev_mlp_out),
            LoadedModel::LlamaTp(_) => Err(Error::ModelError {
                reason: "LlamaTp does not support pipeline-parallel forward_head".into(),
            }),
            LoadedModel::Mamba2(_) => Err(Error::ModelError {
                reason: "Mamba2 does not support forward_head".into(),
            }),
            LoadedModel::Hybrid(_) => Err(Error::ModelError {
                reason: "Hybrid does not support forward_head in pipeline mode".into(),
            }),
            LoadedModel::Multimodal(m) => m.llm().forward_head(hidden, prev_mlp_out),
        }
    }

    /// Get the weight tensors for a specific expert.
    ///
    /// Returns `None` if the model is not MoE, the layer/expert is out of range,
    /// or the layer is dense.
    pub fn get_expert_weights(
        &self,
        layer_idx: usize,
        expert_id: usize,
    ) -> Option<ExpertWeights<R>> {
        match self {
            LoadedModel::Llama(m) => m.get_expert_weights(layer_idx, expert_id),
            LoadedModel::Multimodal(m) => m.llm().get_expert_weights(layer_idx, expert_id),
            _ => None,
        }
    }

    /// Replace the weight tensors for a specific expert in-place.
    ///
    /// Used for CPU↔GPU transfers during MoE expert offloading.
    pub fn set_expert_weights(
        &self,
        layer_idx: usize,
        expert_id: usize,
        weights: ExpertWeights<R>,
    ) -> Result<()>
    where
        R::Client: ShapeOps<R>,
    {
        match self {
            LoadedModel::Llama(m) => m.set_expert_weights(layer_idx, expert_id, weights),
            LoadedModel::Multimodal(m) => m.llm().set_expert_weights(layer_idx, expert_id, weights),
            _ => Err(Error::ModelError {
                reason: "set_expert_weights is only supported for Llama MoE models".into(),
            }),
        }
    }

    /// Forward pass for multimodal input.
    ///
    /// For `TextOnly` input, delegates to `forward_with_kv_cache`.
    /// For `Multimodal` input, gets embeddings from the LLM embedding layer,
    /// splices in image/audio embeddings at the marker positions, then runs
    /// through the LLM transformer layers and head.
    ///
    /// Only the `Multimodal` variant of `LoadedModel` supports the full
    /// multimodal path; other variants only support `TextOnly`.
    pub fn forward_multimodal(
        &self,
        input: &ModelInput<R>,
        kv_cache: &mut LayeredKvCache<R>,
        position: usize,
    ) -> Result<Tensor<R>>
    where
        R::Client: ModelClient<R> + BinaryOps<R> + ShapeOps<R>,
    {
        match input {
            ModelInput::TextOnly(input_ids) => {
                self.forward_with_kv_cache(input_ids, kv_cache, position)
            }
            ModelInput::Multimodal {
                input_ids,
                image_embeds,
                audio_embeds,
            } => match self {
                LoadedModel::Multimodal(_) => {
                    // Get token embeddings from the LLM embedding layer
                    let embed_var = self.forward_embed(input_ids)?;
                    let mut hidden = embed_var.tensor().clone();
                    let device = hidden.device();
                    let client = R::default_client(device);

                    // Splice image embeddings at marker positions
                    if let Some((img_embeds, positions)) = image_embeds {
                        hidden = Self::splice_embeddings(&client, &hidden, img_embeds, positions)?;
                    }

                    // Splice audio embeddings at marker positions
                    if let Some((aud_embeds, positions)) = audio_embeds.as_ref() {
                        hidden = Self::splice_embeddings(&client, &hidden, aud_embeds, positions)?;
                    }

                    // Run through LLM layers and head
                    let hidden_var = numr::autograd::Var::new(hidden, false);
                    let (out_var, prev_mlp) = self.forward_layers_range(
                        hidden_var,
                        None,
                        kv_cache,
                        0,
                        self.num_layers(),
                        position,
                    )?;
                    self.forward_head(out_var, prev_mlp)
                }
                _ => Err(Error::ModelError {
                    reason: "Multimodal forward with image/audio embeds is only supported \
                             on LoadedModel::Multimodal"
                        .into(),
                }),
            },
        }
    }

    /// Splice modal embeddings into hidden states at specified positions.
    ///
    /// For each position in `positions`, replaces consecutive tokens starting
    /// at that position with the corresponding embeddings from `modal_embeds`.
    /// `modal_embeds` shape: `[batch, num_modal_tokens, hidden]`
    fn splice_embeddings(
        client: &R::Client,
        hidden: &Tensor<R>,
        modal_embeds: &Tensor<R>,
        positions: &[usize],
    ) -> Result<Tensor<R>>
    where
        R::Client: ShapeOps<R> + BinaryOps<R> + IndexingOps<R>,
    {
        if positions.is_empty() {
            return Ok(hidden.clone());
        }

        let batch = hidden.shape()[0];
        let seq_len = hidden.shape()[1];
        let hidden_dim = hidden.shape()[2];
        let num_modal_tokens = modal_embeds.shape()[1];

        // Build the result by concatenating slices:
        // [before_pos0] [modal_embeds] [after_pos0...]
        // For simplicity with a single contiguous insertion point:
        let pos = positions[0];
        let end = pos + num_modal_tokens;

        if end > seq_len {
            return Err(Error::ModelError {
                reason: format!(
                    "splice_embeddings: insertion at position {pos} with {num_modal_tokens} \
                     tokens exceeds sequence length {seq_len}"
                ),
            });
        }

        // Slice: hidden[:, :pos, :] + modal_embeds + hidden[:, end:, :]
        let mut parts: Vec<Tensor<R>> = Vec::new();

        if pos > 0 {
            let before = hidden.narrow(1, 0, pos).map_err(Error::Numr)?;
            parts.push(before);
        }

        // Use modal_embeds (already [batch, num_modal_tokens, hidden_dim])
        // If batch dim of modal_embeds is 1, broadcast
        let embeds = if modal_embeds.shape()[0] == 1 && batch > 1 {
            modal_embeds
                .broadcast_to(&[batch, num_modal_tokens, hidden_dim])
                .map_err(Error::Numr)?
                .contiguous()
        } else {
            modal_embeds.clone()
        };
        parts.push(embeds);

        if end < seq_len {
            let after = hidden.narrow(1, end, seq_len - end).map_err(Error::Numr)?;
            parts.push(after);
        }

        // Concatenate along sequence dimension
        let part_refs: Vec<&Tensor<R>> = parts.iter().collect();
        client.cat(&part_refs, 1).map_err(Error::Numr)
    }
}

/// CUDA-specific graph-mode inference. Only available when `cuda` feature is enabled.
#[cfg(feature = "cuda")]
impl LoadedModel<numr::runtime::cuda::CudaRuntime> {
    /// Forward pass using a pre-captured CUDA graph's stable-address tensors.
    pub fn forward_graph_mode(
        &self,
        input_ids: &numr::tensor::Tensor<numr::runtime::cuda::CudaRuntime>,
        kv_cache: &mut crate::inference::LayeredKvCache<numr::runtime::cuda::CudaRuntime>,
        device_scalars: &crate::inference::decode_graph::DeviceScalars,
        cos_slice: &numr::autograd::Var<numr::runtime::cuda::CudaRuntime>,
        sin_slice: &numr::autograd::Var<numr::runtime::cuda::CudaRuntime>,
    ) -> Result<numr::tensor::Tensor<numr::runtime::cuda::CudaRuntime>> {
        use numr::runtime::cuda::CudaRuntime;
        let client = CudaRuntime::default_client(input_ids.device());
        match self {
            LoadedModel::Llama(m) => m.forward_graph_mode(
                &client,
                input_ids,
                kv_cache,
                device_scalars,
                cos_slice,
                sin_slice,
            ),
            LoadedModel::LlamaTp(_) => Err(Error::ModelError {
                reason: "LlamaTp does not yet support CUDA graph mode".into(),
            }),
            LoadedModel::Mamba2(_) => Err(Error::ModelError {
                reason: "Mamba2 does not support CUDA graph mode — use forward_with_ssm_state()"
                    .into(),
            }),
            LoadedModel::Hybrid(_) => Err(Error::ModelError {
                reason: "Hybrid model does not yet support CUDA graph mode".into(),
            }),
            LoadedModel::Multimodal(m) => m.llm().forward_graph_mode(
                input_ids,
                kv_cache,
                device_scalars,
                cos_slice,
                sin_slice,
            ),
        }
    }

    /// Graph-mode forward pass with paged KV cache.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_graph_paged(
        &self,
        client: &numr::runtime::cuda::CudaClient,
        input_ids: &numr::tensor::Tensor<numr::runtime::cuda::CudaRuntime>,
        paged_cache: &crate::inference::LayeredPagedKvCache<numr::runtime::cuda::CudaRuntime>,
        slot_mapping: &numr::tensor::Tensor<numr::runtime::cuda::CudaRuntime>,
        block_table: &numr::tensor::Tensor<numr::runtime::cuda::CudaRuntime>,
        device_scalars: &crate::inference::decode_graph::DeviceScalars,
        cos_slice: &numr::autograd::Var<numr::runtime::cuda::CudaRuntime>,
        sin_slice: &numr::autograd::Var<numr::runtime::cuda::CudaRuntime>,
    ) -> Result<numr::tensor::Tensor<numr::runtime::cuda::CudaRuntime>> {
        match self {
            LoadedModel::Llama(m) => m.forward_graph_paged(
                client,
                input_ids,
                paged_cache,
                slot_mapping,
                block_table,
                device_scalars,
                cos_slice,
                sin_slice,
            ),
            LoadedModel::Multimodal(m) => m.llm().forward_graph_paged(
                client,
                input_ids,
                paged_cache,
                slot_mapping,
                block_table,
                device_scalars,
                cos_slice,
                sin_slice,
            ),
            _ => Err(Error::ModelError {
                reason: "Only Llama supports CUDA graph mode with paged attention".into(),
            }),
        }
    }
}
