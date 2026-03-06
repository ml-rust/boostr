//! Inference forward passes and expert weight management for [`LoadedModel`].

use crate::error::{Error, Result};
use crate::inference::kv_cache::LayeredPagedKvCache;
use crate::inference::{LayeredKvCache, LayeredSsmState};
use crate::model::llama::model::blocks::ExpertWeights;
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
            _ => Err(Error::ModelError {
                reason: "set_expert_weights is only supported for Llama MoE models".into(),
            }),
        }
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
            _ => Err(Error::ModelError {
                reason: "Only Llama supports CUDA graph mode with paged attention".into(),
            }),
        }
    }
}
