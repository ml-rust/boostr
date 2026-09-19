//! CUDA-specific graph-mode inference. Only available when `cuda` feature is enabled.

#[cfg(feature = "cuda")]
use crate::error::{Error, Result};
#[cfg(feature = "cuda")]
use crate::model::registry::LoadedModel;
#[cfg(feature = "cuda")]
use numr::runtime::Runtime;

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
            LoadedModel::Mamba1(_) | LoadedModel::Mamba2(_) | LoadedModel::Mamba3(_) => {
                Err(Error::ModelError {
                    reason: "SSM models do not support CUDA graph mode".into(),
                })
            }
            LoadedModel::Hybrid(_) => Err(Error::ModelError {
                reason: "Hybrid model does not yet support CUDA graph mode".into(),
            }),
            LoadedModel::Qwen35(_) => Err(Error::ModelError {
                reason: "qwen35 carries GDN state — use forward_qwen35_graph_mode()".into(),
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

    /// True when [`forward_qwen35_graph_mode`](Self::forward_qwen35_graph_mode)
    /// serves this model: a `qwen35` hybrid whose GDN state and KV cache both
    /// carry across graph replays in place.
    pub fn supports_gdn_graph_mode(&self) -> bool {
        match self {
            LoadedModel::Qwen35(_) => true,
            LoadedModel::Multimodal(m) => m.llm().supports_gdn_graph_mode(),
            _ => false,
        }
    }

    /// Graph-mode decode forward for the `qwen35` hybrid: full-attention
    /// layers over a full-capacity `kv_cache`, GDN layers over `gdn_state`
    /// buffers written in place. See `Qwen35Model::forward_qwen35_graph_mode`
    /// for the per-replay contract.
    pub fn forward_qwen35_graph_mode(
        &self,
        client: &numr::runtime::cuda::CudaClient,
        input_ids: &numr::tensor::Tensor<numr::runtime::cuda::CudaRuntime>,
        kv_cache: &crate::inference::LayeredKvCache<numr::runtime::cuda::CudaRuntime>,
        gdn_state: &crate::inference::LayeredGdnState<numr::runtime::cuda::CudaRuntime>,
        device_scalars: &crate::inference::decode_graph::DeviceScalars,
        mrope: &crate::inference::decode_graph::MropeScalars,
    ) -> Result<numr::tensor::Tensor<numr::runtime::cuda::CudaRuntime>> {
        match self {
            LoadedModel::Qwen35(m) => m.forward_qwen35_graph_mode(
                client,
                input_ids,
                kv_cache,
                gdn_state,
                device_scalars,
                mrope,
            ),
            LoadedModel::Multimodal(m) => m.llm().forward_qwen35_graph_mode(
                client,
                input_ids,
                kv_cache,
                gdn_state,
                device_scalars,
                mrope,
            ),
            _ => Err(Error::ModelError {
                reason: format!(
                    "{} does not carry GDN state — use forward_graph_mode()",
                    self.model_type()
                ),
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
