//! LLaMA CUDA graph-mode forward pass.

#[cfg(feature = "cuda")]
use super::Llama;
#[cfg(feature = "cuda")]
use crate::error::{Error, Result};
#[cfg(feature = "cuda")]
use crate::inference::LayeredKvCache;
#[cfg(feature = "cuda")]
use numr::autograd::var_add;

#[cfg(feature = "cuda")]
impl Llama<numr::runtime::cuda::CudaRuntime> {
    /// Graph-mode forward pass — all CUDA ops use stable device addresses.
    ///
    /// Designed for CUDA graph capture+replay. The caller must:
    /// 1. Pre-allocate `kv_cache` at full capacity (`max_seq_len`) before capture.
    /// 2. Provide `DeviceScalars` with correct seq_len values before each replay.
    /// 3. Provide `cos_slice`/`sin_slice` updated to the current position via D2D copy.
    ///
    /// Returns the logits tensor whose device address is stable across graph replays.
    pub fn forward_graph_mode(
        &self,
        client: &numr::runtime::cuda::CudaClient,
        input_ids: &numr::tensor::Tensor<numr::runtime::cuda::CudaRuntime>,
        kv_cache: &mut LayeredKvCache<numr::runtime::cuda::CudaRuntime>,
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
            + numr::ops::ConditionalOps<numr::runtime::cuda::CudaRuntime>,
    {
        // Embed tokens
        let mut hidden = self.embed_tokens.forward(client, input_ids)?;

        // Transformer layers — deferred residual add fusion
        let mut prev_mlp_out: Option<numr::autograd::Var<numr::runtime::cuda::CudaRuntime>> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let cache = kv_cache.layer_mut(i).ok_or_else(|| Error::ModelError {
                reason: format!("KV cache missing for layer {i}"),
            })?;
            let (h, mlp_out) = layer.forward_graph_mode(
                client,
                &hidden,
                prev_mlp_out.as_ref(),
                cos_slice,
                sin_slice,
                cache,
                device_scalars,
            )?;
            hidden = h;
            prev_mlp_out = Some(mlp_out);
        }

        // Final residual add (deferred from last layer) + norm
        if let Some(last_mlp) = prev_mlp_out {
            hidden = var_add(&hidden, &last_mlp, client).map_err(Error::Numr)?;
        }
        hidden = self.norm.forward(client, &hidden)?;
        let logits = self.lm_head.forward(client, &hidden)?;

        Ok(logits.tensor().clone())
    }
}
