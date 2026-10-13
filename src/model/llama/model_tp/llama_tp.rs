//! The `LlamaTp` model state, its forward passes, and accessors.

use std::sync::Arc;

use super::super::block_tp::LlamaBlockTp;
use crate::distributed::parallel_embedding::VocabParallelEmbedding;
use crate::distributed::tensor_parallel::{ColumnParallelLinear, gather_from_ranks};
use crate::error::{Error, Result};
use crate::model::config::ModelConfig;
use crate::model::traits::ModelClient;
use crate::nn::{RmsNorm, RoPE};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Communicator, Runtime};
use numr::tensor::Tensor;

/// Tensor-parallel LLaMA model.
pub struct LlamaTp<R: Runtime> {
    pub(super) config: ModelConfig,
    pub(super) embed_tokens: VocabParallelEmbedding<R>,
    pub(super) layers: Vec<LlamaBlockTp<R>>,
    pub(super) norm: RmsNorm<R>,
    pub(super) lm_head: ColumnParallelLinear<R>,
    pub(super) rope: RoPE<R>,
    pub(super) comm: Arc<dyn Communicator>,
    pub(super) world_size: usize,
}

impl<R: Runtime<DType = DType>> LlamaTp<R> {
    /// Forward pass with KV cache for autoregressive inference.
    ///
    /// `kv_cache` is a layered cache: one `KvCache` per layer, each with
    /// `local_kv_heads` (= total_kv_heads / world_size) heads.
    ///
    /// Every rank must call it with the same input and position. The vocab
    /// shards gather across ranks, so each rank returns the full
    /// `[B, S, vocab]` logits.
    pub fn forward_with_kv_cache<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        kv_cache: &mut crate::inference::LayeredKvCache<R>,
        position: usize,
    ) -> Result<Tensor<R>>
    where
        C: ModelClient<R> + CompareOps<R> + UtilityOps<R> + TypeConversionOps<R>,
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
        // Embed tokens: [B, S] -> [B, S, hidden]
        let mut hidden = self.embed_tokens.forward(client, input_ids)?;

        // Transformer blocks with KV cache
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_kv = kv_cache.layer_mut(i).ok_or_else(|| Error::InferenceError {
                reason: format!("KV cache missing for layer {}", i),
            })?;
            hidden =
                layer.forward_with_kv_cache(client, &hidden, &self.rope, layer_kv, position)?;
        }

        // Final norm
        hidden = self.norm.forward(client, &hidden)?;

        // LM head (column-parallel): [B, S, hidden] -> [B, S, vocab/N]
        let logits = self.lm_head.forward(client, &hidden)?;
        // Rank r holds vocab rows [r*V/N, (r+1)*V/N). Gather them in rank order.
        let local = logits.tensor().contiguous()?;
        gather_from_ranks(&local, -1, self.comm.as_ref())
    }

    /// Forward pass: token_ids [B, S] -> logits [B, S, vocab/world_size]
    ///
    /// Note: output is the local shard of logits (vocab dimension split).
    /// For full logits, gather across ranks.
    pub fn forward<C>(&self, client: &C, input_ids: &Var<R>) -> Result<Var<R>>
    where
        C: ModelClient<R> + CompareOps<R> + UtilityOps<R> + TypeConversionOps<R>,
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
        // Embed tokens: [B, S] -> [B, S, hidden]
        let mut hidden = self.embed_tokens.forward(client, input_ids.tensor())?;

        // Transformer blocks
        for layer in &self.layers {
            hidden = layer.forward(client, &hidden, &self.rope)?;
        }

        // Final norm
        hidden = self.norm.forward(client, &hidden)?;

        // LM head (column-parallel): [B, S, hidden] -> [B, S, vocab/N]
        self.lm_head.forward(client, &hidden)
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Compute dtype, read from the replicated final norm weight (owned,
    /// never quantized).
    pub fn dtype(&self) -> DType {
        self.norm.weight().tensor().dtype()
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    pub fn comm(&self) -> &dyn Communicator {
        self.comm.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn cpu_setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    fn tiny_config() -> ModelConfig {
        let yaml = r#"
model_type: llama
vocab_size: 32
hidden_size: 16
num_layers: 2
max_seq_len: 32
intermediate_size: 32
rms_norm_eps: 1.0e-5
attention:
  num_heads: 2
  rope_theta: 10000.0
"#;
        serde_saphyr::from_str(yaml).unwrap()
    }

    #[test]
    fn test_llama_tp_from_config() {
        let (_, device) = cpu_setup();
        let config = tiny_config();
        let comm = Arc::new(NoOpCommunicator);
        let model = LlamaTp::<CpuRuntime>::from_config(&config, &device, comm).unwrap();
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.world_size(), 1);
        assert_eq!(model.dtype(), DType::F32);
    }

    #[test]
    fn test_llama_tp_forward_shape() {
        let (client, device) = cpu_setup();
        let config = tiny_config();
        let comm = Arc::new(NoOpCommunicator);
        let model = LlamaTp::<CpuRuntime>::from_config(&config, &device, comm).unwrap();

        let input_ids = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3], &[1, 4], &device).unwrap(),
            false,
        );

        let logits = model.forward(&client, &input_ids).unwrap();
        // world_size=1 → vocab/1 = 32
        assert_eq!(logits.shape(), &[1, 4, 32]);
    }

    #[test]
    fn test_llama_tp_gqa_config() {
        let (_, device) = cpu_setup();
        let yaml = r#"
model_type: llama
vocab_size: 32
hidden_size: 16
num_layers: 1
max_seq_len: 16
intermediate_size: 32
attention:
  num_heads: 4
  num_kv_heads: 2
"#;
        let config: ModelConfig = serde_saphyr::from_str(yaml).unwrap();
        let comm = Arc::new(NoOpCommunicator);
        let model = LlamaTp::<CpuRuntime>::from_config(&config, &device, comm).unwrap();
        assert_eq!(model.layers[0].self_attn.num_heads, 4);
        assert_eq!(model.layers[0].self_attn.num_kv_heads, 2);
    }

    #[test]
    fn test_llama_tp_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LlamaTp<CpuRuntime>>();
    }
}
