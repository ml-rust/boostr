//! Tensor-parallel LLaMA model (Megatron-LM style)
//!
//! Separate from `Llama` to keep zero overhead for non-TP users.
//! Uses `ColumnParallelLinear` for Q/K/V/gate/up projections and
//! `RowParallelLinear` for O/down projections (2 all-reduces per block).

use std::sync::Arc;

use super::block_tp::{LlamaAttentionTp, LlamaBlockTp, LlamaMlpTp};
use crate::distributed::parallel_embedding::VocabParallelEmbedding;
use crate::distributed::tensor_parallel::{ColumnParallelLinear, RowParallelLinear};
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
    config: ModelConfig,
    embed_tokens: VocabParallelEmbedding<R>,
    layers: Vec<LlamaBlockTp<R>>,
    norm: RmsNorm<R>,
    lm_head: ColumnParallelLinear<R>,
    rope: RoPE<R>,
    comm: Arc<dyn Communicator>,
    world_size: usize,
}

impl<R: Runtime<DType = DType>> LlamaTp<R> {
    /// Create TP model from config with zero-initialized weights.
    pub fn from_config(
        config: &ModelConfig,
        device: &R::Device,
        comm: Arc<dyn Communicator>,
    ) -> Result<Self> {
        config.validate()?;

        let attn_cfg = config.attention.as_ref().ok_or_else(|| Error::ModelError {
            reason: "LLaMA requires attention config".into(),
        })?;

        let world_size = comm.world_size();
        let hidden = config.hidden_size;
        let vocab = config.vocab_size;
        let intermediate = config.intermediate_size();
        let num_heads = attn_cfg.num_heads;
        let num_kv_heads = attn_cfg.kv_heads();
        let head_dim = attn_cfg.head_dim(hidden);
        let dt = DType::F32;

        // Validate divisibility
        if num_heads % world_size != 0 {
            return Err(Error::DistributedError {
                reason: format!(
                    "num_heads ({}) not divisible by world_size ({})",
                    num_heads, world_size
                ),
            });
        }
        if num_kv_heads % world_size != 0 {
            return Err(Error::DistributedError {
                reason: format!(
                    "num_kv_heads ({}) not divisible by world_size ({})",
                    num_kv_heads, world_size
                ),
            });
        }
        if intermediate % world_size != 0 {
            return Err(Error::DistributedError {
                reason: format!(
                    "intermediate_size ({}) not divisible by world_size ({})",
                    intermediate, world_size
                ),
            });
        }

        let local_heads = num_heads / world_size;
        let local_kv_heads = num_kv_heads / world_size;
        let local_intermediate = intermediate / world_size;

        // Embedding (vocab-parallel)
        let embed_weight = Tensor::<R>::zeros(&[vocab, hidden], dt, device);
        let embed_tokens = VocabParallelEmbedding::new(&embed_weight, comm.clone(), true)?;

        // RoPE
        let rope = RoPE::<R>::precompute_freqs(
            config.max_seq_len,
            head_dim,
            attn_cfg.rope_theta,
            attn_cfg.rope_scaling.as_ref(),
            device,
        );

        // Layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let block = LlamaBlockTp {
                input_layernorm: RmsNorm::new(
                    Tensor::<R>::ones(&[hidden], dt, device),
                    config.rms_norm_eps as f32,
                    true,
                ),
                self_attn: LlamaAttentionTp {
                    q_proj: ColumnParallelLinear::from_shard(
                        Tensor::<R>::zeros(&[local_heads * head_dim, hidden], dt, device),
                        None,
                        true,
                    ),
                    k_proj: ColumnParallelLinear::from_shard(
                        Tensor::<R>::zeros(&[local_kv_heads * head_dim, hidden], dt, device),
                        None,
                        true,
                    ),
                    v_proj: ColumnParallelLinear::from_shard(
                        Tensor::<R>::zeros(&[local_kv_heads * head_dim, hidden], dt, device),
                        None,
                        true,
                    ),
                    o_proj: RowParallelLinear::from_shard(
                        Tensor::<R>::zeros(&[hidden, local_heads * head_dim], dt, device),
                        None,
                        comm.clone(),
                        true,
                    ),
                    num_heads: local_heads,
                    num_kv_heads: local_kv_heads,
                    head_dim,
                },
                post_attention_layernorm: RmsNorm::new(
                    Tensor::<R>::ones(&[hidden], dt, device),
                    config.rms_norm_eps as f32,
                    true,
                ),
                mlp: LlamaMlpTp {
                    gate_proj: ColumnParallelLinear::from_shard(
                        Tensor::<R>::zeros(&[local_intermediate, hidden], dt, device),
                        None,
                        true,
                    ),
                    up_proj: ColumnParallelLinear::from_shard(
                        Tensor::<R>::zeros(&[local_intermediate, hidden], dt, device),
                        None,
                        true,
                    ),
                    down_proj: RowParallelLinear::from_shard(
                        Tensor::<R>::zeros(&[hidden, local_intermediate], dt, device),
                        None,
                        comm.clone(),
                        true,
                    ),
                },
            };
            layers.push(block);
        }

        // Final norm (replicated)
        let norm = RmsNorm::new(
            Tensor::<R>::ones(&[hidden], dt, device),
            config.rms_norm_eps as f32,
            true,
        );

        // LM head (column-parallel over vocab)
        let lm_head = ColumnParallelLinear::from_shard(
            Tensor::<R>::zeros(&[vocab / world_size, hidden], dt, device),
            None,
            true,
        );

        Ok(Self {
            config: config.clone(),
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            comm,
            world_size,
        })
    }

    /// Create TP model from a VarBuilder (loads real weights, shards them).
    pub fn from_varbuilder(
        vb: &mut crate::nn::VarBuilder<R>,
        config: &ModelConfig,
        comm: Arc<dyn Communicator>,
    ) -> Result<Self> {
        config.validate()?;

        let attn_cfg = config.attention.as_ref().ok_or_else(|| Error::ModelError {
            reason: "LLaMA requires attention config".into(),
        })?;

        let world_size = comm.world_size();
        let rank = comm.rank();
        let hidden = config.hidden_size;
        let num_heads = attn_cfg.num_heads;
        let num_kv_heads = attn_cfg.kv_heads();
        let head_dim = attn_cfg.head_dim(hidden);

        if num_heads % world_size != 0 || num_kv_heads % world_size != 0 {
            return Err(Error::DistributedError {
                reason: "heads not divisible by world_size".into(),
            });
        }

        let local_heads = num_heads / world_size;
        let local_kv_heads = num_kv_heads / world_size;

        // RoPE
        let rope = RoPE::<R>::precompute_freqs(
            config.max_seq_len,
            head_dim,
            attn_cfg.rope_theta,
            attn_cfg.rope_scaling.as_ref(),
            vb.device(),
        );

        let mut model_vb = vb.pp("model");

        // Embedding
        let embed_weight = model_vb.take_tensor("embed_tokens.weight")?;
        let embed_tokens = VocabParallelEmbedding::new(&embed_weight, comm.clone(), false)?;

        // Layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let mut layers_vb = model_vb.pp("layers");
            let mut layer_vb = layers_vb.pp(&i.to_string());

            let mut attn_vb = layer_vb.pp("self_attn");
            // Column-parallel: split dim=0 (output features)
            let q_shard = attn_vb.take_tensor_shard("q_proj.weight", 0, rank, world_size)?;
            let k_shard = attn_vb.take_tensor_shard("k_proj.weight", 0, rank, world_size)?;
            let v_shard = attn_vb.take_tensor_shard("v_proj.weight", 0, rank, world_size)?;
            // Row-parallel: split dim=1 (input features)
            let o_shard = attn_vb.take_tensor_shard("o_proj.weight", 1, rank, world_size)?;

            let mut mlp_vb = layer_vb.pp("mlp");
            let gate_shard = mlp_vb.take_tensor_shard("gate_proj.weight", 0, rank, world_size)?;
            let up_shard = mlp_vb.take_tensor_shard("up_proj.weight", 0, rank, world_size)?;
            let down_shard = mlp_vb.take_tensor_shard("down_proj.weight", 1, rank, world_size)?;

            let block = LlamaBlockTp {
                input_layernorm: RmsNorm::new(
                    layer_vb.take_tensor("input_layernorm.weight")?,
                    config.rms_norm_eps as f32,
                    false,
                ),
                self_attn: LlamaAttentionTp {
                    q_proj: ColumnParallelLinear::from_shard(q_shard, None, false),
                    k_proj: ColumnParallelLinear::from_shard(k_shard, None, false),
                    v_proj: ColumnParallelLinear::from_shard(v_shard, None, false),
                    o_proj: RowParallelLinear::from_shard(o_shard, None, comm.clone(), false),
                    num_heads: local_heads,
                    num_kv_heads: local_kv_heads,
                    head_dim,
                },
                post_attention_layernorm: RmsNorm::new(
                    layer_vb.take_tensor("post_attention_layernorm.weight")?,
                    config.rms_norm_eps as f32,
                    false,
                ),
                mlp: LlamaMlpTp {
                    gate_proj: ColumnParallelLinear::from_shard(gate_shard, None, false),
                    up_proj: ColumnParallelLinear::from_shard(up_shard, None, false),
                    down_proj: RowParallelLinear::from_shard(down_shard, None, comm.clone(), false),
                },
            };
            layers.push(block);
        }

        let norm = RmsNorm::new(
            model_vb.take_tensor("norm.weight")?,
            config.rms_norm_eps as f32,
            false,
        );

        // LM head (column-parallel)
        let lm_head_shard = vb.take_tensor_shard("lm_head.weight", 0, rank, world_size)?;
        let lm_head = ColumnParallelLinear::from_shard(lm_head_shard, None, false);

        Ok(Self {
            config: config.clone(),
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            comm,
            world_size,
        })
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
        serde_yaml::from_str(yaml).unwrap()
    }

    #[test]
    fn test_llama_tp_from_config() {
        let (_, device) = cpu_setup();
        let config = tiny_config();
        let comm = Arc::new(NoOpCommunicator);
        let model = LlamaTp::<CpuRuntime>::from_config(&config, &device, comm).unwrap();
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.world_size(), 1);
    }

    #[test]
    fn test_llama_tp_forward_shape() {
        let (client, device) = cpu_setup();
        let config = tiny_config();
        let comm = Arc::new(NoOpCommunicator);
        let model = LlamaTp::<CpuRuntime>::from_config(&config, &device, comm).unwrap();

        let input_ids = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3], &[1, 4], &device),
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
        let config: ModelConfig = serde_yaml::from_str(yaml).unwrap();
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
