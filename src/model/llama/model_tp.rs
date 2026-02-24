//! Tensor-parallel LLaMA model (Megatron-LM style)
//!
//! Separate from `Llama` to keep zero overhead for non-TP users.
//! Uses `ColumnParallelLinear` for Q/K/V/gate/up projections and
//! `RowParallelLinear` for O/down projections (2 all-reduces per block).

use std::sync::Arc;

use crate::distributed::parallel_embedding::VocabParallelEmbedding;
use crate::distributed::tensor_parallel::{ColumnParallelLinear, RowParallelLinear};
use crate::error::{Error, Result};
use crate::model::config::ModelConfig;
use crate::model::traits::ModelClient;
use crate::nn::{RmsNorm, RoPE};
use crate::ops::impl_generic::attention::multi_head_attention_impl;
use crate::ops::impl_generic::attention::rope::apply_rope_impl;
use numr::autograd::{Var, var_add, var_mul, var_reshape, var_silu};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, TypeConversionOps,
    UtilityOps,
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

struct LlamaBlockTp<R: Runtime> {
    input_layernorm: RmsNorm<R>,
    self_attn: LlamaAttentionTp<R>,
    post_attention_layernorm: RmsNorm<R>,
    mlp: LlamaMlpTp<R>,
}

struct LlamaAttentionTp<R: Runtime> {
    q_proj: ColumnParallelLinear<R>,
    k_proj: ColumnParallelLinear<R>,
    v_proj: ColumnParallelLinear<R>,
    o_proj: RowParallelLinear<R>,
    num_heads: usize,    // LOCAL heads (total / world_size)
    num_kv_heads: usize, // LOCAL kv heads (total / world_size)
    head_dim: usize,
}

struct LlamaMlpTp<R: Runtime> {
    gate_proj: ColumnParallelLinear<R>,
    up_proj: ColumnParallelLinear<R>,
    down_proj: RowParallelLinear<R>,
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
                    config.rms_norm_eps,
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
                    config.rms_norm_eps,
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
            config.rms_norm_eps,
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
                    config.rms_norm_eps,
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
                    config.rms_norm_eps,
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
            config.rms_norm_eps,
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
        R::Client: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + IndexingOps<R> + ShapeOps<R>,
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

// ── LlamaBlockTp ────────────────────────────────────────────────────

impl<R: Runtime<DType = DType>> LlamaBlockTp<R> {
    fn forward<C>(&self, client: &C, x: &Var<R>, rope: &RoPE<R>) -> Result<Var<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + IndexingOps<R> + ShapeOps<R>,
    {
        let normed = self.input_layernorm.forward(client, x)?;
        let attn_out = self.self_attn.forward(client, &normed, rope)?;
        let h = var_add(x, &attn_out, client).map_err(Error::Numr)?;

        let normed = self.post_attention_layernorm.forward(client, &h)?;
        let mlp_out = self.mlp.forward(client, &normed)?;
        var_add(&h, &mlp_out, client).map_err(Error::Numr)
    }
}

// ── LlamaAttentionTp ────────────────────────────────────────────────

impl<R: Runtime<DType = DType>> LlamaAttentionTp<R> {
    fn forward<C>(&self, client: &C, x: &Var<R>, rope: &RoPE<R>) -> Result<Var<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + IndexingOps<R> + ShapeOps<R>,
    {
        let shape = x.shape().to_vec();
        let batch = shape[0];
        let seq_len = shape[1];

        // Q/K/V: column-parallel, each rank gets local heads
        let q = self.q_proj.forward(client, x)?;
        let k = self.k_proj.forward(client, x)?;
        let v = self.v_proj.forward(client, x)?;

        // Reshape to [B, S, local_H, D] then permute to [B, local_H, S, D]
        let q = var_reshape(&q, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let k = var_reshape(&k, &[batch, seq_len, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let v = var_reshape(&v, &[batch, seq_len, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;

        let q = numr::autograd::var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let q = var_contiguous(&q);
        let k = numr::autograd::var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let k = var_contiguous(&k);
        let v = numr::autograd::var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let v = var_contiguous(&v);

        // RoPE on Q and K
        let q = apply_rope_impl(client, &q, rope.cos_cache(), rope.sin_cache())?;
        let k = apply_rope_impl(client, &k, rope.cos_cache(), rope.sin_cache())?;

        // GQA repeat KV if needed (local heads)
        let (k, v) = if self.num_kv_heads < self.num_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            let k = repeat_kv(&k, repeat).map_err(Error::Numr)?;
            let v = repeat_kv(&v, repeat).map_err(Error::Numr)?;
            (k, v)
        } else {
            (k, v)
        };

        // Local attention (no communication)
        let attn_out = multi_head_attention_impl(client, &q, &k, &v, None, self.num_heads)?;

        // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        let attn_out =
            numr::autograd::var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out);
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        // O projection (row-parallel → all-reduce)
        self.o_proj.forward(client, &attn_out)
    }
}

fn var_contiguous<R: Runtime>(v: &Var<R>) -> Var<R> {
    Var::new(v.tensor().contiguous(), v.requires_grad())
}

fn repeat_kv<R: Runtime>(x: &Var<R>, repeat: usize) -> numr::error::Result<Var<R>> {
    if repeat == 1 {
        return Ok(x.clone());
    }
    let shape = x.shape();
    let [b, h_kv, s, d] = [shape[0], shape[1], shape[2], shape[3]];
    let expanded = x.tensor().reshape(&[b, h_kv, 1, s, d])?;
    let expanded = expanded.broadcast_to(&[b, h_kv, repeat, s, d])?;
    let result = expanded.contiguous().reshape(&[b, h_kv * repeat, s, d])?;
    Ok(Var::new(result, x.requires_grad()))
}

// ── LlamaMlpTp ──────────────────────────────────────────────────────

impl<R: Runtime<DType = DType>> LlamaMlpTp<R> {
    fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + IndexingOps<R> + ShapeOps<R>,
    {
        let gate = self.gate_proj.forward(client, x)?;
        let up = self.up_proj.forward(client, x)?;
        let gate_silu = var_silu(&gate, client).map_err(Error::Numr)?;
        let hidden = var_mul(&gate_silu, &up, client).map_err(Error::Numr)?;
        // down_proj is row-parallel → all-reduce
        self.down_proj.forward(client, &hidden)
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
