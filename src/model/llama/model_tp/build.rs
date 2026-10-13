//! `LlamaTp` constructors: zero-initialized shards from a config, and
//! real weights sharded out of a `VarBuilder`.

use std::sync::Arc;

use super::super::block_tp::{LlamaAttentionTp, LlamaBlockTp, LlamaMlpTp};
use super::LlamaTp;
use crate::distributed::parallel_embedding::VocabParallelEmbedding;
use crate::distributed::tensor_parallel::{ColumnParallelLinear, RowParallelLinear};
use crate::error::{Error, Result};
use crate::model::config::ModelConfig;
use crate::nn::{RmsNorm, RoPE};
use numr::dtype::DType;
use numr::runtime::{Communicator, Runtime};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> LlamaTp<R> {
    /// Create TP model from config with zero-initialized weights.
    pub fn from_config(
        config: &ModelConfig,
        device: &R::Device,
        comm: Arc<dyn Communicator>,
    ) -> Result<Self> {
        config.validate()?;
        if !config.block.is_llama() {
            return Err(Error::ModelError {
                reason: format!(
                    "tensor parallelism runs the Llama block only; model type '{}' needs its \
                     own norms, MLP or attention options. Load it on one device",
                    config.model_type
                ),
            });
        }

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
        let sliding_window = attn_cfg.sliding_window();
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
        if !intermediate.is_multiple_of(world_size) {
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
        let embed_weight = Tensor::<R>::zeros(&[vocab, hidden], dt, device)?;
        let embed_tokens = VocabParallelEmbedding::new(&embed_weight, comm.clone(), true)?;

        // RoPE
        let rope = RoPE::<R>::precompute_freqs(
            config.max_seq_len,
            head_dim,
            attn_cfg.rope_theta,
            attn_cfg.rope_scaling.as_ref(),
            device,
        )?;

        // Layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let block = LlamaBlockTp {
                input_layernorm: RmsNorm::new(
                    Tensor::<R>::ones(&[hidden], dt, device)?,
                    config.rms_norm_eps as f32,
                    true,
                ),
                self_attn: LlamaAttentionTp {
                    q_proj: ColumnParallelLinear::from_shard(
                        Tensor::<R>::zeros(&[local_heads * head_dim, hidden], dt, device)?,
                        None,
                        true,
                    ),
                    k_proj: ColumnParallelLinear::from_shard(
                        Tensor::<R>::zeros(&[local_kv_heads * head_dim, hidden], dt, device)?,
                        None,
                        true,
                    ),
                    v_proj: ColumnParallelLinear::from_shard(
                        Tensor::<R>::zeros(&[local_kv_heads * head_dim, hidden], dt, device)?,
                        None,
                        true,
                    ),
                    o_proj: RowParallelLinear::from_shard(
                        Tensor::<R>::zeros(&[hidden, local_heads * head_dim], dt, device)?,
                        None,
                        comm.clone(),
                        true,
                    ),
                    num_heads: local_heads,
                    num_kv_heads: local_kv_heads,
                    head_dim,
                    q_norm: None,
                    k_norm: None,
                    sliding_window,
                },
                post_attention_layernorm: RmsNorm::new(
                    Tensor::<R>::ones(&[hidden], dt, device)?,
                    config.rms_norm_eps as f32,
                    true,
                ),
                mlp: LlamaMlpTp {
                    gate_proj: ColumnParallelLinear::from_shard(
                        Tensor::<R>::zeros(&[local_intermediate, hidden], dt, device)?,
                        None,
                        true,
                    ),
                    up_proj: ColumnParallelLinear::from_shard(
                        Tensor::<R>::zeros(&[local_intermediate, hidden], dt, device)?,
                        None,
                        true,
                    ),
                    down_proj: RowParallelLinear::from_shard(
                        Tensor::<R>::zeros(&[hidden, local_intermediate], dt, device)?,
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
            Tensor::<R>::ones(&[hidden], dt, device)?,
            config.rms_norm_eps as f32,
            true,
        );

        // LM head (column-parallel over vocab)
        let lm_head = ColumnParallelLinear::from_shard(
            Tensor::<R>::zeros(&[vocab / world_size, hidden], dt, device)?,
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
        if !config.block.is_llama() {
            return Err(Error::ModelError {
                reason: format!(
                    "tensor parallelism runs the Llama block only; model type '{}' needs its \
                     own norms, MLP or attention options. Load it on one device",
                    config.model_type
                ),
            });
        }

        let attn_cfg = config.attention.as_ref().ok_or_else(|| Error::ModelError {
            reason: "LLaMA requires attention config".into(),
        })?;

        let world_size = comm.world_size();
        let rank = comm.rank();
        let hidden = config.hidden_size;
        let num_heads = attn_cfg.num_heads;
        let num_kv_heads = attn_cfg.kv_heads();
        let head_dim = attn_cfg.head_dim(hidden);
        let sliding_window = attn_cfg.sliding_window();

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
        )?;

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
                    q_norm: {
                        let mut avb = layer_vb.pp("self_attn");
                        avb.take_tensor_optional("q_norm.weight")?
                            .map(|w| RmsNorm::new(w, config.rms_norm_eps as f32, false))
                    },
                    k_norm: {
                        let mut avb = layer_vb.pp("self_attn");
                        avb.take_tensor_optional("k_norm.weight")?
                            .map(|w| RmsNorm::new(w, config.rms_norm_eps as f32, false))
                    },
                    sliding_window,
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
}
