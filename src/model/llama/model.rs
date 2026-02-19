//! LLaMA model architecture
//!
//! Implements the LLaMA transformer with:
//! - Token embeddings + LM head
//! - N x (RmsNorm -> GQA Attention -> residual -> RmsNorm -> SwiGLU MLP -> residual)
//! - Final RmsNorm

use crate::error::{Error, Result};
use crate::model::config::ModelConfig;
use crate::model::traits::{Model, ModelClient};
use crate::nn::{Embedding, Linear, RmsNorm, RoPE};
use crate::ops::impl_generic::attention::multi_head_attention_impl;
use crate::ops::impl_generic::rope::apply_rope_impl;
use numr::autograd::{Var, var_add, var_mul, var_reshape, var_sigmoid};
use numr::dtype::DType;
use numr::ops::{IndexingOps, ReduceOps, ScalarOps, ShapeOps, TensorOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Full LLaMA model
pub struct Llama<R: Runtime> {
    config: ModelConfig,
    embed_tokens: Embedding<R>,
    layers: Vec<LlamaBlock<R>>,
    norm: RmsNorm<R>,
    lm_head: Linear<R>,
    rope: RoPE<R>,
}

/// Single transformer block
struct LlamaBlock<R: Runtime> {
    input_layernorm: RmsNorm<R>,
    self_attn: LlamaAttention<R>,
    post_attention_layernorm: RmsNorm<R>,
    mlp: LlamaMlp<R>,
}

/// GQA attention with Q/K/V projections
struct LlamaAttention<R: Runtime> {
    q_proj: Linear<R>,
    k_proj: Linear<R>,
    v_proj: Linear<R>,
    o_proj: Linear<R>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

/// SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x))
struct LlamaMlp<R: Runtime> {
    gate_proj: Linear<R>,
    up_proj: Linear<R>,
    down_proj: Linear<R>,
}

// ── Model impl ──────────────────────────────────────────────────────

impl<R: Runtime<DType = DType>> Model<R> for Llama<R> {
    fn from_varbuilder(vb: &mut crate::nn::VarBuilder<R>, config: &ModelConfig) -> Result<Self> {
        config.validate()?;

        let attn_cfg = config.attention.as_ref().ok_or_else(|| Error::ModelError {
            reason: "LLaMA requires attention config".into(),
        })?;

        let hidden = config.hidden_size;
        let num_heads = attn_cfg.num_heads;
        let num_kv_heads = attn_cfg.kv_heads();
        let head_dim = attn_cfg.head_dim(hidden);

        // RoPE cache (borrow device before mutable borrows)
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
        let embed_tokens = Embedding::new(embed_weight, false);

        // Transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let mut layers_vb = model_vb.pp("layers");
            let mut layer_vb = layers_vb.pp(&i.to_string());

            let mut attn_vb = layer_vb.pp("self_attn");
            let q_proj = Linear::new(attn_vb.take_tensor("q_proj.weight")?, None, false);
            let k_proj = Linear::new(attn_vb.take_tensor("k_proj.weight")?, None, false);
            let v_proj = Linear::new(attn_vb.take_tensor("v_proj.weight")?, None, false);
            let o_proj = Linear::new(attn_vb.take_tensor("o_proj.weight")?, None, false);

            let mut mlp_vb = layer_vb.pp("mlp");
            let gate_proj = Linear::new(mlp_vb.take_tensor("gate_proj.weight")?, None, false);
            let up_proj = Linear::new(mlp_vb.take_tensor("up_proj.weight")?, None, false);
            let down_proj = Linear::new(mlp_vb.take_tensor("down_proj.weight")?, None, false);

            let block = LlamaBlock {
                input_layernorm: RmsNorm::new(
                    layer_vb.take_tensor("input_layernorm.weight")?,
                    config.rms_norm_eps,
                    false,
                ),
                self_attn: LlamaAttention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                },
                post_attention_layernorm: RmsNorm::new(
                    layer_vb.take_tensor("post_attention_layernorm.weight")?,
                    config.rms_norm_eps,
                    false,
                ),
                mlp: LlamaMlp {
                    gate_proj,
                    up_proj,
                    down_proj,
                },
            };
            layers.push(block);
        }

        // Final norm
        let norm = RmsNorm::new(
            model_vb.take_tensor("norm.weight")?,
            config.rms_norm_eps,
            false,
        );

        // LM head
        let lm_head = Linear::new(vb.take_tensor("lm_head.weight")?, None, false);

        Ok(Self {
            config: config.clone(),
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
        })
    }

    fn from_config(config: &ModelConfig, device: &R::Device) -> Result<Self> {
        config.validate()?;

        let attn_cfg = config.attention.as_ref().ok_or_else(|| Error::ModelError {
            reason: "LLaMA requires attention config".into(),
        })?;

        let hidden = config.hidden_size;
        let vocab = config.vocab_size;
        let intermediate = config.intermediate_size();
        let num_heads = attn_cfg.num_heads;
        let num_kv_heads = attn_cfg.kv_heads();
        let head_dim = attn_cfg.head_dim(hidden);
        let dt = DType::F32;

        // Embedding
        let embed_weight = Tensor::<R>::zeros(&[vocab, hidden], dt, device);
        let embed_tokens = Embedding::new(embed_weight, true);

        // RoPE cache
        let rope = RoPE::<R>::precompute_freqs(
            config.max_seq_len,
            head_dim,
            attn_cfg.rope_theta,
            attn_cfg.rope_scaling.as_ref(),
            device,
        );

        // Transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let block = LlamaBlock {
                input_layernorm: RmsNorm::new(
                    Tensor::<R>::ones(&[hidden], dt, device),
                    config.rms_norm_eps,
                    true,
                ),
                self_attn: LlamaAttention {
                    q_proj: Linear::new(
                        Tensor::<R>::zeros(&[num_heads * head_dim, hidden], dt, device),
                        None,
                        true,
                    ),
                    k_proj: Linear::new(
                        Tensor::<R>::zeros(&[num_kv_heads * head_dim, hidden], dt, device),
                        None,
                        true,
                    ),
                    v_proj: Linear::new(
                        Tensor::<R>::zeros(&[num_kv_heads * head_dim, hidden], dt, device),
                        None,
                        true,
                    ),
                    o_proj: Linear::new(
                        Tensor::<R>::zeros(&[hidden, num_heads * head_dim], dt, device),
                        None,
                        true,
                    ),
                    num_heads,
                    num_kv_heads,
                    head_dim,
                },
                post_attention_layernorm: RmsNorm::new(
                    Tensor::<R>::ones(&[hidden], dt, device),
                    config.rms_norm_eps,
                    true,
                ),
                mlp: LlamaMlp {
                    gate_proj: Linear::new(
                        Tensor::<R>::zeros(&[intermediate, hidden], dt, device),
                        None,
                        true,
                    ),
                    up_proj: Linear::new(
                        Tensor::<R>::zeros(&[intermediate, hidden], dt, device),
                        None,
                        true,
                    ),
                    down_proj: Linear::new(
                        Tensor::<R>::zeros(&[hidden, intermediate], dt, device),
                        None,
                        true,
                    ),
                },
            };
            layers.push(block);
        }

        // Final norm
        let norm = RmsNorm::new(
            Tensor::<R>::ones(&[hidden], dt, device),
            config.rms_norm_eps,
            true,
        );

        // LM head
        let lm_head = Linear::new(Tensor::<R>::zeros(&[vocab, hidden], dt, device), None, true);

        Ok(Self {
            config: config.clone(),
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
        })
    }

    fn forward<C>(&self, client: &C, input_ids: &Var<R>) -> Result<Var<R>>
    where
        C: ModelClient<R>,
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

        // LM head: [B, S, hidden] -> [B, S, vocab]
        self.lm_head.forward(client, &hidden)
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

// ── LlamaBlock ──────────────────────────────────────────────────────

impl<R: Runtime<DType = DType>> LlamaBlock<R> {
    fn forward<C>(&self, client: &C, x: &Var<R>, rope: &RoPE<R>) -> Result<Var<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + IndexingOps<R> + ShapeOps<R>,
    {
        // Pre-norm attention + residual
        let normed = self.input_layernorm.forward(client, x)?;
        let attn_out = self.self_attn.forward(client, &normed, rope)?;
        let h = var_add(x, &attn_out, client).map_err(Error::Numr)?;

        // Pre-norm MLP + residual
        let normed = self.post_attention_layernorm.forward(client, &h)?;
        let mlp_out = self.mlp.forward(client, &normed)?;
        var_add(&h, &mlp_out, client).map_err(Error::Numr)
    }
}

// ── LlamaAttention ──────────────────────────────────────────────────

impl<R: Runtime<DType = DType>> LlamaAttention<R> {
    fn forward<C>(&self, client: &C, x: &Var<R>, rope: &RoPE<R>) -> Result<Var<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + IndexingOps<R> + ShapeOps<R>,
    {
        let shape = x.shape().to_vec();
        let batch = shape[0];
        let seq_len = shape[1];

        // Q/K/V projections: [B, S, hidden] -> [B, S, heads*head_dim]
        let q = self.q_proj.forward(client, x)?;
        let k = self.k_proj.forward(client, x)?;
        let v = self.v_proj.forward(client, x)?;

        // Reshape to [B, S, H, D] then permute to [B, H, S, D]
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

        // Apply RoPE to Q and K (call impl_generic directly)
        let q = apply_rope_impl(client, &q, rope.cos_cache(), rope.sin_cache())?;
        let k = apply_rope_impl(client, &k, rope.cos_cache(), rope.sin_cache())?;

        // GQA: repeat K/V heads to match Q heads if needed
        let (k, v) = if self.num_kv_heads < self.num_heads {
            let repeat = self.num_heads / self.num_kv_heads;
            let k = repeat_kv(&k, repeat).map_err(Error::Numr)?;
            let v = repeat_kv(&v, repeat).map_err(Error::Numr)?;
            (k, v)
        } else {
            (k, v)
        };

        // Multi-head attention
        let attn_out = multi_head_attention_impl(client, &q, &k, &v, None, self.num_heads)?;

        // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        let attn_out =
            numr::autograd::var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out);
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        // Output projection
        self.o_proj.forward(client, &attn_out)
    }
}

/// Make a Var contiguous (copies data if non-contiguous layout).
fn var_contiguous<R: Runtime>(v: &Var<R>) -> Var<R> {
    Var::new(v.tensor().contiguous(), v.requires_grad())
}

/// Repeat KV heads for GQA: [B, H_kv, S, D] -> [B, H_kv * repeat, S, D]
fn repeat_kv<R: Runtime>(x: &Var<R>, repeat: usize) -> numr::error::Result<Var<R>> {
    if repeat == 1 {
        return Ok(x.clone());
    }
    let shape = x.shape();
    let [b, h_kv, s, d] = [shape[0], shape[1], shape[2], shape[3]];

    // Reshape to [B, H_kv, 1, S, D] then broadcast to [B, H_kv, repeat, S, D]
    let expanded = x.tensor().reshape(&[b, h_kv, 1, s, d])?;
    let expanded = expanded.broadcast_to(&[b, h_kv, repeat, s, d])?;
    // Reshape to [B, H_kv * repeat, S, D] — need contiguous for reshape after broadcast
    let result = expanded.contiguous().reshape(&[b, h_kv * repeat, s, d])?;
    Ok(Var::new(result, x.requires_grad()))
}

// ── LlamaMlp ────────────────────────────────────────────────────────

impl<R: Runtime<DType = DType>> LlamaMlp<R> {
    /// SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
    fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + IndexingOps<R> + ShapeOps<R>,
    {
        let gate = self.gate_proj.forward(client, x)?;
        let up = self.up_proj.forward(client, x)?;

        // silu(gate) = gate * sigmoid(gate)
        let gate_sigmoid = var_sigmoid(&gate, client).map_err(Error::Numr)?;
        let gate_silu = var_mul(&gate, &gate_sigmoid, client).map_err(Error::Numr)?;

        // silu(gate) * up
        let hidden = var_mul(&gate_silu, &up, client).map_err(Error::Numr)?;

        self.down_proj.forward(client, &hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

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
    fn test_llama_from_config() {
        let (_, device) = cpu_setup();
        let config = tiny_config();
        let model = Llama::<CpuRuntime>::from_config(&config, &device).unwrap();
        assert_eq!(model.layers.len(), 2);
    }

    #[test]
    fn test_llama_forward_shape() {
        let (client, device) = cpu_setup();
        let config = tiny_config();
        let model = Llama::<CpuRuntime>::from_config(&config, &device).unwrap();

        // Input: [B=1, S=4] token IDs
        let input_ids = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3], &[1, 4], &device),
            false,
        );

        let logits = model.forward(&client, &input_ids).unwrap();
        // Output: [1, 4, 32] (batch, seq, vocab)
        assert_eq!(logits.shape(), &[1, 4, 32]);
    }

    #[test]
    fn test_llama_gqa_config() {
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
        let model = Llama::<CpuRuntime>::from_config(&config, &device).unwrap();
        assert_eq!(model.layers[0].self_attn.num_heads, 4);
        assert_eq!(model.layers[0].self_attn.num_kv_heads, 2);
    }

    #[test]
    fn test_llama_gqa_forward() {
        let (client, device) = cpu_setup();
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
        let model = Llama::<CpuRuntime>::from_config(&config, &device).unwrap();

        let input_ids = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2], &[1, 3], &device),
            false,
        );
        let logits = model.forward(&client, &input_ids).unwrap();
        assert_eq!(logits.shape(), &[1, 3, 32]);
    }

    #[test]
    fn test_swiglu_mlp() {
        let (client, device) = cpu_setup();
        let mlp = LlamaMlp {
            gate_proj: Linear::new(
                Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[2, 4], &device),
                None,
                false,
            ),
            up_proj: Linear::new(
                Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[2, 4], &device),
                None,
                false,
            ),
            down_proj: Linear::new(
                Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[4, 2], &device),
                None,
                false,
            ),
        };

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[1, 4], &device),
            false,
        );
        let out = mlp.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }
}
