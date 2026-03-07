//! LLaMA model: struct definition, construction, and training forward pass.

use super::blocks::{LlamaBlock, build_block_from_config, build_block_from_varbuilder};
use crate::error::{Error, Result};
use crate::model::config::ModelConfig;
use crate::model::traits::{Model, ModelClient};
use crate::nn::{Embedding, Linear, MaybeQuantLinear, RmsNorm, RoPE};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Full LLaMA model
pub struct Llama<R: Runtime> {
    pub(super) config: ModelConfig,
    pub(super) embed_tokens: Embedding<R>,
    pub(super) layers: Vec<LlamaBlock<R>>,
    pub(super) norm: RmsNorm<R>,
    pub(super) lm_head: MaybeQuantLinear<R>,
    pub(super) rope: RoPE<R>,
}

impl<R: Runtime<DType = DType>> Model<R> for Llama<R> {
    fn from_varbuilder(vb: &mut crate::nn::VarBuilder<R>, config: &ModelConfig) -> Result<Self>
    where
        R::Client: crate::quant::DequantOps<R> + numr::ops::TypeConversionOps<R>,
    {
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

        // Embedding (dequantize if GGUF stored it as quantized)
        let embed_weight = model_vb.take_tensor_dequant("embed_tokens.weight", DType::F32)?;
        let embed_tokens = Embedding::new(embed_weight, false);

        // Transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let mut layers_vb = model_vb.pp("layers");
            let mut layer_vb = layers_vb.pp(&i.to_string());
            let block = build_block_from_varbuilder(
                &mut layer_vb,
                config,
                num_heads,
                num_kv_heads,
                head_dim,
            )?;
            layers.push(block);
        }

        // Final norm
        let norm = RmsNorm::new(
            model_vb.take_tensor("norm.weight")?,
            config.rms_norm_eps as f32,
            false,
        );

        // LM head (may be tied to embedding weights)
        let lm_head = if config.tie_word_embeddings {
            let embed_w = embed_tokens.weight().tensor().clone();
            MaybeQuantLinear::Standard(Linear::new(embed_w, None, false))
        } else {
            vb.take_maybe_quant_linear("lm_head.weight", None)?
        };

        // Pre-cast RoPE caches to match weight dtype (avoids per-token F32→BF16 casts)
        let mut rope = rope;
        if let Some(first_layer) = layers.first() {
            let weight_dtype = first_layer.input_layernorm.weight().tensor().dtype();
            rope.cast_caches(weight_dtype);
        }

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
            layers.push(build_block_from_config(
                config,
                device,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate,
                dt,
            ));
        }

        // Final norm
        let norm = RmsNorm::new(
            Tensor::<R>::ones(&[hidden], dt, device),
            config.rms_norm_eps as f32,
            true,
        );

        // LM head
        let lm_head = MaybeQuantLinear::Standard(Linear::new(
            Tensor::<R>::zeros(&[vocab, hidden], dt, device),
            None,
            true,
        ));

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

        // LM head: [B, S, hidden] -> [B, S, vocab]
        self.lm_head.forward(client, &hidden)
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

impl<R: Runtime<DType = DType>> Llama<R> {
    /// Extract the weight tensors (gate, up, down projections) for a single MoE expert.
    ///
    /// Returns `None` if `layer_idx` is out of range, the layer is dense (non-MoE),
    /// or `expert_id` is out of range.
    pub fn get_expert_weights(
        &self,
        layer_idx: usize,
        expert_id: usize,
    ) -> Option<super::blocks::ExpertWeights<R>> {
        use super::blocks::LlamaFfn;
        let block = self.layers.get(layer_idx)?;
        match &block.mlp {
            LlamaFfn::Moe(moe) => moe.get_expert_weights(expert_id),
            LlamaFfn::Dense(_) => None,
        }
    }

    /// Replace the weight tensors for a single MoE expert in-place.
    ///
    /// Returns an error if `layer_idx` is out of range, the layer is dense,
    /// `expert_id` is out of range, or an internal error occurs.
    pub fn set_expert_weights(
        &self,
        layer_idx: usize,
        expert_id: usize,
        weights: super::blocks::ExpertWeights<R>,
    ) -> crate::error::Result<()>
    where
        R::Client: ShapeOps<R>,
    {
        use super::blocks::LlamaFfn;
        use crate::error::Error;
        let block = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| Error::ModelError {
                reason: format!(
                    "layer_idx {} out of range (num_layers={})",
                    layer_idx,
                    self.layers.len()
                ),
            })?;
        match &block.mlp {
            LlamaFfn::Moe(moe) => moe.set_expert_weights(expert_id, weights),
            LlamaFfn::Dense(_) => Err(Error::ModelError {
                reason: format!("layer {} is a dense FFN layer, not MoE", layer_idx),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::LayeredKvCache;
    use crate::test_utils::cpu_setup;
    use numr::dtype::DType;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

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
    fn test_llama_forward_with_kv_cache_shape() {
        let (client, device) = cpu_setup();
        let config = tiny_config();
        let model = Llama::<CpuRuntime>::from_config(&config, &device).unwrap();

        let num_kv_heads = config.attention.as_ref().unwrap().kv_heads();
        let head_dim = config
            .attention
            .as_ref()
            .unwrap()
            .head_dim(config.hidden_size);

        let mut kv_cache = LayeredKvCache::<CpuRuntime>::new_positional(
            config.num_layers,
            1,
            num_kv_heads,
            16,
            config.max_seq_len,
            head_dim,
            DType::F32,
            &device,
        )
        .unwrap();

        // Prefill: 4 tokens at position 0
        let input_ids = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3], &[1, 4], &device);
        let logits = model
            .forward_with_kv_cache(&client, &input_ids, &mut kv_cache, 0)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 4, 32]);
        assert_eq!(kv_cache.seq_len(), 4);

        // Decode: 1 token at position 4
        let next_token = Tensor::<CpuRuntime>::from_slice(&[5i64], &[1, 1], &device);
        let logits = model
            .forward_with_kv_cache(&client, &next_token, &mut kv_cache, 4)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 1, 32]);
        assert_eq!(kv_cache.seq_len(), 5);

        // Decode another token at position 5
        let next_token = Tensor::<CpuRuntime>::from_slice(&[6i64], &[1, 1], &device);
        let logits = model
            .forward_with_kv_cache(&client, &next_token, &mut kv_cache, 5)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 1, 32]);
        assert_eq!(kv_cache.seq_len(), 6);
    }

    #[test]
    fn test_llama_kv_cache_gqa() {
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

        let num_kv_heads = config.attention.as_ref().unwrap().kv_heads();
        let head_dim = config
            .attention
            .as_ref()
            .unwrap()
            .head_dim(config.hidden_size);

        let mut kv_cache = LayeredKvCache::<CpuRuntime>::new_positional(
            config.num_layers,
            1,
            num_kv_heads,
            8,
            config.max_seq_len,
            head_dim,
            DType::F32,
            &device,
        )
        .unwrap();

        // Prefill
        let input_ids = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2], &[1, 3], &device);
        let logits = model
            .forward_with_kv_cache(&client, &input_ids, &mut kv_cache, 0)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 3, 32]);

        // Decode
        let next = Tensor::<CpuRuntime>::from_slice(&[3i64], &[1, 1], &device);
        let logits = model
            .forward_with_kv_cache(&client, &next, &mut kv_cache, 3)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 1, 32]);
        assert_eq!(kv_cache.seq_len(), 4);
    }

    #[test]
    fn test_swiglu_mlp() {
        use super::super::blocks::mlp::LlamaMlp;
        use crate::nn::MaybeQuantLinear;
        let (client, device) = cpu_setup();
        let mlp = LlamaMlp {
            gate_proj: MaybeQuantLinear::Standard(Linear::new(
                Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[2, 4], &device),
                None,
                false,
            )),
            up_proj: MaybeQuantLinear::Standard(Linear::new(
                Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[2, 4], &device),
                None,
                false,
            )),
            down_proj: MaybeQuantLinear::Standard(Linear::new(
                Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[4, 2], &device),
                None,
                false,
            )),
        };

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[1, 4], &device),
            false,
        );
        let out = mlp.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    fn tiny_alibi_config() -> ModelConfig {
        let yaml = r#"
model_type: falcon
vocab_size: 32
hidden_size: 16
num_layers: 2
max_seq_len: 32
intermediate_size: 32
rms_norm_eps: 1.0e-5
attention:
  num_heads: 2
  use_alibi: true
"#;
        serde_yaml::from_str(yaml).unwrap()
    }

    #[test]
    fn test_alibi_forward_shape() {
        let (client, device) = cpu_setup();
        let config = tiny_alibi_config();
        let model = Llama::<CpuRuntime>::from_config(&config, &device).unwrap();
        assert!(model.layers[0].self_attn.use_alibi);

        let input_ids = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3], &[1, 4], &device),
            false,
        );
        let logits = model.forward(&client, &input_ids).unwrap();
        assert_eq!(logits.shape(), &[1, 4, 32]);
    }

    #[test]
    fn test_alibi_kv_cache_shape() {
        let (client, device) = cpu_setup();
        let config = tiny_alibi_config();
        let model = Llama::<CpuRuntime>::from_config(&config, &device).unwrap();

        let num_kv_heads = config.attention.as_ref().unwrap().kv_heads();
        let head_dim = config
            .attention
            .as_ref()
            .unwrap()
            .head_dim(config.hidden_size);

        let mut kv_cache = LayeredKvCache::<CpuRuntime>::new_positional(
            config.num_layers,
            1,
            num_kv_heads,
            16,
            config.max_seq_len,
            head_dim,
            DType::F32,
            &device,
        )
        .unwrap();

        // Prefill: 4 tokens
        let input_ids = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3], &[1, 4], &device);
        let logits = model
            .forward_with_kv_cache(&client, &input_ids, &mut kv_cache, 0)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 4, 32]);
        assert_eq!(kv_cache.seq_len(), 4);

        // Decode: 1 token
        let next = Tensor::<CpuRuntime>::from_slice(&[5i64], &[1, 1], &device);
        let logits = model
            .forward_with_kv_cache(&client, &next, &mut kv_cache, 4)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 1, 32]);
        assert_eq!(kv_cache.seq_len(), 5);
    }
}
