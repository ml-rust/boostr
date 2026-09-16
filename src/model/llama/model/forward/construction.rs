//! `Llama`'s `Model` trait implementation: construction from a checkpoint or
//! from scratch, and the training forward pass.

use super::types::Llama;
use crate::error::{Error, Result};
use crate::model::config::ModelConfig;
use crate::model::llama::model::blocks::{build_block_from_config, build_block_from_varbuilder};
use crate::model::llama::model::lm_head::build_lm_head;
use crate::model::traits::{Model, ModelClient};
use crate::model::vocab_growth::fit_vocab_rows;
use crate::nn::{Embedding, Linear, MaybeQuantLinear, RmsNorm, RoPE};
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Model<R> for Llama<R> {
    fn from_varbuilder(vb: &mut crate::nn::VarBuilder<R>, config: &ModelConfig) -> Result<Self>
    where
        R::Client: crate::quant::DequantOps<R>
            + numr::ops::TypeConversionOps<R>
            + ReduceOps<R>
            + ShapeOps<R>,
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
        )?;

        let mut model_vb = vb.pp("model");

        // Embedding (dequantize if GGUF stored it as quantized)
        // The checkpoint key is captured here, from the builder that reads
        // it, and handed to `build_lm_head`: a TIED head multiplies by this
        // very tensor and an importance collection records it under this
        // name. Reading it off the builder is what stops the name from
        // becoming a second copy that can drift from the loader.
        let embed_name = model_vb.full_name("embed_tokens.weight");
        let embed_weight = model_vb.take_tensor_dequant("embed_tokens.weight", DType::F32)?;
        // `model_vb.device()` here, not `vb.device()`: `vb` is mutably borrowed by
        // `model_vb` for the rest of this scope.
        let embed_weight = fit_vocab_rows(
            embed_weight,
            config,
            model_vb.device(),
            "embed_tokens.weight",
        )?;
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
        let lm_head = build_lm_head(vb, config, &embed_tokens, &embed_name)?;

        // Pre-cast RoPE caches to match weight dtype (avoids per-token F32→BF16 casts)
        let mut rope = rope;
        if let Some(first_layer) = layers.first() {
            let weight_dtype = first_layer.input_layernorm.weight().tensor().dtype();
            rope.cast_caches(weight_dtype)?;
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
        let embed_weight = Tensor::<R>::zeros(&[vocab, hidden], dt, device)?;
        let embed_tokens = Embedding::new(embed_weight, true);

        // RoPE cache
        let rope = RoPE::<R>::precompute_freqs(
            config.max_seq_len,
            head_dim,
            attn_cfg.rope_theta,
            attn_cfg.rope_scaling.as_ref(),
            device,
        )?;

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
            )?);
        }

        // Final norm
        let norm = RmsNorm::new(
            Tensor::<R>::ones(&[hidden], dt, device)?,
            config.rms_norm_eps as f32,
            true,
        );

        // LM head. When tied, it must share the embedding's autograd identity,
        // not just its values: autograd accumulates by `TensorId`, so a shared
        // id sums the gradient arriving through the embedding lookup and
        // through the output projection, and an optimizer keyed by `TensorId`
        // updates the one weight exactly once.
        //
        // `Tensor::clone` mints a FRESH `TensorId`, so passing the embedding's
        // tensor to `Linear::new` would produce a second independent leaf that
        // duplicates the values but not the identity — the head would train
        // separately from the embedding while appearing tied.
        let lm_head = if config.tie_word_embeddings {
            MaybeQuantLinear::Standard(Linear::with_ids(
                embed_tokens.weight().tensor().clone(),
                embed_tokens.weight().id(),
                None,
                true,
            ))
        } else {
            MaybeQuantLinear::Standard(Linear::new(
                Tensor::<R>::zeros(&[vocab, hidden], dt, device)?,
                None,
                true,
            ))
        };

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
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        let hidden = self.forward_hidden(client, input_ids.tensor())?;
        self.lm_head.forward(client, &hidden)
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::tiny_config;
    use super::*;
    use crate::nn::MaybeQuantLinear;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    #[test]
    fn test_llama_from_config() {
        let (_, device) = cpu_setup();
        let config = tiny_config();
        let model = Llama::<CpuRuntime>::from_config(&config, &device).unwrap();
        assert_eq!(model.layers.len(), 2);
    }

    fn lm_head_id(model: &Llama<CpuRuntime>) -> numr::tensor::TensorId {
        match &model.lm_head {
            MaybeQuantLinear::Standard(linear) => linear.weight().id(),
            _ => panic!("from_config always builds a standard lm_head"),
        }
    }

    /// A tied head must share the embedding's autograd IDENTITY, not just its
    /// values. Autograd accumulates by `TensorId`, so only a shared id sums the
    /// gradient from both the embedding lookup and the output projection, and
    /// only then does a `TensorId`-keyed optimizer update the weight once.
    #[test]
    fn tied_lm_head_shares_the_embedding_tensor_id() {
        let (_, device) = cpu_setup();
        let mut config = tiny_config();
        config.tie_word_embeddings = true;
        let model = Llama::<CpuRuntime>::from_config(&config, &device).unwrap();
        assert_eq!(model.embed_tokens.weight().id(), lm_head_id(&model));
    }

    /// The discriminating half: without this, the test above would also pass on
    /// an implementation that gave every weight the same id.
    #[test]
    fn untied_lm_head_has_its_own_tensor_id() {
        let (_, device) = cpu_setup();
        let mut config = tiny_config();
        config.tie_word_embeddings = false;
        let model = Llama::<CpuRuntime>::from_config(&config, &device).unwrap();
        assert_ne!(model.embed_tokens.weight().id(), lm_head_id(&model));
    }

    #[test]
    fn test_llama_forward_shape() {
        let (client, device) = cpu_setup();
        let config = tiny_config();
        let model = Llama::<CpuRuntime>::from_config(&config, &device).unwrap();

        let input_ids = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3], &[1, 4], &device).unwrap(),
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
        let config: ModelConfig = serde_saphyr::from_str(yaml).unwrap();
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
        let config: ModelConfig = serde_saphyr::from_str(yaml).unwrap();
        let model = Llama::<CpuRuntime>::from_config(&config, &device).unwrap();

        let input_ids = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2], &[1, 3], &device).unwrap(),
            false,
        );
        let logits = model.forward(&client, &input_ids).unwrap();
        assert_eq!(logits.shape(), &[1, 3, 32]);
    }

    #[test]
    fn test_swiglu_mlp() {
        use crate::model::llama::model::blocks::mlp::LlamaMlp;
        use crate::nn::MaybeQuantLinear;
        let (client, device) = cpu_setup();
        let mlp = LlamaMlp {
            gate_proj: MaybeQuantLinear::Standard(Linear::new(
                Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[2, 4], &device).unwrap(),
                None,
                false,
            )),
            up_proj: MaybeQuantLinear::Standard(Linear::new(
                Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[2, 4], &device).unwrap(),
                None,
                false,
            )),
            down_proj: MaybeQuantLinear::Standard(Linear::new(
                Tensor::<CpuRuntime>::from_slice(&[0.1f32; 8], &[4, 2], &device).unwrap(),
                None,
                false,
            )),
        };

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[1, 4], &device).unwrap(),
            false,
        );
        let out = mlp.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[1, 4]);
    }

    #[test]
    fn test_alibi_forward_shape() {
        let config = super::super::test_support::tiny_alibi_config();
        let (client, device) = cpu_setup();
        let model = Llama::<CpuRuntime>::from_config(&config, &device).unwrap();
        assert!(model.layers[0].self_attn.use_alibi);

        let input_ids = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3], &[1, 4], &device).unwrap(),
            false,
        );
        let logits = model.forward(&client, &input_ids).unwrap();
        assert_eq!(logits.shape(), &[1, 4, 32]);
    }
}
