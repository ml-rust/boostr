//! LLaMA model: struct, training forward pass, and inference with KV cache.

use super::blocks::{LlamaBlock, build_block_from_config, build_block_from_varbuilder};
use crate::error::{Error, Result};
use crate::inference::LayeredKvCache;
use crate::inference::kv_cache::LayeredPagedKvCache;
use crate::model::config::ModelConfig;
use crate::model::traits::{Model, ModelClient};
use crate::nn::{Embedding, Linear, MaybeQuantLinear, RmsNorm, RoPE};
use crate::ops::traits::{KvCacheOps, PagedAttentionOps};
use numr::autograd::{Var, var_add};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Full LLaMA model
pub struct Llama<R: Runtime> {
    config: ModelConfig,
    embed_tokens: Embedding<R>,
    layers: Vec<LlamaBlock<R>>,
    norm: RmsNorm<R>,
    lm_head: MaybeQuantLinear<R>,
    rope: RoPE<R>,
}

impl<R: Runtime<DType = DType>> Model<R> for Llama<R> {
    fn from_varbuilder(vb: &mut crate::nn::VarBuilder<R>, config: &ModelConfig) -> Result<Self>
    where
        R::Client: crate::quant::DequantOps<R>,
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

// ── Inference (KV cache) ────────────────────────────────────────────

impl<R: Runtime<DType = DType>> Llama<R> {
    /// Forward pass for inference with KV cache.
    ///
    /// Unlike `Model::forward`, this:
    /// - Accepts `Tensor<R>` input (no autograd overhead)
    /// - Uses a KV cache for efficient autoregressive decoding
    /// - Takes a position offset for RoPE
    ///
    /// # Arguments
    /// * `client` - Runtime client
    /// * `input_ids` - Token IDs `[B, S]`
    /// * `kv_cache` - Layered KV cache (one per transformer layer)
    /// * `position` - RoPE position offset (= number of previously decoded tokens)
    ///
    /// # Returns
    /// Logits `[B, S, vocab_size]`
    pub fn forward_with_kv_cache<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        position: usize,
    ) -> Result<Tensor<R>>
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
        let profile = std::env::var("BLAZR_PROFILE").is_ok();
        let device = input_ids.device();
        let rc = R::default_client(device);

        macro_rules! sync_log {
            ($t:expr, $msg:expr) => {
                if profile {
                    rc.synchronize();
                    eprintln!("[profile] {}: {:?}", $msg, $t.elapsed());
                }
            };
        }

        let t = std::time::Instant::now();

        // Embed tokens: [B, S] -> [B, S, hidden]
        let mut hidden = self.embed_tokens.forward(client, input_ids)?;
        sync_log!(t, "embed");

        // Transformer blocks with KV cache — deferred residual add fusion
        let mut prev_mlp_out: Option<Var<R>> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let t_layer = std::time::Instant::now();
            let cache = kv_cache.layer_mut(i).ok_or_else(|| Error::ModelError {
                reason: format!("KV cache missing for layer {i}"),
            })?;
            let (h, mlp_out) = layer.forward_with_kv_cache(
                client,
                &hidden,
                prev_mlp_out.as_ref(),
                &self.rope,
                cache,
                position,
            )?;
            hidden = h;
            prev_mlp_out = Some(mlp_out);
            sync_log!(t_layer, format!("layer {i}"));
        }

        // Final residual add (deferred from last layer) + norm
        let t_norm = std::time::Instant::now();
        if let Some(last_mlp) = prev_mlp_out {
            hidden = var_add(&hidden, &last_mlp, client).map_err(Error::Numr)?;
        }
        hidden = self.norm.forward(client, &hidden)?;
        sync_log!(t_norm, "norm");

        // LM head: [B, S, hidden] -> [B, S, vocab]
        let t_lm = std::time::Instant::now();
        let logits = self.lm_head.forward(client, &hidden)?;
        sync_log!(t_lm, "lm_head");

        if profile {
            eprintln!("[profile] total forward: {:?}", t.elapsed());
        }

        Ok(logits.tensor().clone())
    }

    /// Return a reference to the model's RoPE module (for cos/sin cache access).
    pub fn rope(&self) -> &crate::nn::RoPE<R> {
        &self.rope
    }

    /// Forward pass for inference with paged KV cache.
    ///
    /// Uses PagedAttention with block table indirection instead of contiguous KV cache.
    ///
    /// # Arguments
    /// * `client` - Runtime client
    /// * `input_ids` - Token IDs `[B, S]`
    /// * `paged_cache` - Layered paged KV cache
    /// * `slot_mapping` - Slot mapping tensor `[B*S]` (I32)
    /// * `block_table` - Block table tensor `[B, max_num_blocks]` (I32)
    /// * `seq_len_k` - Total KV sequence length (including new tokens)
    /// * `position` - RoPE position offset
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_paged_kv_cache<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        paged_cache: &LayeredPagedKvCache<R>,
        slot_mapping: &Tensor<R>,
        block_table: &Tensor<R>,
        seq_len_k: usize,
        position: usize,
    ) -> Result<Tensor<R>>
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
            + KvCacheOps<R>
            + PagedAttentionOps<R>,
    {
        // Embed tokens: [B, S] -> [B, S, hidden]
        let mut hidden = self.embed_tokens.forward(client, input_ids)?;

        // Transformer blocks with paged KV cache — deferred residual add fusion
        let mut prev_mlp_out: Option<Var<R>> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let (h, mlp_out) = layer.forward_with_paged_kv_cache(
                client,
                &hidden,
                prev_mlp_out.as_ref(),
                &self.rope,
                paged_cache,
                i,
                slot_mapping,
                block_table,
                seq_len_k,
                position,
            )?;
            hidden = h;
            prev_mlp_out = Some(mlp_out);
        }

        // Final residual add (deferred from last layer) + norm
        if let Some(last_mlp) = prev_mlp_out {
            hidden = var_add(&hidden, &last_mlp, client).map_err(Error::Numr)?;
        }
        hidden = self.norm.forward(client, &hidden)?;

        // LM head: [B, S, hidden] -> [B, S, vocab]
        let logits = self.lm_head.forward(client, &hidden)?;

        Ok(logits.tensor().clone())
    }
}

// ── CUDA graph-mode forward ──────────────────────────────────────────

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
        input_ids: &Tensor<numr::runtime::cuda::CudaRuntime>,
        kv_cache: &mut LayeredKvCache<numr::runtime::cuda::CudaRuntime>,
        device_scalars: &crate::inference::decode_graph::DeviceScalars,
        cos_slice: &numr::autograd::Var<numr::runtime::cuda::CudaRuntime>,
        sin_slice: &numr::autograd::Var<numr::runtime::cuda::CudaRuntime>,
    ) -> Result<Tensor<numr::runtime::cuda::CudaRuntime>>
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
        use super::super::blocks::LlamaMlp;
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
}
