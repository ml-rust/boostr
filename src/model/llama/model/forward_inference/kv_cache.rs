//! Forward pass for inference with a flat (non-paged) KV cache.

use super::super::Llama;
use crate::error::{Error, Result};
use crate::inference::LayeredKvCache;
use crate::model::traits::ModelClient;
use crate::quant::traits::DequantOps;
use numr::autograd::var_add;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = numr::dtype::DType>> Llama<R> {
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
            + ConditionalOps<R>
            + DequantOps<R>,
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
        let mut prev_mlp_out: Option<numr::autograd::Var<R>> = None;
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
}

#[cfg(test)]
mod tests {
    use super::super::super::Llama;
    use crate::inference::LayeredKvCache;
    use crate::model::config::ModelConfig;
    use crate::model::traits::Model;
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
        serde_saphyr::from_str(yaml).unwrap()
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
        serde_saphyr::from_str(yaml).unwrap()
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
        let input_ids =
            Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3], &[1, 4], &device).unwrap();
        let logits = model
            .forward_with_kv_cache(&client, &input_ids, &mut kv_cache, 0)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 4, 32]);
        assert_eq!(kv_cache.seq_len(), 4);

        // Decode: 1 token at position 4
        let next_token = Tensor::<CpuRuntime>::from_slice(&[5i64], &[1, 1], &device).unwrap();
        let logits = model
            .forward_with_kv_cache(&client, &next_token, &mut kv_cache, 4)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 1, 32]);
        assert_eq!(kv_cache.seq_len(), 5);

        // Decode another token at position 5
        let next_token = Tensor::<CpuRuntime>::from_slice(&[6i64], &[1, 1], &device).unwrap();
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
        let config: ModelConfig = serde_saphyr::from_str(yaml).unwrap();
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
        let input_ids = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2], &[1, 3], &device).unwrap();
        let logits = model
            .forward_with_kv_cache(&client, &input_ids, &mut kv_cache, 0)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 3, 32]);

        // Decode
        let next = Tensor::<CpuRuntime>::from_slice(&[3i64], &[1, 1], &device).unwrap();
        let logits = model
            .forward_with_kv_cache(&client, &next, &mut kv_cache, 3)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 1, 32]);
        assert_eq!(kv_cache.seq_len(), 4);
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
        let input_ids =
            Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3], &[1, 4], &device).unwrap();
        let logits = model
            .forward_with_kv_cache(&client, &input_ids, &mut kv_cache, 0)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 4, 32]);
        assert_eq!(kv_cache.seq_len(), 4);

        // Decode: 1 token
        let next = Tensor::<CpuRuntime>::from_slice(&[5i64], &[1, 1], &device).unwrap();
        let logits = model
            .forward_with_kv_cache(&client, &next, &mut kv_cache, 4)
            .unwrap();
        assert_eq!(logits.shape(), &[1, 1, 32]);
        assert_eq!(kv_cache.seq_len(), 5);
    }
}
