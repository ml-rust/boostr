//! [`HybridModel`] forward passes and accessors.

use super::build::{HybridBlock, HybridModel};
use crate::error::{Error, Result};
use crate::inference::kv_cache::layered::LayeredKvCacheConfig;
use crate::inference::{KvCache, LayeredKvCache, LayeredSsmState};
use crate::model::traits::ModelClient;
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ConvOps, IndexingOps, NormalizationOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> HybridModel<R>
where
    R::Client: IndexingOps<R>,
{
    /// Forward pass with a KV cache for the attention layers and SSM state
    /// for the SSM layers. Returns the logits `[B, S, vocab]`.
    ///
    /// `kv_cache` holds [`num_attention_layers`](Self::num_attention_layers)
    /// layers shaped by [`kv_cache_dims`](Self::kv_cache_dims), and
    /// `ssm_state` holds [`num_ssm_layers`](Self::num_ssm_layers) layers.
    /// `position` is the absolute position of the first token, the cache
    /// length.
    pub fn forward_hybrid<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        ssm_state: &mut LayeredSsmState<R>,
        position: usize,
    ) -> Result<Tensor<R>>
    where
        C: ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + DequantOps<R>,
    {
        let hidden = self.run_layers(client, input_ids, kv_cache, Some(ssm_state), position)?;
        Ok(self.lm_head.forward(client, &hidden)?.tensor().clone())
    }

    /// Forward pass of a model with no SSM layer, such as an MLA model,
    /// with a KV cache alone. Returns the logits `[B, S, vocab]`.
    ///
    /// # Errors
    ///
    /// A model with an SSM layer: it needs [`forward_hybrid`](Self::forward_hybrid).
    pub fn forward_with_kv_cache<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        position: usize,
    ) -> Result<Tensor<R>>
    where
        C: ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + DequantOps<R>,
    {
        if self.num_ssm_layers() > 0 {
            return Err(Error::ModelError {
                reason: format!(
                    "this hybrid model has {} SSM layers, which keep recurrent state; run \
                     forward_hybrid with an SSM state",
                    self.num_ssm_layers()
                ),
            });
        }
        let hidden = self.run_layers(client, input_ids, kv_cache, None, position)?;
        Ok(self.lm_head.forward(client, &hidden)?.tensor().clone())
    }

    /// Embed, every layer, final norm. `ssm_state` is `None` only for a
    /// model with no SSM layer.
    fn run_layers<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        mut ssm_state: Option<&mut LayeredSsmState<R>>,
        position: usize,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + DequantOps<R>,
    {
        let mut hidden = self.embed_tokens.forward(client, input_ids)?;
        let mut attn_idx = 0usize;
        let mut ssm_idx = 0usize;
        for (i, block) in self.blocks.iter().enumerate() {
            hidden = match block {
                HybridBlock::Attention(b) => {
                    let cache = next_kv_layer(kv_cache, &mut attn_idx, i)?;
                    b.forward_with_kv_cache(client, &hidden, &self.rope, cache, position)?
                }
                HybridBlock::Mla(b) => {
                    let cache = next_kv_layer(kv_cache, &mut attn_idx, i)?;
                    b.forward_with_kv_cache(client, &hidden, cache, position)?
                }
                HybridBlock::Ssm(b) => {
                    let state = ssm_state
                        .as_deref_mut()
                        .and_then(|s| s.layer_mut(ssm_idx))
                        .ok_or_else(|| Error::ModelError {
                            reason: format!("SSM state missing for layer {i} (ssm_idx={ssm_idx})"),
                        })?;
                    ssm_idx += 1;
                    b.forward_inference(client, &hidden, state)?
                }
            };
        }
        self.norm.forward(client, &hidden)
    }

    /// Contextualized hidden states for embedding extraction.
    ///
    /// Runs embed + all attention/SSM blocks + final norm (no `lm_head`) with
    /// fresh throwaway KV cache and SSM state. Returns shape `[B, S, hidden]`.
    pub fn forward_hidden<C>(&self, client: &C, input_ids: &Tensor<R>) -> Result<Var<R>>
    where
        C: ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + DequantOps<R>,
    {
        let shape = input_ids.shape();
        let (batch, seq_len) = (shape[0], shape[1]);
        let device = input_ids.device();
        let dtype = self.embed_tokens.weight().tensor().dtype();
        let (num_kv_heads, head_dim) = self.kv_cache_dims().unwrap_or((1, 1));
        let kv_config = LayeredKvCacheConfig {
            batch_size: batch,
            num_kv_heads,
            initial_capacity: seq_len,
            max_seq_len: self.config.max_seq_len,
            head_dim,
            dtype,
        };
        let mut kv_cache =
            LayeredKvCache::<R>::new(self.num_attention_layers(), &kv_config, device)?;
        let mut ssm_state = match &self.mamba_config {
            Some(mamba) => Some(LayeredSsmState::<R>::new(
                self.num_ssm_layers(),
                batch,
                mamba,
                dtype,
                device,
            )?),
            None => None,
        };
        self.run_layers(client, input_ids, &mut kv_cache, ssm_state.as_mut(), 0)
    }
}

/// The KV cache of the next attention layer, layer `layer` of the model.
fn next_kv_layer<'a, R: Runtime<DType = DType>>(
    kv_cache: &'a mut LayeredKvCache<R>,
    attn_idx: &mut usize,
    layer: usize,
) -> Result<&'a mut KvCache<R>>
where
    R::Client: IndexingOps<R>,
{
    let idx = *attn_idx;
    *attn_idx += 1;
    kv_cache.layer_mut(idx).ok_or_else(|| Error::ModelError {
        reason: format!("KV cache missing for attention layer {layer} (attn_idx={idx})"),
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::super::blocks::FeedForward;
    use super::*;
    use crate::model::config::UniversalConfig;
    use crate::nn::{Embedding, Linear, Mla, MlaConfig, RmsNorm, VarBuilder, VarMap};
    use crate::test_utils::{Lcg, cpu_setup, max_abs_diff};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    type T = Tensor<CpuRuntime>;
    const TOL: f32 = 1e-4;

    /// An oxidizr-shaped config: `ssm[i]` makes layer `i` Mamba2, every
    /// other layer MLA, with a MoE or an MLP feed-forward.
    pub(crate) fn config(ssm: &[bool], moe: bool) -> UniversalConfig {
        let ssm_layers: Vec<usize> = (0..ssm.len()).filter(|&i| ssm[i]).collect();
        let attention_layers: Vec<usize> = (0..ssm.len()).filter(|&i| !ssm[i]).collect();
        let mut yaml = format!(
            "model_type: {}\nvocab_size: 24\nhidden_size: 16\nnum_layers: {}\n\
             max_seq_len: 16\nrms_norm_eps: 0.00001\n\
             attention:\n  num_heads: 2\n  kv_latent_dim: 8\n  q_latent_dim: 6\n  d_rope: 4\n",
            if ssm_layers.is_empty() {
                "llama"
            } else {
                "hybrid"
            },
            ssm.len()
        );
        if moe {
            yaml.push_str(
                "moe:\n  num_experts: 3\n  experts_per_tok: 2\n  shared_expert: true\n  \
                 intermediate_size: 12\n",
            );
        } else {
            yaml.push_str("intermediate_size: 20\n");
        }
        if !ssm_layers.is_empty() {
            yaml.push_str(&format!(
                "ssm:\n  variant: mamba2\n  num_heads: 2\n  head_dim: 16\n  state_size: 4\n  \
                 chunk_size: 4\n  n_groups: 1\n  conv_kernel: 4\n  expand: 2\n\
                 hybrid_layers:\n  ssm_layers: {ssm_layers:?}\n  attention_layers: {attention_layers:?}\n"
            ));
        }
        serde_saphyr::from_str(&yaml).unwrap()
    }

    /// Every tensor an oxidizr checkpoint of `config` holds, under its
    /// current names, with seeded values.
    pub(crate) fn checkpoint(
        config: &UniversalConfig,
        device: &CpuDevice,
    ) -> Vec<(String, Tensor<CpuRuntime>)> {
        let mut shapes: Vec<(String, Vec<usize>)> = vec![
            ("model.embed_tokens.weight".into(), vec![24, 16]),
            ("model.norm.weight".into(), vec![16]),
            ("lm_head.weight".into(), vec![24, 16]),
        ];
        for i in 0..config.num_layers {
            let p = format!("model.layers.{i}");
            let mut add =
                |name: &str, shape: &[usize]| shapes.push((format!("{p}.{name}"), shape.to_vec()));
            if config.is_ssm_layer(i) {
                add("norm.weight", &[16]);
                add("mamba2.in_proj.weight", &[74, 16]);
                add("mamba2.conv1d.weight", &[40, 1, 4]);
                add("mamba2.out_proj.weight", &[16, 32]);
                add("mamba2.A_log", &[2]);
                add("mamba2.dt_bias", &[2]);
                add("mamba2.D", &[2]);
                continue;
            }
            add("input_layernorm.weight", &[16]);
            add("post_attention_layernorm.weight", &[16]);
            add("self_attn.q_a_proj.weight", &[6, 16]);
            add("self_attn.q_a_layernorm.weight", &[6]);
            add("self_attn.q_b_proj.weight", &[24, 6]);
            add("self_attn.kv_a_proj_with_mqa.weight", &[12, 16]);
            add("self_attn.kv_a_layernorm.weight", &[8]);
            add("self_attn.kv_b_proj.weight", &[32, 8]);
            add("self_attn.o_proj.weight", &[16, 16]);
            if config.moe.is_some() {
                add("moe.router.gate.weight", &[3, 16]);
                for e in ["experts.0", "experts.1", "experts.2", "shared_expert"] {
                    add(&format!("moe.{e}.gate_proj.weight"), &[12, 16]);
                    add(&format!("moe.{e}.up_proj.weight"), &[12, 16]);
                    add(&format!("moe.{e}.down_proj.weight"), &[16, 12]);
                }
            } else {
                add("mlp.gate_proj.weight", &[20, 16]);
                add("mlp.up_proj.weight", &[20, 16]);
                add("mlp.down_proj.weight", &[16, 20]);
            }
        }
        let mut rng = Lcg(0x4d4c_4100);
        shapes
            .into_iter()
            .map(|(name, shape)| {
                let n: usize = shape.iter().product();
                let data: Vec<f32> = if shape.len() == 1 && !name.ends_with("A_log") {
                    rng.data(n, 0.2).iter().map(|v| 1.0 + v).collect()
                } else {
                    rng.data(n, 0.3)
                };
                (name, T::from_slice(&data, &shape, device).unwrap())
            })
            .collect()
    }

    pub(crate) fn map(tensors: &[(String, Tensor<CpuRuntime>)]) -> VarMap<CpuRuntime> {
        let mut map = VarMap::new();
        for (name, t) in tensors {
            map.insert(name.clone(), t.clone());
        }
        map
    }

    fn model(config: &UniversalConfig, tensors: &[(String, T)]) -> HybridModel<CpuRuntime> {
        let (_, dev) = cpu_setup();
        let mut map = map(tensors);
        HybridModel::from_varbuilder(&mut VarBuilder::new(&mut map, &dev), config).unwrap()
    }

    fn ids(n: usize) -> Vec<i64> {
        (0..n as i64).map(|i| (i * 7 + 3) % 24).collect()
    }

    /// Logits of the boostr training modules over the whole sequence: the
    /// unabsorbed `Mla::forward`, the MoE or MLP, the norms and residuals.
    fn reference(config: &UniversalConfig, tensors: &[(String, T)], tokens: &[i64]) -> Vec<f32> {
        let (client, dev) = cpu_setup();
        let mut map = map(tensors);
        let mut vb = VarBuilder::new(&mut map, &dev);
        let eps = config.rms_norm_eps as f32;
        let mla_config = MlaConfig::from_universal(config).unwrap();
        let input = T::from_slice(tokens, &[1, tokens.len()], &dev).unwrap();
        let mut model_vb = vb.pp("model");
        let embed = Embedding::new(model_vb.take_tensor("embed_tokens.weight").unwrap(), false);
        let mut x = embed.forward(&client, &input).unwrap();
        for i in 0..config.num_layers {
            let mut layers = model_vb.pp("layers");
            let mut layer = layers.pp(&i.to_string());
            let norm = |t| RmsNorm::new(t, eps, false);
            let in_norm = norm(layer.take_tensor("input_layernorm.weight").unwrap());
            let post_norm = norm(
                layer
                    .take_tensor("post_attention_layernorm.weight")
                    .unwrap(),
            );
            let mla = Mla::from_varbuilder(&mut layer.pp("self_attn"), &mla_config).unwrap();
            let ffn = FeedForward::load(&mut layer, config).unwrap();
            let attn = mla
                .forward(&client, &in_norm.forward(&client, &x).unwrap())
                .unwrap();
            let h = numr::autograd::var_add(&x, &attn, &client).unwrap();
            let f = ffn
                .forward(&client, &post_norm.forward(&client, &h).unwrap())
                .unwrap();
            x = numr::autograd::var_add(&h, &f, &client).unwrap();
        }
        let norm = RmsNorm::new(model_vb.take_tensor("norm.weight").unwrap(), eps, false);
        let x = norm.forward(&client, &x).unwrap();
        let head = Linear::new(vb.take_tensor("lm_head.weight").unwrap(), None, false);
        head.forward(&client, &x).unwrap().tensor().to_vec::<f32>()
    }

    /// Logits of `model` over `tokens`, fed as consecutive chunks.
    fn served(model: &HybridModel<CpuRuntime>, tokens: &[i64], chunks: &[usize]) -> Vec<f32> {
        let (client, dev) = cpu_setup();
        let (heads, width) = model.kv_cache_dims().unwrap();
        let kv_config = LayeredKvCacheConfig {
            batch_size: 1,
            num_kv_heads: heads,
            initial_capacity: 4,
            max_seq_len: 16,
            head_dim: width,
            dtype: DType::F32,
        };
        let mut kv = LayeredKvCache::new(model.num_attention_layers(), &kv_config, &dev).unwrap();
        let mut ssm = model
            .mamba_config()
            .map(|m| LayeredSsmState::new(model.num_ssm_layers(), 1, m, DType::F32, &dev).unwrap());
        let mut out = Vec::new();
        let mut at = 0;
        for &n in chunks {
            let piece = T::from_slice(&tokens[at..at + n], &[1, n], &dev).unwrap();
            let logits = match ssm.as_mut() {
                Some(state) => model.forward_hybrid(&client, &piece, &mut kv, state, at),
                None => model.forward_with_kv_cache(&client, &piece, &mut kv, at),
            };
            out.extend(logits.unwrap().to_vec::<f32>());
            at += n;
        }
        assert_eq!(at, tokens.len());
        out
    }

    fn assert_close(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let diff = max_abs_diff(got, want);
        assert!(diff <= TOL, "{what}: max diff {diff}");
    }

    #[test]
    fn mla_moe_model_logits_match_the_training_modules() {
        let (_, dev) = cpu_setup();
        let config = config(&[false, false], true);
        let tensors = checkpoint(&config, &dev);
        let m = model(&config, &tensors);
        assert!(m.is_mla() && m.mamba_config().is_none());
        assert_eq!(m.kv_cache_dims(), Some((1, 12)));
        assert_eq!((m.num_attention_layers(), m.num_ssm_layers()), (2, 0));
        let tokens = ids(9);
        let want = reference(&config, &tensors, &tokens);
        assert_close(&served(&m, &tokens, &[9]), &want, "one shot");
        assert_close(&served(&m, &tokens, &[1; 9]), &want, "token by token");
        assert_close(
            &served(&m, &tokens, &[5, 1, 2, 1]),
            &want,
            "prefill then decode",
        );
    }

    #[test]
    fn mla_mlp_model_decode_matches_the_training_modules() {
        let (_, dev) = cpu_setup();
        let config = config(&[false, false, false], false);
        let tensors = checkpoint(&config, &dev);
        let m = model(&config, &tensors);
        let tokens = ids(7);
        let want = reference(&config, &tensors, &tokens);
        assert_close(&served(&m, &tokens, &[1; 7]), &want, "token by token");
    }

    #[test]
    fn mamba2_mla_moe_hybrid_decode_matches_its_prefill() {
        let (_, dev) = cpu_setup();
        // nano-start's layout: Mamba2 layers, then one MLA + MoE layer.
        let config = config(&[true, true, false], true);
        let tensors = checkpoint(&config, &dev);
        let m = model(&config, &tensors);
        assert_eq!((m.num_attention_layers(), m.num_ssm_layers()), (1, 2));
        assert!(m.mamba_config().is_some());
        let tokens = ids(8);
        let one_shot = served(&m, &tokens, &[8]);
        assert!(one_shot.iter().all(|v| v.is_finite()));
        assert_close(&served(&m, &tokens, &[1; 8]), &one_shot, "token by token");
        assert_close(
            &served(&m, &tokens, &[5, 1, 1, 1]),
            &one_shot,
            "prefill then decode",
        );

        let (client, _) = cpu_setup();
        let input = T::from_slice(&tokens, &[1, 8], &dev).unwrap();
        let mut kv = LayeredKvCache::new(
            1,
            &LayeredKvCacheConfig {
                batch_size: 1,
                num_kv_heads: 1,
                initial_capacity: 8,
                max_seq_len: 16,
                head_dim: 12,
                dtype: DType::F32,
            },
            &dev,
        )
        .unwrap();
        let err = m
            .forward_with_kv_cache(&client, &input, &mut kv, 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("forward_hybrid"), "{err}");
    }

    #[test]
    fn forward_hidden_runs_an_mla_model() {
        let (client, dev) = cpu_setup();
        let config = config(&[true, false], false);
        let tensors = checkpoint(&config, &dev);
        let m = model(&config, &tensors);
        let input = T::from_slice(&ids(5), &[1, 5], &dev).unwrap();
        let hidden = m.forward_hidden(&client, &input).unwrap();
        assert_eq!(hidden.shape(), &[1, 5, 16]);
    }
}
