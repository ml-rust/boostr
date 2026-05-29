//! Constructors for `Encoder`: f32 weights and quantized weights.

use crate::error::{Error, Result};
use crate::model::encoder::config::EncoderConfig;
use crate::model::encoder::model::layer::EncoderLayer;
use crate::model::encoder::model::{Encoder, Pooling};
use crate::nn::{Embedding, LayerNorm, Linear, MaybeQuantLinear, Weight};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Encoder<R> {
    /// Create an encoder from pre-loaded f32 weight tensors.
    ///
    /// `get` is a closure that fetches tensors by HuggingFace-style name.
    /// All projection weights are stored as `Linear<R>` (full-precision).
    pub fn from_weights<F>(config: EncoderConfig, pooling: Pooling, mut get: F) -> Result<Self>
    where
        F: FnMut(&str) -> Result<Tensor<R>>,
    {
        let token_embed = Embedding::new(get("embeddings.word_embeddings.weight")?, false);
        let position_embed = Embedding::new(get("embeddings.position_embeddings.weight")?, false);
        let embed_norm = LayerNorm::new(
            get("embeddings.layer_norm.weight")?,
            get("embeddings.layer_norm.bias")?,
            config.layer_norm_eps as f32,
            false,
        );

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("encoder.layer.{i}");

            let q_proj = MaybeQuantLinear::Standard(Linear::new(
                get(&format!("{p}.attention.self.query.weight"))?,
                Some(get(&format!("{p}.attention.self.query.bias"))?),
                false,
            ));
            let k_proj = MaybeQuantLinear::Standard(Linear::new(
                get(&format!("{p}.attention.self.key.weight"))?,
                Some(get(&format!("{p}.attention.self.key.bias"))?),
                false,
            ));
            let v_proj = MaybeQuantLinear::Standard(Linear::new(
                get(&format!("{p}.attention.self.value.weight"))?,
                Some(get(&format!("{p}.attention.self.value.bias"))?),
                false,
            ));
            let o_proj = MaybeQuantLinear::Standard(Linear::new(
                get(&format!("{p}.attention.output.dense.weight"))?,
                Some(get(&format!("{p}.attention.output.dense.bias"))?),
                false,
            ));
            let attn_norm = LayerNorm::new(
                get(&format!("{p}.attention.output.LayerNorm.weight"))?,
                get(&format!("{p}.attention.output.LayerNorm.bias"))?,
                config.layer_norm_eps as f32,
                false,
            );
            let ffn_up = MaybeQuantLinear::Standard(Linear::new(
                get(&format!("{p}.intermediate.dense.weight"))?,
                Some(get(&format!("{p}.intermediate.dense.bias"))?),
                false,
            ));
            let ffn_down = MaybeQuantLinear::Standard(Linear::new(
                get(&format!("{p}.output.dense.weight"))?,
                Some(get(&format!("{p}.output.dense.bias"))?),
                false,
            ));
            let ffn_norm = LayerNorm::new(
                get(&format!("{p}.output.LayerNorm.weight"))?,
                get(&format!("{p}.output.LayerNorm.bias"))?,
                config.layer_norm_eps as f32,
                false,
            );

            layers.push(EncoderLayer {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                attn_norm,
                ffn_up,
                ffn_down,
                ffn_norm,
                num_heads: config.num_attention_heads,
                head_dim: config.head_dim(),
                hidden_act: config.hidden_act,
            });
        }

        Ok(Encoder {
            config,
            token_embed,
            position_embed,
            embed_norm,
            layers,
            pooling,
            #[cfg(feature = "cuda")]
            forward_cache: std::sync::Arc::new(
                crate::model::encoder::model::graph_cache::EncoderForwardCache::new(),
            ),
        })
    }

    /// Create an encoder from quantized GGUF weights.
    ///
    /// The six projection matrices per layer (q, k, v, o, ffn_up, ffn_down) are
    /// loaded quantized and kept compressed in device memory.
    /// Embeddings and LayerNorms remain in f32.
    ///
    /// A single `get` closure handles all tensor names and returns `Weight<R>`:
    /// - Return `Weight::Quantized(qt)` for projection weight names.
    /// - Return `Weight::Standard(t)` for embeddings, norms, and biases.
    ///
    /// The single-closure design is intentional: it lets callers hold exactly one
    /// `&mut` borrow of their reader (e.g. `Gguf`), since Rust cannot alias two
    /// simultaneous `&mut` borrows across separate closures.
    ///
    /// Bias tensors are optional. For names ending in `.bias`, returning
    /// `Err(_)` is treated as "no bias" — the projection will be bias-free.
    pub fn from_weights_quant<G>(
        config: EncoderConfig,
        pooling: Pooling,
        mut get: G,
    ) -> Result<Self>
    where
        G: FnMut(&str) -> Result<Weight<R>>,
    {
        let f32_weight = |w: Weight<R>, name: &str| -> Result<Tensor<R>> {
            w.as_tensor().cloned().map_err(|_| Error::ModelError {
                reason: format!("expected f32 tensor for '{name}', got quantized"),
            })
        };

        let token_embed = Embedding::new(
            f32_weight(
                get("embeddings.word_embeddings.weight")?,
                "embeddings.word_embeddings.weight",
            )?,
            false,
        );
        let position_embed = Embedding::new(
            f32_weight(
                get("embeddings.position_embeddings.weight")?,
                "embeddings.position_embeddings.weight",
            )?,
            false,
        );
        let embed_norm = LayerNorm::new(
            f32_weight(
                get("embeddings.layer_norm.weight")?,
                "embeddings.layer_norm.weight",
            )?,
            f32_weight(
                get("embeddings.layer_norm.bias")?,
                "embeddings.layer_norm.bias",
            )?,
            config.layer_norm_eps as f32,
            false,
        );

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("encoder.layer.{i}");

            let load_proj = |get: &mut G,
                             weight_name: &str,
                             bias_name: &str|
             -> Result<(Weight<R>, Option<Tensor<R>>)> {
                let weight = get(weight_name)?;
                let bias = match get(bias_name) {
                    Ok(Weight::Standard(t)) => Some(t),
                    _ => None,
                };
                Ok((weight, bias))
            };

            let (qw, qb) = load_proj(
                &mut get,
                &format!("{p}.attention.self.query.weight"),
                &format!("{p}.attention.self.query.bias"),
            )?;
            let q_proj = MaybeQuantLinear::from_weight(qw, qb);

            let (kw, kb) = load_proj(
                &mut get,
                &format!("{p}.attention.self.key.weight"),
                &format!("{p}.attention.self.key.bias"),
            )?;
            let k_proj = MaybeQuantLinear::from_weight(kw, kb);

            let (vw, vb) = load_proj(
                &mut get,
                &format!("{p}.attention.self.value.weight"),
                &format!("{p}.attention.self.value.bias"),
            )?;
            let v_proj = MaybeQuantLinear::from_weight(vw, vb);

            let (ow, ob) = load_proj(
                &mut get,
                &format!("{p}.attention.output.dense.weight"),
                &format!("{p}.attention.output.dense.bias"),
            )?;
            let o_proj = MaybeQuantLinear::from_weight(ow, ob);

            let attn_norm = LayerNorm::new(
                f32_weight(
                    get(&format!("{p}.attention.output.LayerNorm.weight"))?,
                    &format!("{p}.attention.output.LayerNorm.weight"),
                )?,
                f32_weight(
                    get(&format!("{p}.attention.output.LayerNorm.bias"))?,
                    &format!("{p}.attention.output.LayerNorm.bias"),
                )?,
                config.layer_norm_eps as f32,
                false,
            );

            let (uw, ub) = load_proj(
                &mut get,
                &format!("{p}.intermediate.dense.weight"),
                &format!("{p}.intermediate.dense.bias"),
            )?;
            let ffn_up = MaybeQuantLinear::from_weight(uw, ub);

            let (dw, db) = load_proj(
                &mut get,
                &format!("{p}.output.dense.weight"),
                &format!("{p}.output.dense.bias"),
            )?;
            let ffn_down = MaybeQuantLinear::from_weight(dw, db);

            let ffn_norm = LayerNorm::new(
                f32_weight(
                    get(&format!("{p}.output.LayerNorm.weight"))?,
                    &format!("{p}.output.LayerNorm.weight"),
                )?,
                f32_weight(
                    get(&format!("{p}.output.LayerNorm.bias"))?,
                    &format!("{p}.output.LayerNorm.bias"),
                )?,
                config.layer_norm_eps as f32,
                false,
            );

            layers.push(EncoderLayer {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                attn_norm,
                ffn_up,
                ffn_down,
                ffn_norm,
                num_heads: config.num_attention_heads,
                head_dim: config.head_dim(),
                hidden_act: config.hidden_act,
            });
        }

        Ok(Encoder {
            config,
            token_embed,
            position_embed,
            embed_norm,
            layers,
            pooling,
            #[cfg(feature = "cuda")]
            forward_cache: std::sync::Arc::new(
                crate::model::encoder::model::graph_cache::EncoderForwardCache::new(),
            ),
        })
    }
}
