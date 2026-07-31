//! BERT / XLM-RoBERTa encoder constructors: f32 weights and quantized weights.

use crate::error::{Error, Result};
use crate::model::encoder::config::{EncoderConfig, FfnVariant, QkNormScope};
use crate::model::encoder::model::layer::{EncoderLayer, NormLayer};
use crate::model::encoder::model::{Encoder, Pooling};
use crate::nn::{Embedding, LayerNorm, Linear, MaybeQuantLinear, Weight};
use crate::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::{Runtime, RuntimeClient};
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

        // Token-type ("segment") embedding, row 0.
        //
        // BERT adds a segment vector to every token, and single-segment
        // inference always uses row 0. Omitting it shifts every token by the
        // same learned constant before the first block — which changes the
        // pooled vector without changing a single shape. Measured against
        // llama.cpp on all-MiniLM-L6-v2, dropping it moved the sentence
        // embedding to cosine 0.85; restoring it gives 0.999+.
        //
        // Optional: a checkpoint that carries no such table (some distills, and
        // the SafeTensors callers that share this constructor) simply has none.
        let token_type_embed =
            token_type_row0(get("embeddings.token_type_embeddings.weight").ok(), &config)?;

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
                attn_norm: NormLayer::LayerNorm(attn_norm),
                ffn_up,
                ffn_gate: None,
                ffn_down,
                ffn_norm: NormLayer::LayerNorm(ffn_norm),
                num_heads: config.num_attention_heads,
                num_kv_heads: config.num_attention_heads,
                head_dim: config.head_dim(),
                hidden_act: config.hidden_act,
                ffn_variant: FfnVariant::Standard,
                norm_scheme: config.norm_scheme,
                attn: config.layer_attention(i),
                rope: None,
                q_norm: None,
                k_norm: None,
                qk_norm_scope: QkNormScope::PerHead,
                attn_norm_2: None,
                post_attn_norm: None,
                post_ffn_norm: None,
            });
        }

        Ok(Encoder {
            config,
            token_embed,
            position_embed,
            embed_norm: Some(embed_norm),
            layers,
            pooling,
            token_type_embed,
            output_norm: None,
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
    ///
    /// When `config.compute_dtype == DType::F16`, the `client` is used to:
    ///
    /// - Dequantize projection `QuantTensor`s to F16 (via `DequantOps::dequantize`)
    ///   so `quant_matmul` is never called on the F16 path.
    /// - Cast all F32 embedding tables, LayerNorm weights/biases to F16
    ///   so the full forward runs in F16 GEMM/activations/norms.
    ///
    /// When `compute_dtype == DType::F32` (the default) the client is unused.
    pub fn from_weights_quant<G, C>(
        config: EncoderConfig,
        pooling: Pooling,
        client: &C,
        mut get: G,
    ) -> Result<Self>
    where
        G: FnMut(&str) -> Result<Weight<R>>,
        C: RuntimeClient<R> + TypeConversionOps<R> + DequantOps<R>,
    {
        let cdtype = config.compute_dtype;

        // Extract a standard (dense) tensor from a Weight, erroring on quantized.
        let extract_f32 = |w: Weight<R>, name: &str| -> Result<Tensor<R>> {
            w.as_tensor().cloned().map_err(|_| Error::ModelError {
                reason: format!("expected f32 tensor for '{name}', got quantized"),
            })
        };

        // Cast a tensor to compute_dtype when F16 is requested; otherwise return as-is.
        let maybe_cast = |t: Tensor<R>| -> Result<Tensor<R>> {
            if cdtype == DType::F16 && t.dtype() == DType::F32 {
                client.cast(&t, DType::F16).map_err(Error::Numr)
            } else {
                Ok(t)
            }
        };

        // Cast an optional bias tensor to compute_dtype.
        let maybe_cast_opt = |opt: Option<Tensor<R>>| -> Result<Option<Tensor<R>>> {
            opt.map(maybe_cast).transpose()
        };

        // Convert a projection weight to a Standard (dense) F16 MaybeQuantLinear
        // or leave it as-is (Standard F32 or Quantized) depending on compute_dtype.
        let proj_to_maybe_quant = |weight: Weight<R>,
                                   bias: Option<Tensor<R>>,
                                   client: &C|
         -> Result<MaybeQuantLinear<R>> {
            if cdtype == DType::F16 {
                // Dequantize to F16 and build a Standard linear — no quant_matmul on F16 path.
                let dense = match weight {
                    Weight::Quantized(qt) => {
                        client
                            .dequantize(&qt, DType::F16)
                            .map_err(|e| Error::QuantError {
                                reason: format!("dequantize to F16 failed: {e:#}"),
                            })?
                    }
                    Weight::Standard(t) => client.cast(&t, DType::F16).map_err(Error::Numr)?,
                    Weight::DecomposedQuant(_) => {
                        return Err(Error::ModelError {
                            reason:
                                "F16 compute_dtype does not support DecomposedQuant projections"
                                    .into(),
                        });
                    }
                };
                let bias_f16 = maybe_cast_opt(bias)?;
                Ok(MaybeQuantLinear::Standard(Linear::new(
                    dense, bias_f16, false,
                )))
            } else {
                Ok(MaybeQuantLinear::from_weight(weight, bias))
            }
        };

        let raw_token_embed = extract_f32(
            get("embeddings.word_embeddings.weight")?,
            "embeddings.word_embeddings.weight",
        )?;
        let token_embed = Embedding::new(maybe_cast(raw_token_embed)?, false);

        let raw_pos_embed = extract_f32(
            get("embeddings.position_embeddings.weight")?,
            "embeddings.position_embeddings.weight",
        )?;
        let position_embed = Embedding::new(maybe_cast(raw_pos_embed)?, false);

        let raw_en_w = extract_f32(
            get("embeddings.layer_norm.weight")?,
            "embeddings.layer_norm.weight",
        )?;
        let raw_en_b = extract_f32(
            get("embeddings.layer_norm.bias")?,
            "embeddings.layer_norm.bias",
        )?;
        let embed_norm = LayerNorm::new(
            maybe_cast(raw_en_w)?,
            maybe_cast(raw_en_b)?,
            config.layer_norm_eps as f32,
            false,
        );

        // Token-type ("segment") embedding, row 0 — see `from_weights` above.
        let raw_token_type = get("embeddings.token_type_embeddings.weight")
            .ok()
            .map(|w| extract_f32(w, "embeddings.token_type_embeddings.weight"))
            .transpose()?;
        let token_type_embed = token_type_row0(raw_token_type, &config)?
            .map(maybe_cast)
            .transpose()?;

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
            let q_proj = proj_to_maybe_quant(qw, qb, client)?;

            let (kw, kb) = load_proj(
                &mut get,
                &format!("{p}.attention.self.key.weight"),
                &format!("{p}.attention.self.key.bias"),
            )?;
            let k_proj = proj_to_maybe_quant(kw, kb, client)?;

            let (vw, vb) = load_proj(
                &mut get,
                &format!("{p}.attention.self.value.weight"),
                &format!("{p}.attention.self.value.bias"),
            )?;
            let v_proj = proj_to_maybe_quant(vw, vb, client)?;

            let (ow, ob) = load_proj(
                &mut get,
                &format!("{p}.attention.output.dense.weight"),
                &format!("{p}.attention.output.dense.bias"),
            )?;
            let o_proj = proj_to_maybe_quant(ow, ob, client)?;

            let attn_norm_w = extract_f32(
                get(&format!("{p}.attention.output.LayerNorm.weight"))?,
                &format!("{p}.attention.output.LayerNorm.weight"),
            )?;
            let attn_norm_b = extract_f32(
                get(&format!("{p}.attention.output.LayerNorm.bias"))?,
                &format!("{p}.attention.output.LayerNorm.bias"),
            )?;
            let attn_norm = LayerNorm::new(
                maybe_cast(attn_norm_w)?,
                maybe_cast(attn_norm_b)?,
                config.layer_norm_eps as f32,
                false,
            );

            let (uw, ub) = load_proj(
                &mut get,
                &format!("{p}.intermediate.dense.weight"),
                &format!("{p}.intermediate.dense.bias"),
            )?;
            let ffn_up = proj_to_maybe_quant(uw, ub, client)?;

            let (dw, db) = load_proj(
                &mut get,
                &format!("{p}.output.dense.weight"),
                &format!("{p}.output.dense.bias"),
            )?;
            let ffn_down = proj_to_maybe_quant(dw, db, client)?;

            let ffn_norm_w = extract_f32(
                get(&format!("{p}.output.LayerNorm.weight"))?,
                &format!("{p}.output.LayerNorm.weight"),
            )?;
            let ffn_norm_b = extract_f32(
                get(&format!("{p}.output.LayerNorm.bias"))?,
                &format!("{p}.output.LayerNorm.bias"),
            )?;
            let ffn_norm = LayerNorm::new(
                maybe_cast(ffn_norm_w)?,
                maybe_cast(ffn_norm_b)?,
                config.layer_norm_eps as f32,
                false,
            );

            layers.push(EncoderLayer {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                attn_norm: NormLayer::LayerNorm(attn_norm),
                ffn_up,
                ffn_gate: None,
                ffn_down,
                ffn_norm: NormLayer::LayerNorm(ffn_norm),
                num_heads: config.num_attention_heads,
                num_kv_heads: config.num_attention_heads,
                head_dim: config.head_dim(),
                hidden_act: config.hidden_act,
                ffn_variant: FfnVariant::Standard,
                norm_scheme: config.norm_scheme,
                attn: config.layer_attention(i),
                rope: None,
                q_norm: None,
                k_norm: None,
                qk_norm_scope: QkNormScope::PerHead,
                attn_norm_2: None,
                post_attn_norm: None,
                post_ffn_norm: None,
            });
        }

        Ok(Encoder {
            config,
            token_embed,
            position_embed,
            embed_norm: Some(embed_norm),
            layers,
            pooling,
            token_type_embed,
            output_norm: None,
            #[cfg(feature = "cuda")]
            forward_cache: std::sync::Arc::new(
                crate::model::encoder::model::graph_cache::EncoderForwardCache::new(),
            ),
        })
    }
}

/// Extract row 0 of a token-type embedding table as a `[1, hidden_size]` tensor.
///
/// `None` in, `None` out: the table is genuinely absent for some checkpoints,
/// and a zero stand-in would be indistinguishable from a real all-zero row
/// while hiding a failed load.
fn token_type_row0<R: Runtime<DType = DType>>(
    table: Option<Tensor<R>>,
    config: &EncoderConfig,
) -> Result<Option<Tensor<R>>> {
    let Some(table) = table else {
        return Ok(None);
    };
    let hidden_size = config.hidden_size;
    let data: Vec<f32> = table.to_vec();
    if data.len() < hidden_size {
        return Err(Error::ModelError {
            reason: format!(
                "token_type_embeddings has {} elements, need at least {hidden_size}",
                data.len()
            ),
        });
    }
    Ok(Some(Tensor::<R>::from_slice(
        &data[..hidden_size],
        &[1, hidden_size],
        table.device(),
    )))
}
