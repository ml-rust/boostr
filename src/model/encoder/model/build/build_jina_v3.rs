//! jina-bert-v3 encoder constructor (jina-embeddings-v3).
//!
//! Loads from the `jina-bert-v3` GGUF tensor namespace. Relative to BERT:
//!
//! - Fused, **biased** QKV: `blk.{i}.attn_qkv.{weight,bias}`, split at load
//!   time into three `[hidden, hidden]` projections and three `[hidden]`
//!   biases. The weight is dequantized to dense F32 before the split — quant
//!   block boundaries need not align with the H×H boundary, so slicing a
//!   compressed buffer could corrupt one or more projections.
//! - RoPE in place of a learned position table (base 20 000, not 10 000).
//! - Standard (non-gated) GELU FFN, with biases on both `ffn_up` and
//!   `ffn_down`; there is no `ffn_gate` tensor.
//! - Post-norm LayerNorm, an embedding LayerNorm, and token-type row 0 — all
//!   as in NomicBert.

use crate::error::{Error, Result};
use crate::model::encoder::config::{EncoderConfig, FfnVariant, QkNormScope};
use crate::model::encoder::model::layer::{EncoderLayer, NormLayer};
use crate::model::encoder::model::{Encoder, Pooling};
use crate::nn::{Embedding, LayerNorm, Linear, MaybeQuantLinear, RoPE, Weight};
use crate::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::sync::Arc;

impl<R: Runtime<DType = DType>> Encoder<R> {
    /// Create a jina-bert-v3 encoder from GGUF tensor names, with optional F16
    /// compute.
    ///
    /// `get` returns `Weight<R>` by GGUF name — the same contract as
    /// [`Encoder::from_weights_nomic`].
    pub fn from_weights_jina_v3<G, C>(
        config: EncoderConfig,
        pooling: Pooling,
        client: &C,
        mut get: G,
    ) -> Result<Self>
    where
        G: FnMut(&str) -> Result<Weight<R>>,
        C: RuntimeClient<R> + TypeConversionOps<R> + DequantOps<R>,
        R::Client: TypeConversionOps<R>,
    {
        let hidden_size = config.hidden_size;
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let eps = config.layer_norm_eps as f32;
        let cdtype = config.compute_dtype;

        let extract_f32 = |w: Weight<R>, name: &str| -> Result<Tensor<R>> {
            w.as_tensor().cloned().map_err(|_| Error::ModelError {
                reason: format!("expected f32 tensor for '{name}', got quantized"),
            })
        };

        let maybe_cast = |t: Tensor<R>| -> Result<Tensor<R>> {
            if cdtype == DType::F16 && t.dtype() == DType::F32 {
                client.cast(&t, DType::F16).map_err(Error::Numr)
            } else {
                Ok(t)
            }
        };
        let maybe_cast_opt = |opt: Option<Tensor<R>>| -> Result<Option<Tensor<R>>> {
            opt.map(maybe_cast).transpose()
        };

        let proj_to_maybe_quant = |weight: Weight<R>,
                                   bias: Option<Tensor<R>>,
                                   client: &C|
         -> Result<MaybeQuantLinear<R>> {
            if cdtype == DType::F16 {
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

        let raw_token_embd = extract_f32(get("token_embd.weight")?, "token_embd.weight")?;
        let device = raw_token_embd.device().clone();
        let token_embed = Embedding::new(maybe_cast(raw_token_embd)?, false);

        // RoPE supplies positions; `position_embed` holds a sentinel that the
        // forward pass never looks up (see `uses_learned_positions`).
        let sentinel_raw =
            Tensor::<R>::from_slice(&vec![0.0f32; hidden_size], &[1, hidden_size], &device);
        let position_embed = Embedding::new(maybe_cast(sentinel_raw)?, false);

        let raw_en_w = extract_f32(get("token_embd_norm.weight")?, "token_embd_norm.weight")?;
        let raw_en_b = extract_f32(get("token_embd_norm.bias")?, "token_embd_norm.bias")?;
        let embed_norm = LayerNorm::new(maybe_cast(raw_en_w)?, maybe_cast(raw_en_b)?, eps, false);

        // Token-type embedding, row 0 (single-segment inference). This file
        // declares a single segment type, so the tensor is one row wide and
        // row 0 is all of it; the slice below covers both layouts.
        let token_types_raw = extract_f32(get("token_types.weight")?, "token_types.weight")?;
        let token_types_data: Vec<f32> = token_types_raw.to_vec();
        if token_types_data.len() < hidden_size {
            return Err(Error::ModelError {
                reason: format!(
                    "token_types.weight has {} elements, need at least {hidden_size}",
                    token_types_data.len()
                ),
            });
        }
        let row0_tensor =
            Tensor::<R>::from_slice(&token_types_data[..hidden_size], &[1, hidden_size], &device);
        let token_type_embed = Some(maybe_cast(row0_tensor)?);

        let mut rope = RoPE::<R>::precompute_freqs(
            config.max_position_embeddings,
            head_dim,
            config.rope_freq_base,
            None,
            &device,
        );
        if cdtype == DType::F16 {
            rope.cast_caches(DType::F16);
        }
        let rope = Arc::new(rope);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let qkv_name = format!("blk.{i}.attn_qkv.weight");
            let qkv_data: Vec<f32> = match get(&qkv_name)? {
                Weight::Quantized(ref qt) => client
                    .dequantize(qt, DType::F32)
                    .map_err(|e| Error::QuantError {
                        reason: format!("{qkv_name} dequant to F32 failed: {e:#}"),
                    })?
                    .to_vec(),
                Weight::Standard(ref t) => t.to_vec(),
                Weight::DecomposedQuant(_) => {
                    return Err(Error::ModelError {
                        reason: format!("{qkv_name}: DecomposedQuant is not supported"),
                    });
                }
            };

            let proj_elems = hidden_size * hidden_size;
            if qkv_data.len() < 3 * proj_elems {
                return Err(Error::ModelError {
                    reason: format!(
                        "{qkv_name}: expected {} elements (3*{hidden_size}^2), got {}",
                        3 * proj_elems,
                        qkv_data.len()
                    ),
                });
            }

            // The fused bias splits on the same boundaries as the weight.
            // Dropping it would leave Q, K and V each off by a learned constant
            // — an error the shapes cannot show.
            let qkv_bias_name = format!("blk.{i}.attn_qkv.bias");
            let qkv_bias_data: Vec<f32> =
                extract_f32(get(&qkv_bias_name)?, &qkv_bias_name)?.to_vec();
            if qkv_bias_data.len() < 3 * hidden_size {
                return Err(Error::ModelError {
                    reason: format!(
                        "{qkv_bias_name}: expected {} elements (3*{hidden_size}), got {}",
                        3 * hidden_size,
                        qkv_bias_data.len()
                    ),
                });
            }

            let split_proj = |part: usize| -> Result<MaybeQuantLinear<R>> {
                let w = Tensor::<R>::from_slice(
                    &qkv_data[part * proj_elems..(part + 1) * proj_elems],
                    &[hidden_size, hidden_size],
                    &device,
                );
                let b = Tensor::<R>::from_slice(
                    &qkv_bias_data[part * hidden_size..(part + 1) * hidden_size],
                    &[hidden_size],
                    &device,
                );
                proj_to_maybe_quant(Weight::Standard(w), Some(b), client)
            };
            let q_proj = split_proj(0)?;
            let k_proj = split_proj(1)?;
            let v_proj = split_proj(2)?;

            let o_bias = extract_f32(
                get(&format!("blk.{i}.attn_output.bias"))?,
                &format!("blk.{i}.attn_output.bias"),
            )?;
            let o_proj = proj_to_maybe_quant(
                get(&format!("blk.{i}.attn_output.weight"))?,
                Some(o_bias),
                client,
            )?;

            let attn_norm = LayerNorm::new(
                maybe_cast(extract_f32(
                    get(&format!("blk.{i}.attn_output_norm.weight"))?,
                    &format!("blk.{i}.attn_output_norm.weight"),
                )?)?,
                maybe_cast(extract_f32(
                    get(&format!("blk.{i}.attn_output_norm.bias"))?,
                    &format!("blk.{i}.attn_output_norm.bias"),
                )?)?,
                eps,
                false,
            );

            // Standard GELU FFN: up → activation → down, both biased. No gate.
            let up_bias = extract_f32(
                get(&format!("blk.{i}.ffn_up.bias"))?,
                &format!("blk.{i}.ffn_up.bias"),
            )?;
            let ffn_up = proj_to_maybe_quant(
                get(&format!("blk.{i}.ffn_up.weight"))?,
                Some(up_bias),
                client,
            )?;

            let down_bias = extract_f32(
                get(&format!("blk.{i}.ffn_down.bias"))?,
                &format!("blk.{i}.ffn_down.bias"),
            )?;
            let ffn_down = proj_to_maybe_quant(
                get(&format!("blk.{i}.ffn_down.weight"))?,
                Some(down_bias),
                client,
            )?;

            let ffn_norm = LayerNorm::new(
                maybe_cast(extract_f32(
                    get(&format!("blk.{i}.layer_output_norm.weight"))?,
                    &format!("blk.{i}.layer_output_norm.weight"),
                )?)?,
                maybe_cast(extract_f32(
                    get(&format!("blk.{i}.layer_output_norm.bias"))?,
                    &format!("blk.{i}.layer_output_norm.bias"),
                )?)?,
                eps,
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
                num_heads,
                num_kv_heads: num_heads,
                head_dim,
                hidden_act: config.hidden_act,
                ffn_variant: FfnVariant::Standard,
                norm_scheme: config.norm_scheme,
                attn: config.layer_attention(i),
                rope: Some(Arc::clone(&rope)),
                q_norm: None,
                k_norm: None,
                qk_norm_scope: QkNormScope::default(),
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
