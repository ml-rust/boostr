//! jina-bert-v2 encoder constructor (jina-embeddings-v2).
//!
//! Loads from the `jina-bert-v2` GGUF tensor namespace. Five things here exist
//! nowhere else in this encoder:
//!
//! 1. **ALiBi.** No `position_embd`, no `rope.freq_base`. Position reaches
//!    attention only as the per-head distance penalty built by `SpanMasks`
//!    from `config.alibi_max_bias`; this builder therefore attaches no RoPE
//!    cache at all.
//! 2. **QK-norm is LayerNorm with bias, over the whole hidden vector** — not
//!    the per-head RmsNorm Gemma and Qwen3 use. Both the norm type and the axis
//!    differ, so neither can stand in for the other.
//! 3. **`attn_norm_2`.** The layer input is re-added a second time after the
//!    attention norm, then normalised again.
//! 4. **GeGLU with asymmetric bias**: `ffn_gate` and `ffn_up` are weight-only,
//!    `ffn_down` is biased.
//! 5. Separate biased Q/K/V, plus the token-type row 0 treatment NomicBert has.

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
    /// Create a jina-bert-v2 encoder from GGUF tensor names, with optional F16
    /// compute.
    ///
    /// `get` returns `Weight<R>` by GGUF name — the same contract as
    /// [`Encoder::from_weights_nomic`].
    pub fn from_weights_jina_v2<G, C>(
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
        let num_kv_heads = config.resolved_num_kv_heads();
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

        // ALiBi supplies positions; `position_embed` holds a sentinel that the
        // forward pass never looks up (see `uses_learned_positions`).
        let sentinel_raw =
            Tensor::<R>::from_slice(&vec![0.0f32; hidden_size], &[1, hidden_size], &device)?;
        let position_embed = Embedding::new(maybe_cast(sentinel_raw)?, false);

        let raw_en_w = extract_f32(get("token_embd_norm.weight")?, "token_embd_norm.weight")?;
        let raw_en_b = extract_f32(get("token_embd_norm.bias")?, "token_embd_norm.bias")?;
        let embed_norm = LayerNorm::new(maybe_cast(raw_en_w)?, maybe_cast(raw_en_b)?, eps, false);

        // Token-type embedding: row 0 only (single-segment inference).
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
            Tensor::<R>::from_slice(&token_types_data[..hidden_size], &[1, hidden_size], &device)?;
        let token_type_embed = Some(maybe_cast(row0_tensor)?);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let q_bias = extract_f32(
                get(&format!("blk.{i}.attn_q.bias"))?,
                &format!("blk.{i}.attn_q.bias"),
            )?;
            let q_proj = proj_to_maybe_quant(
                get(&format!("blk.{i}.attn_q.weight"))?,
                Some(q_bias),
                client,
            )?;
            let k_bias = extract_f32(
                get(&format!("blk.{i}.attn_k.bias"))?,
                &format!("blk.{i}.attn_k.bias"),
            )?;
            let k_proj = proj_to_maybe_quant(
                get(&format!("blk.{i}.attn_k.weight"))?,
                Some(k_bias),
                client,
            )?;
            let v_bias = extract_f32(
                get(&format!("blk.{i}.attn_v.bias"))?,
                &format!("blk.{i}.attn_v.bias"),
            )?;
            let v_proj = proj_to_maybe_quant(
                get(&format!("blk.{i}.attn_v.weight"))?,
                Some(v_bias),
                client,
            )?;
            let o_bias = extract_f32(
                get(&format!("blk.{i}.attn_output.bias"))?,
                &format!("blk.{i}.attn_output.bias"),
            )?;
            let o_proj = proj_to_maybe_quant(
                get(&format!("blk.{i}.attn_output.weight"))?,
                Some(o_bias),
                client,
            )?;

            // QK-norm over the full hidden vector — weights are [hidden_size],
            // and `QkNormScope::Hidden` makes the attention path apply them
            // before the reshape into heads.
            let q_norm = Some(NormLayer::LayerNorm(load_layer_norm(
                &mut get,
                &format!("blk.{i}.attn_q_norm"),
                eps,
                maybe_cast,
            )?));
            let k_norm = Some(NormLayer::LayerNorm(load_layer_norm(
                &mut get,
                &format!("blk.{i}.attn_k_norm"),
                eps,
                maybe_cast,
            )?));

            let attn_norm = load_layer_norm(
                &mut get,
                &format!("blk.{i}.attn_output_norm"),
                eps,
                maybe_cast,
            )?;
            let attn_norm_2 = Some(NormLayer::LayerNorm(load_layer_norm(
                &mut get,
                &format!("blk.{i}.attn_norm_2"),
                eps,
                maybe_cast,
            )?));

            // GeGLU: gate and up carry no bias, down does.
            let ffn_gate = Some(proj_to_maybe_quant(
                get(&format!("blk.{i}.ffn_gate.weight"))?,
                None,
                client,
            )?);
            let ffn_up =
                proj_to_maybe_quant(get(&format!("blk.{i}.ffn_up.weight"))?, None, client)?;
            let down_bias = extract_f32(
                get(&format!("blk.{i}.ffn_down.bias"))?,
                &format!("blk.{i}.ffn_down.bias"),
            )?;
            let ffn_down = proj_to_maybe_quant(
                get(&format!("blk.{i}.ffn_down.weight"))?,
                Some(down_bias),
                client,
            )?;

            let ffn_norm = load_layer_norm(
                &mut get,
                &format!("blk.{i}.layer_output_norm"),
                eps,
                maybe_cast,
            )?;

            layers.push(EncoderLayer {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                attn_norm: NormLayer::LayerNorm(attn_norm),
                ffn_up,
                ffn_gate,
                ffn_down,
                ffn_norm: NormLayer::LayerNorm(ffn_norm),
                num_heads,
                num_kv_heads,
                head_dim,
                hidden_act: config.hidden_act,
                ffn_variant: FfnVariant::GatedGelu,
                norm_scheme: config.norm_scheme,
                attn: config.layer_attention(i),
                // No RoPE: positions arrive as the ALiBi score bias.
                rope: None,
                q_norm,
                k_norm,
                qk_norm_scope: QkNormScope::Hidden,
                attn_norm_2,
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

/// Load a biased LayerNorm from the `{name}.weight` / `{name}.bias` pair.
///
/// A free function rather than a closure so it borrows `get` only for the call:
/// a closure capturing `get` would hold the mutable borrow for the whole layer
/// body and lock out every other tensor fetch.
fn load_layer_norm<R, G, F>(get: &mut G, name: &str, eps: f32, cast: F) -> Result<LayerNorm<R>>
where
    R: Runtime<DType = DType>,
    G: FnMut(&str) -> Result<Weight<R>>,
    F: Fn(Tensor<R>) -> Result<Tensor<R>>,
{
    let dense = |w: Weight<R>, full: &str| -> Result<Tensor<R>> {
        w.as_tensor().cloned().map_err(|_| Error::ModelError {
            reason: format!("expected f32 tensor for '{full}', got quantized"),
        })
    };
    let w_name = format!("{name}.weight");
    let b_name = format!("{name}.bias");
    let weight = cast(dense(get(&w_name)?, &w_name)?)?;
    let bias = cast(dense(get(&b_name)?, &b_name)?)?;
    Ok(LayerNorm::new(weight, bias, eps, false))
}
