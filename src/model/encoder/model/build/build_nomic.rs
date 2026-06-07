//! NomicBert encoder constructor.
//!
//! Loads from the `nomic-bert` GGUF tensor namespace. Key differences from BERT:
//! - Fused QKV tensor `blk.{i}.attn_qkv.weight` ([3H, H]), split at load time.
//!   The raw Weight may be quantized; we dequantize to dense F32 first, then
//!   split the flat slice.  Splitting a quantized block-compressed tensor before
//!   dequantization is unsafe because blocks may span the split boundary.
//! - SwiGLU FFN: separate gate / up / down projections, no bias.
//! - RoPE: precomputed once and shared across layers via `Arc`.
//! - Token-type embedding: row 0 of `token_types.weight` added to token embeddings.
//! - No learned absolute position embedding (RoPE handles positions).

use crate::error::{Error, Result};
use crate::model::encoder::config::{EncoderConfig, FfnVariant};
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
    /// Create a NomicBert encoder from GGUF tensor names, with optional F16 compute.
    ///
    /// `get` returns `Weight<R>` by GGUF name — the same contract as
    /// `from_weights_quant`.  Quantized weights are passed through as
    /// `Weight::Quantized`; the builder dequantizes them before splitting
    /// the fused QKV (required — quant block alignment forbids splitting first).
    ///
    /// When `config.compute_dtype == DType::F16`, the `client` is used to:
    ///
    /// - Dequantize or cast all projection weights to F16 dense tensors so the
    ///   forward pass hits the F16 WMMA GEMM on CUDA, not the quant_matmul path.
    /// - Cast all F32 embedding tables, LayerNorm weights/biases, the sentinel
    ///   position tensor, and the RoPE cos/sin caches to F16 so every op in the
    ///   forward runs in a single dtype.
    ///
    /// When `compute_dtype == DType::F32` (the default) the client is not used
    /// for any cast; weights are stored exactly as loaded.
    pub fn from_weights_nomic<G, C>(
        config: EncoderConfig,
        pooling: Pooling,
        client: &C,
        mut get: G,
    ) -> Result<Self>
    where
        G: FnMut(&str) -> Result<Weight<R>>,
        C: RuntimeClient<R> + TypeConversionOps<R> + DequantOps<R>,
        // `RoPE::cast_caches` obtains the default client internally to cast the
        // cos/sin caches to F16.
        R::Client: TypeConversionOps<R>,
    {
        let hidden_size = config.hidden_size;
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let eps = config.layer_norm_eps as f32;
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

        // Load token embeddings; capture device for all subsequent tensor allocations.
        let raw_token_embd = extract_f32(get("token_embd.weight")?, "token_embd.weight")?;
        let device = raw_token_embd.device().clone();
        let token_embed = Embedding::new(maybe_cast(raw_token_embd)?, false);

        // NomicBert positions are handled by per-layer RoPE; the position_embed field
        // is populated with a sentinel zero tensor that is never called.
        let sentinel_raw =
            Tensor::<R>::from_slice(&vec![0.0f32; hidden_size], &[1, hidden_size], &device);
        let sentinel_pos = maybe_cast(sentinel_raw)?;
        let position_embed = Embedding::new(sentinel_pos, false);

        let raw_en_w = extract_f32(get("token_embd_norm.weight")?, "token_embd_norm.weight")?;
        let raw_en_b = extract_f32(get("token_embd_norm.bias")?, "token_embd_norm.bias")?;
        let embed_norm = LayerNorm::new(maybe_cast(raw_en_w)?, maybe_cast(raw_en_b)?, eps, false);

        // Token-type embedding: load [type_vocab_size, hidden_size], extract row 0.
        // Row 0 is the only row used (single-segment inference).
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
        let row0: Vec<f32> = token_types_data[..hidden_size].to_vec();
        let row0_tensor = Tensor::<R>::from_slice(&row0, &[1, hidden_size], &device);
        let token_type_embed = Some(maybe_cast(row0_tensor)?);

        // Precompute RoPE frequency cache once; share across all layers via Arc.
        // When cdtype == F16 the caches are cast to F16 before wrapping in Arc so
        // rope.forward applies to F16 Q/K without a per-token dtype cast.
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
            // Fused QKV weight: shape [3*hidden_size, hidden_size].
            // Dequantize to a dense F32 Vec<f32> first, then split into three
            // [hidden_size, hidden_size] row-slices Q / K / V.
            //
            // The split MUST happen after dequantization: quant block boundaries
            // may not align with the H×H boundary, so slicing a compressed buffer
            // would corrupt one or more of the three projections.
            let qkv_weight = get(&format!("blk.{i}.attn_qkv.weight"))?;
            let qkv_data: Vec<f32> = match qkv_weight {
                Weight::Quantized(ref qt) => {
                    let dense_f32 =
                        client
                            .dequantize(qt, DType::F32)
                            .map_err(|e| Error::QuantError {
                                reason: format!("blk.{i}.attn_qkv dequant to F32 failed: {e:#}"),
                            })?;
                    dense_f32.to_vec()
                }
                Weight::Standard(ref t) => t.to_vec(),
                Weight::DecomposedQuant(_) => {
                    return Err(Error::ModelError {
                        reason: format!(
                            "blk.{i}.attn_qkv.weight: DecomposedQuant is not supported"
                        ),
                    });
                }
            };

            let proj_elems = hidden_size * hidden_size;
            if qkv_data.len() < 3 * proj_elems {
                return Err(Error::ModelError {
                    reason: format!(
                        "blk.{i}.attn_qkv.weight: expected {} elements (3*{hidden_size}^2), got {}",
                        3 * proj_elems,
                        qkv_data.len()
                    ),
                });
            }

            let q_tensor = Tensor::<R>::from_slice(
                &qkv_data[0..proj_elems],
                &[hidden_size, hidden_size],
                &device,
            );
            let k_tensor = Tensor::<R>::from_slice(
                &qkv_data[proj_elems..2 * proj_elems],
                &[hidden_size, hidden_size],
                &device,
            );
            let v_tensor = Tensor::<R>::from_slice(
                &qkv_data[2 * proj_elems..3 * proj_elems],
                &[hidden_size, hidden_size],
                &device,
            );

            // Wrap split slices as Standard weights then apply proj_to_maybe_quant,
            // which casts to F16 (when cdtype F16) and builds MaybeQuantLinear::Standard.
            // No bias on q/k/v projections in NomicBert.
            let q_proj = proj_to_maybe_quant(Weight::Standard(q_tensor), None, client)?;
            let k_proj = proj_to_maybe_quant(Weight::Standard(k_tensor), None, client)?;
            let v_proj = proj_to_maybe_quant(Weight::Standard(v_tensor), None, client)?;

            // Attention output projection — no bias.
            let o_weight = get(&format!("blk.{i}.attn_output.weight"))?;
            let o_proj = proj_to_maybe_quant(o_weight, None, client)?;

            let attn_norm_w = extract_f32(
                get(&format!("blk.{i}.attn_output_norm.weight"))?,
                &format!("blk.{i}.attn_output_norm.weight"),
            )?;
            let attn_norm_b = extract_f32(
                get(&format!("blk.{i}.attn_output_norm.bias"))?,
                &format!("blk.{i}.attn_output_norm.bias"),
            )?;
            let attn_norm = LayerNorm::new(
                maybe_cast(attn_norm_w)?,
                maybe_cast(attn_norm_b)?,
                eps,
                false,
            );

            // SwiGLU: gate, up, and down projections — no bias on any of them.
            let gate_weight = get(&format!("blk.{i}.ffn_gate.weight"))?;
            let ffn_gate = Some(proj_to_maybe_quant(gate_weight, None, client)?);

            let up_weight = get(&format!("blk.{i}.ffn_up.weight"))?;
            let ffn_up = proj_to_maybe_quant(up_weight, None, client)?;

            let down_weight = get(&format!("blk.{i}.ffn_down.weight"))?;
            let ffn_down = proj_to_maybe_quant(down_weight, None, client)?;

            let ffn_norm_w = extract_f32(
                get(&format!("blk.{i}.layer_output_norm.weight"))?,
                &format!("blk.{i}.layer_output_norm.weight"),
            )?;
            let ffn_norm_b = extract_f32(
                get(&format!("blk.{i}.layer_output_norm.bias"))?,
                &format!("blk.{i}.layer_output_norm.bias"),
            )?;
            let ffn_norm =
                LayerNorm::new(maybe_cast(ffn_norm_w)?, maybe_cast(ffn_norm_b)?, eps, false);

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
                num_kv_heads: num_heads,
                head_dim,
                hidden_act: config.hidden_act,
                ffn_variant: FfnVariant::GatedSilu,
                rope: Some(Arc::clone(&rope)),
                q_norm: None,
                k_norm: None,
                post_attn_norm: None,
                post_ffn_norm: None,
            });
        }

        Ok(Encoder {
            config,
            token_embed,
            position_embed,
            embed_norm,
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
