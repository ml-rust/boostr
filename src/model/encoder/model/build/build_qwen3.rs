//! Qwen3-embedding encoder constructor.
//!
//! Loads from the `qwen3` GGUF tensor namespace. Key differences from Gemma:
//!
//! - Pre-norm RMSNorm with plain residuals (`x = x + sublayer(norm(x))`), not
//!   the sandwich scheme — there are no `post_attention_norm`/`post_ffw_norm`
//!   tensors in the file.
//! - SwiGLU FFN (`ffn_down(silu(gate(x)) * up(x))`), not GeGLU.
//! - Causal attention, and the sentence vector is the last real token's hidden
//!   state rather than a mean.
//! - No token-embedding scale, no sliding window, one RoPE base for all blocks.
//! - Head dim comes from `attention.key_length` and is *not*
//!   `hidden_size / head_count`: for the 0.6B model those are 128 and 64, so the
//!   Q/K/V projections are wider than the residual stream.
//!
//! Shared with Gemma: separate Q/K/V projections, GQA, QK-norm on Q and K after
//! reshape and before RoPE, a final `output_norm` before pooling, no biases.

use crate::error::{Error, Result};
use crate::model::encoder::config::{EncoderConfig, FfnVariant, QkNormScope};
use crate::model::encoder::model::layer::{EncoderLayer, NormLayer};
use crate::model::encoder::model::{Encoder, Pooling};
use crate::nn::{Embedding, Linear, MaybeQuantLinear, RmsNorm, RoPE, Weight};
use crate::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::sync::Arc;

impl<R: Runtime<DType = DType>> Encoder<R> {
    /// Create a Qwen3-embedding encoder from GGUF tensor names.
    ///
    /// `get` returns `Weight<R>` by GGUF tensor name. Quantized weights are
    /// passed through as `Weight::Quantized`; on the F16 compute path the
    /// builder dequantizes them to dense F16 so the forward hits numr's WMMA
    /// GEMM rather than the quant_matmul path.
    pub fn from_weights_qwen3<G, C>(
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
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.resolved_num_kv_heads();
        let head_dim = config.resolved_head_dim();
        let eps = config.rms_eps as f32;
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

        let proj_to_maybe_quant = |weight: Weight<R>, client: &C| -> Result<MaybeQuantLinear<R>> {
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
                Ok(MaybeQuantLinear::Standard(Linear::new(dense, None, false)))
            } else {
                Ok(MaybeQuantLinear::from_weight(weight, None))
            }
        };

        let raw_token_embd = extract_f32(get("token_embd.weight")?, "token_embd.weight")?;
        let device = raw_token_embd.device().clone();
        let token_embed = Embedding::new(maybe_cast(raw_token_embd)?, false);

        // Qwen3 uses RoPE; no learned position embedding. Sentinel zero table,
        // never called (the forward skips it for RoPE families).
        let sentinel_raw =
            Tensor::<R>::from_slice(&vec![0.0f32; hidden_size], &[1, hidden_size], &device);
        let position_embed = Embedding::new(maybe_cast(sentinel_raw)?, false);

        // No token_embd_norm tensor in this architecture, and no norm is applied
        // to the embeddings before the first block. A unit-weight LayerNorm is
        // NOT a stand-in: it would still mean-centre and rescale the residual
        // stream that every block downstream reads.

        // One RoPE base — Qwen3 does not interleave — shared across blocks.
        let mut rope = RoPE::<R>::precompute_freqs(
            config.max_position_embeddings,
            head_dim,
            config.rope_freq_base,
            None,
            &device,
        )?;
        if cdtype == DType::F16 {
            rope.cast_caches(DType::F16);
        }
        let rope = Arc::new(rope);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let attn_norm_w = extract_f32(
                get(&format!("blk.{i}.attn_norm.weight"))?,
                &format!("blk.{i}.attn_norm.weight"),
            )?;
            let attn_norm = RmsNorm::new(maybe_cast(attn_norm_w)?, eps, false);

            let q_proj = proj_to_maybe_quant(get(&format!("blk.{i}.attn_q.weight"))?, client)?;
            let k_proj = proj_to_maybe_quant(get(&format!("blk.{i}.attn_k.weight"))?, client)?;
            let v_proj = proj_to_maybe_quant(get(&format!("blk.{i}.attn_v.weight"))?, client)?;
            let o_proj = proj_to_maybe_quant(get(&format!("blk.{i}.attn_output.weight"))?, client)?;

            // QK-norm over head_dim, applied after reshape and before RoPE.
            let q_norm_w = extract_f32(
                get(&format!("blk.{i}.attn_q_norm.weight"))?,
                &format!("blk.{i}.attn_q_norm.weight"),
            )?;
            let q_norm = Some(NormLayer::RmsNorm(RmsNorm::new(
                maybe_cast(q_norm_w)?,
                eps,
                false,
            )));

            let k_norm_w = extract_f32(
                get(&format!("blk.{i}.attn_k_norm.weight"))?,
                &format!("blk.{i}.attn_k_norm.weight"),
            )?;
            let k_norm = Some(NormLayer::RmsNorm(RmsNorm::new(
                maybe_cast(k_norm_w)?,
                eps,
                false,
            )));

            let ffn_norm_w = extract_f32(
                get(&format!("blk.{i}.ffn_norm.weight"))?,
                &format!("blk.{i}.ffn_norm.weight"),
            )?;
            let ffn_norm = RmsNorm::new(maybe_cast(ffn_norm_w)?, eps, false);

            let ffn_gate = Some(proj_to_maybe_quant(
                get(&format!("blk.{i}.ffn_gate.weight"))?,
                client,
            )?);
            let ffn_up = proj_to_maybe_quant(get(&format!("blk.{i}.ffn_up.weight"))?, client)?;
            let ffn_down = proj_to_maybe_quant(get(&format!("blk.{i}.ffn_down.weight"))?, client)?;

            layers.push(EncoderLayer {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                attn_norm: NormLayer::RmsNorm(attn_norm),
                ffn_up,
                ffn_gate,
                ffn_down,
                ffn_norm: NormLayer::RmsNorm(ffn_norm),
                num_heads,
                num_kv_heads,
                head_dim,
                hidden_act: config.hidden_act,
                ffn_variant: FfnVariant::GatedSilu,
                norm_scheme: config.norm_scheme,
                attn: config.layer_attention(i),
                rope: Some(Arc::clone(&rope)),
                q_norm,
                k_norm,
                // Pre-norm: no sandwich post-norms exist in the file.
                qk_norm_scope: QkNormScope::PerHead,
                attn_norm_2: None,
                post_attn_norm: None,
                post_ffn_norm: None,
            });
        }

        // Final output_norm (RMSNorm) over all hidden states before pooling.
        let output_norm_w = extract_f32(get("output_norm.weight")?, "output_norm.weight")?;
        let output_norm = Some(RmsNorm::new(maybe_cast(output_norm_w)?, eps, false));

        Ok(Encoder {
            config,
            token_embed,
            position_embed,
            embed_norm: None,
            layers,
            pooling,
            token_type_embed: None,
            output_norm,
            #[cfg(feature = "cuda")]
            forward_cache: std::sync::Arc::new(
                crate::model::encoder::model::graph_cache::EncoderForwardCache::new(),
            ),
        })
    }
}
