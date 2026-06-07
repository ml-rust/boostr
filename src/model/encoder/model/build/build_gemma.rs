//! Gemma-embedding encoder constructor.
//!
//! Loads from the `gemma-embedding` GGUF tensor namespace. Key differences from BERT:
//!
//! - Separate Q, K, V projections (not fused). K and V use `num_kv_heads` heads (GQA).
//! - Sandwich RMSNorm: pre-norm input → sublayer → post-norm output → residual add.
//! - QK-norm: RmsNorm applied to Q and K after reshape to [B, H, S, D], before RoPE.
//! - RoPE applied after QK-norm; caches cast to F16 on the F16 compute path.
//! - GeGLU FFN: ffn_down(gelu(ffn_gate(x)) * ffn_up(x)).
//! - Token embedding scale: multiply by sqrt(hidden_size) after lookup.
//! - Final output_norm (RMSNorm) applied to all hidden states before mean pooling.
//! - No learned absolute position embedding; no token-type embedding; no biases anywhere.
//! - Bidirectional (full) attention. The sliding_window from GGUF is stored in config
//!   but not enforced — our embedding sessions are well within the window size.

use crate::error::{Error, Result};
use crate::model::encoder::config::{EncoderConfig, FfnVariant};
use crate::model::encoder::model::layer::{EncoderLayer, NormLayer};
use crate::model::encoder::model::{Encoder, Pooling};
use crate::nn::{Embedding, LayerNorm, Linear, MaybeQuantLinear, RmsNorm, RoPE, Weight};
use crate::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::sync::Arc;

impl<R: Runtime<DType = DType>> Encoder<R> {
    /// Create a Gemma-embedding encoder from GGUF tensor names, with optional F16 compute.
    ///
    /// `get` returns `Weight<R>` by GGUF tensor name. Quantized weights are passed through
    /// as `Weight::Quantized`; the builder dequantizes them to F16 on the F16 path.
    ///
    /// When `config.compute_dtype == DType::F16`, the `client` is used to:
    /// - Dequantize or cast all projection weights to F16 dense tensors so the forward
    ///   pass hits numr's F16 WMMA GEMM kernel, not the quant_matmul path.
    /// - Cast all F32 RMSNorm weights and token embedding to F16.
    /// - Cast RoPE cos/sin caches to F16 via `rope.cast_caches(DType::F16)`.
    ///
    /// When `compute_dtype == DType::F32` (the default) the client performs no casts.
    pub fn from_weights_gemma<G, C>(
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

        // Convert a projection weight to a Standard (dense) F16 MaybeQuantLinear
        // or leave it as-is (Standard F32 or Quantized) depending on compute_dtype.
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

        // Load token embeddings and capture device for subsequent allocations.
        let raw_token_embd = extract_f32(get("token_embd.weight")?, "token_embd.weight")?;
        let device = raw_token_embd.device().clone();
        let token_embed = Embedding::new(maybe_cast(raw_token_embd)?, false);

        // Gemma uses RoPE; no learned position embedding. Provide a sentinel zero
        // embedding (never called in the forward pass — see mod.rs GemmaEmbedding branch).
        let sentinel_raw =
            Tensor::<R>::from_slice(&vec![0.0f32; hidden_size], &[1, hidden_size], &device);
        let position_embed = Embedding::new(maybe_cast(sentinel_raw)?, false);

        // Gemma has no token_embd_norm. Use an identity LayerNorm (weight=1, bias=0).
        // This is the least-invasive approach: no Option wrapping, no change to the
        // forward path that always calls embed_norm.
        let identity_w =
            Tensor::<R>::from_slice(&vec![1.0f32; hidden_size], &[hidden_size], &device);
        let identity_b =
            Tensor::<R>::from_slice(&vec![0.0f32; hidden_size], &[hidden_size], &device);
        let embed_norm =
            LayerNorm::new(maybe_cast(identity_w)?, maybe_cast(identity_b)?, eps, false);

        // Precompute RoPE frequency cache; share across all layers via Arc.
        // On the F16 path, cast caches to F16 so rope.forward runs without per-token casts.
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
            // Pre-attention RMSNorm (sandwich: applied to input before attention sublayer).
            let attn_norm_w = extract_f32(
                get(&format!("blk.{i}.attn_norm.weight"))?,
                &format!("blk.{i}.attn_norm.weight"),
            )?;
            let attn_norm = RmsNorm::new(maybe_cast(attn_norm_w)?, eps, false);

            // Separate Q, K, V projections. No biases anywhere in Gemma.
            let q_weight = get(&format!("blk.{i}.attn_q.weight"))?;
            let q_proj = proj_to_maybe_quant(q_weight, client)?;

            let k_weight = get(&format!("blk.{i}.attn_k.weight"))?;
            let k_proj = proj_to_maybe_quant(k_weight, client)?;

            let v_weight = get(&format!("blk.{i}.attn_v.weight"))?;
            let v_proj = proj_to_maybe_quant(v_weight, client)?;

            let o_weight = get(&format!("blk.{i}.attn_output.weight"))?;
            let o_proj = proj_to_maybe_quant(o_weight, client)?;

            // QK-norm: RmsNorm over head_dim applied to Q and K after reshape,
            // before RoPE (Gemma-specific; weight shape [head_dim]).
            let q_norm_w = extract_f32(
                get(&format!("blk.{i}.attn_q_norm.weight"))?,
                &format!("blk.{i}.attn_q_norm.weight"),
            )?;
            let q_norm = Some(RmsNorm::new(maybe_cast(q_norm_w)?, eps, false));

            let k_norm_w = extract_f32(
                get(&format!("blk.{i}.attn_k_norm.weight"))?,
                &format!("blk.{i}.attn_k_norm.weight"),
            )?;
            let k_norm = Some(RmsNorm::new(maybe_cast(k_norm_w)?, eps, false));

            // Post-attention sandwich RMSNorm (applied to attn output before residual add).
            let post_attn_norm_w = extract_f32(
                get(&format!("blk.{i}.post_attention_norm.weight"))?,
                &format!("blk.{i}.post_attention_norm.weight"),
            )?;
            let post_attn_norm = Some(RmsNorm::new(maybe_cast(post_attn_norm_w)?, eps, false));

            // Pre-FFN RMSNorm (sandwich: applied to input before FFN sublayer).
            let ffn_norm_w = extract_f32(
                get(&format!("blk.{i}.ffn_norm.weight"))?,
                &format!("blk.{i}.ffn_norm.weight"),
            )?;
            let ffn_norm = RmsNorm::new(maybe_cast(ffn_norm_w)?, eps, false);

            // GeGLU: gate, up, and down projections — no biases.
            let gate_weight = get(&format!("blk.{i}.ffn_gate.weight"))?;
            let ffn_gate = Some(proj_to_maybe_quant(gate_weight, client)?);

            let up_weight = get(&format!("blk.{i}.ffn_up.weight"))?;
            let ffn_up = proj_to_maybe_quant(up_weight, client)?;

            let down_weight = get(&format!("blk.{i}.ffn_down.weight"))?;
            let ffn_down = proj_to_maybe_quant(down_weight, client)?;

            // Post-FFN sandwich RMSNorm (applied to FFN output before residual add).
            let post_ffn_norm_w = extract_f32(
                get(&format!("blk.{i}.post_ffw_norm.weight"))?,
                &format!("blk.{i}.post_ffw_norm.weight"),
            )?;
            let post_ffn_norm = Some(RmsNorm::new(maybe_cast(post_ffn_norm_w)?, eps, false));

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
                ffn_variant: FfnVariant::GatedGelu,
                rope: Some(Arc::clone(&rope)),
                q_norm,
                k_norm,
                post_attn_norm,
                post_ffn_norm,
            });
        }

        // Final output_norm (RMSNorm) applied to all hidden states before mean pooling.
        let output_norm_w = extract_f32(get("output_norm.weight")?, "output_norm.weight")?;
        let output_norm = Some(RmsNorm::new(maybe_cast(output_norm_w)?, eps, false));

        Ok(Encoder {
            config,
            token_embed,
            position_embed,
            embed_norm,
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
