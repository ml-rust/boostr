//! Additive attention-span masks (sliding window and/or causality).
//!
//! These masks encode *which positions a query may see*, as opposed to the
//! padding mask, which encodes *which positions hold real tokens*. Both are
//! additive biases folded into the same `softmax_with_bias` call.
//!
//! Masks are built on the host and uploaded once per forward pass, before the
//! layer loop. They must never be constructed inside a CUDA graph capture
//! region: an inline host-to-device copy there bakes a host stack pointer into
//! the graph and faults with `CUDA_ERROR_ILLEGAL_ADDRESS` on replay.

use crate::error::{Error, Result};
use crate::model::encoder::config::{EncoderConfig, LayerAttention, alibi_slopes};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Additive bias applied to masked-out score positions.
///
/// F16 saturates at ±65504, so `-1e9` would overflow to `-inf` and poison the
/// softmax. `-30000.0` is representable and still drives the weight to zero.
fn masked_bias(dtype: DType) -> f32 {
    if dtype == DType::F16 { -30000.0 } else { -1e9 }
}

/// Build the additive `[1, 1, seq_len, seq_len]` span mask for `spec`.
///
/// Returns `None` when `spec` permits every pair at this sequence length — the
/// common case for full-attention blocks and for windowed blocks on inputs
/// shorter than the window. Callers skip the mask entirely in that case, so
/// short sequences keep the exact fast path they had before windowing existed.
pub(in crate::model::encoder) fn additive_span_mask<R: Runtime<DType = DType>>(
    spec: LayerAttention,
    seq_len: usize,
    dtype: DType,
    device: &R::Device,
) -> Result<Option<Tensor<R>>> {
    if !spec_binds(spec, seq_len) {
        return Ok(None);
    }

    let bias = masked_bias(dtype);
    let mut data = vec![0.0f32; seq_len * seq_len];
    for q in 0..seq_len {
        for k in 0..seq_len {
            if !spec.attends(q, k) {
                data[q * seq_len + k] = bias;
            }
        }
    }
    Ok(Some(Tensor::<R>::from_slice(
        &data,
        &[1, 1, seq_len, seq_len],
        device,
    )?))
}

/// Build the additive `[1, n_heads, seq_len, seq_len]` ALiBi bias.
///
/// Entry `(h, q, k)` is `slope[h] * -|q - k|` for a pair this block may attend,
/// and the masked sentinel for a pair it may not. That is exactly what ggml
/// computes: `ggml_soft_max_ext` receives a mask holding `-|p0 - p1|` (or
/// `-inf`) and multiplies it by the head's slope while scanning.
///
/// Unlike a span mask, this is per-head and is never `None` — an ALiBi model
/// has no other source of position information, so skipping the bias would
/// silently turn it into a bag-of-words encoder.
fn alibi_bias<R: Runtime<DType = DType>>(
    spec: LayerAttention,
    n_heads: usize,
    max_bias: f32,
    seq_len: usize,
    dtype: DType,
    device: &R::Device,
) -> Result<Tensor<R>> {
    let slopes = alibi_slopes(n_heads, max_bias);
    let masked = masked_bias(dtype);

    let mut data = vec![0.0f32; n_heads * seq_len * seq_len];
    for (h, slope) in slopes.iter().enumerate() {
        let head_base = h * seq_len * seq_len;
        for q in 0..seq_len {
            for k in 0..seq_len {
                data[head_base + q * seq_len + k] = if spec.attends(q, k) {
                    slope * -(q.abs_diff(k) as f32)
                } else {
                    masked
                };
            }
        }
    }

    Ok(Tensor::<R>::from_slice(
        &data,
        &[1, n_heads, seq_len, seq_len],
        device,
    )?)
}

/// Whether `spec` actually excludes any pair at this sequence length.
fn spec_binds(spec: LayerAttention, seq_len: usize) -> bool {
    if seq_len <= 1 {
        return false;
    }
    if spec.causal {
        return true;
    }
    // A symmetric window only bites once some pair is further apart than the
    // half-width. The furthest pair in the sequence is `seq_len - 1` apart.
    match spec.max_distance() {
        Some(half) => seq_len - 1 > half,
        None => false,
    }
}

/// The span masks a forward pass needs, built once and shared by every layer
/// with the same attention spec.
///
/// At most two distinct masks exist: one for local (windowed) blocks and one for
/// global blocks. Both are `None` for a plain bidirectional encoder, which is
/// why BERT and NomicBert pay nothing for this.
pub struct SpanMasks<R: Runtime> {
    local: Option<Tensor<R>>,
    global: Option<Tensor<R>>,
    /// Per-head ALiBi bias `[1, n_heads, S, S]`, for architectures that carry
    /// no RoPE and no learned position table.
    ///
    /// It already folds in whatever the block's span spec excludes, so it
    /// replaces the span mask rather than adding to it. ALiBi families here do
    /// not interleave local and global blocks, so one tensor serves every
    /// block — see [`SpanMasks::build`].
    alibi: Option<Tensor<R>>,
}

impl<R: Runtime<DType = DType>> SpanMasks<R> {
    /// An empty set — no span constraints at all.
    pub fn none() -> Self {
        Self {
            local: None,
            global: None,
            alibi: None,
        }
    }

    /// Build the masks this config needs at `seq_len`.
    ///
    /// Call before the layer loop, and — on the CUDA graph path — before capture
    /// begins.
    pub fn build(config: &EncoderConfig, seq_len: usize, device: &R::Device) -> Result<Self> {
        let dtype = config.compute_dtype;

        // ALiBi replaces the span mask outright: the per-head bias tensor
        // already carries the block's exclusions. Interleaving would need one
        // such tensor per attention spec; no ALiBi architecture here does that,
        // and `layer_attention(0)` is every block's spec when it does not.
        if let Some(max_bias) = config.alibi_max_bias {
            return Ok(Self {
                local: None,
                global: None,
                alibi: Some(alibi_bias::<R>(
                    config.layer_attention(0),
                    config.num_attention_heads,
                    max_bias,
                    seq_len,
                    dtype,
                    device,
                )?),
            });
        }

        // Layer 0 is local whenever the architecture interleaves at all, and
        // the first non-interleaved index is global; asking the config keeps
        // this in lockstep with the RoPE base assignment.
        let local = if config.interleaves_attention() {
            additive_span_mask::<R>(config.layer_attention(0), seq_len, dtype, device)?
        } else {
            None
        };

        let global_index = if config.interleaves_attention() {
            config.sliding_window_pattern - 1
        } else {
            0
        };
        let global =
            additive_span_mask::<R>(config.layer_attention(global_index), seq_len, dtype, device)?;

        Ok(Self {
            local,
            global,
            alibi: None,
        })
    }

    /// The mask for a block with the given spec.
    pub(in crate::model::encoder) fn for_spec(&self, spec: LayerAttention) -> Option<&Tensor<R>> {
        if self.alibi.is_some() {
            return self.alibi.as_ref();
        }
        if spec.window.is_some() {
            self.local.as_ref()
        } else {
            self.global.as_ref()
        }
    }

    /// Every mask tensor that exists, for callers that must keep the device
    /// allocations alive — a CUDA graph records their addresses, so they have to
    /// outlive the capture rather than being dropped with the calling frame.
    #[cfg(feature = "cuda")]
    pub(in crate::model::encoder) fn tensors(&self) -> Vec<&Tensor<R>> {
        [
            self.local.as_ref(),
            self.global.as_ref(),
            self.alibi.as_ref(),
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

/// Reject a packed (varlen) forward whose sliding window would actually bind.
///
/// The varlen attention kernel supports full and causal attention but not a
/// bounded span, so a windowed block cannot be evaluated correctly there. The
/// window is provably inert while every sequence is shorter than the
/// half-width, which is the regime the packed path is used in; beyond that the
/// caller must use the padded path, which does apply the mask.
///
/// This is a hard error rather than a silent fallback: returning unwindowed
/// results for a windowed model is precisely the failure this module exists to
/// prevent.
pub(in crate::model::encoder) fn ensure_varlen_span_is_unconstrained(
    config: &EncoderConfig,
    max_seqlen: usize,
) -> Result<()> {
    if config.varlen_span_is_unconstrained(max_seqlen) {
        return Ok(());
    }
    if config.causal {
        return Err(Error::ModelError {
            reason: "causal attention is not supported on the packed (varlen) encoder \
                     path; use the padded path"
                .into(),
        });
    }
    if config.alibi_max_bias.is_some() {
        return Err(Error::ModelError {
            reason: "ALiBi position bias is not supported on the packed (varlen) \
                     encoder path; use the padded path"
                .into(),
        });
    }
    let spec = config.layer_attention(0);
    Err(Error::ModelError {
        reason: format!(
            "sequence of {max_seqlen} tokens exceeds the sliding-window half-width of \
             {} for this model's local attention blocks, which the packed (varlen) \
             path cannot mask. Use the padded path for inputs this long.",
            spec.max_distance().unwrap_or(0)
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::encoder::config::ArchFamily;
    use numr::runtime::cpu::CpuRuntime;

    fn jina_v2_config(seq_heads: usize) -> EncoderConfig {
        EncoderConfig {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: seq_heads,
            intermediate_size: 3072,
            max_position_embeddings: 8192,
            arch_family: ArchFamily::JinaBertV2,
            alibi_max_bias: Some(8.0),
            ..Default::default()
        }
    }

    /// The bias must be per-head and must carry the distance penalty ggml
    /// applies: `slope[h] * -|q - k|`, zero on the diagonal.
    ///
    /// A `[1, 1, S, S]` bias — the shape every other mask here uses — would
    /// broadcast without error and give every head head 0's slope.
    #[test]
    fn alibi_bias_is_per_head_and_linear_in_distance() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let (heads, seq) = (12usize, 5usize);
        let config = jina_v2_config(heads);

        let masks = SpanMasks::<CpuRuntime>::build(&config, seq, &device)
            .expect("mask build must succeed on CPU");
        let bias = masks
            .for_spec(config.layer_attention(0))
            .expect("an ALiBi model must always produce a bias");
        assert_eq!(bias.shape(), &[1, heads, seq, seq]);

        let data: Vec<f32> = bias.to_vec();
        let slopes = alibi_slopes(heads, 8.0);
        for (h, slope) in slopes.iter().enumerate() {
            for q in 0..seq {
                for k in 0..seq {
                    let got = data[h * seq * seq + q * seq + k];
                    let want = slope * -(q.abs_diff(k) as f32);
                    assert!(
                        (got - want).abs() < 1e-6,
                        "head {h} ({q},{k}): {got} vs {want}"
                    );
                }
            }
        }

        // Head 0 and head 11 must genuinely differ, which is the whole point of
        // a per-head bias.
        let head0 = data[4];
        let head11 = data[11 * seq * seq + 4];
        assert!((head0 - head11).abs() > 1e-3, "{head0} vs {head11}");
    }

    /// A non-ALiBi bidirectional encoder must still pay nothing: no mask at all.
    #[test]
    fn no_alibi_means_no_mask_for_a_plain_encoder() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let config = EncoderConfig {
            hidden_size: 384,
            num_attention_heads: 12,
            num_hidden_layers: 6,
            ..Default::default()
        };
        let masks = SpanMasks::<CpuRuntime>::build(&config, 8, &device)
            .expect("mask build must succeed on CPU");
        assert!(masks.for_spec(config.layer_attention(0)).is_none());
    }
}
