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
use crate::model::encoder::config::{EncoderConfig, LayerAttention};
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
) -> Option<Tensor<R>> {
    if !spec_binds(spec, seq_len) {
        return None;
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
    Some(Tensor::<R>::from_slice(
        &data,
        &[1, 1, seq_len, seq_len],
        device,
    ))
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
}

impl<R: Runtime<DType = DType>> SpanMasks<R> {
    /// An empty set — no span constraints at all.
    pub fn none() -> Self {
        Self {
            local: None,
            global: None,
        }
    }

    /// Build the masks this config needs at `seq_len`.
    ///
    /// Call before the layer loop, and — on the CUDA graph path — before capture
    /// begins.
    pub fn build(config: &EncoderConfig, seq_len: usize, device: &R::Device) -> Self {
        let dtype = config.compute_dtype;

        // Layer 0 is local whenever the architecture interleaves at all, and
        // the first non-interleaved index is global; asking the config keeps
        // this in lockstep with the RoPE base assignment.
        let local = config
            .interleaves_attention()
            .then(|| additive_span_mask::<R>(config.layer_attention(0), seq_len, dtype, device))
            .flatten();

        let global_index = if config.interleaves_attention() {
            config.sliding_window_pattern - 1
        } else {
            0
        };
        let global =
            additive_span_mask::<R>(config.layer_attention(global_index), seq_len, dtype, device);

        Self { local, global }
    }

    /// The mask for a block with the given spec.
    pub(in crate::model::encoder) fn for_spec(&self, spec: LayerAttention) -> Option<&Tensor<R>> {
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
        [self.local.as_ref(), self.global.as_ref()]
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
