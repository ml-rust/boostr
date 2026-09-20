//! `qwen35` gated full-attention block: weights and construction. The
//! forward is in `forward`.
//!
//! # Norm and residual ownership
//!
//! `Qwen35AttentionBlock` is the mixer, the analogue of
//! [`GdnBlock`](super::GdnBlock) for the full-attention layers: it takes the
//! `attn_norm`-ed hidden state and returns the `attn_output` projection. The
//! enclosing layer owns `attn_norm`, the residual add,
//! `post_attention_norm` and the FFN.
//!
//! # Weight layout
//!
//! Ports `build_layer_attn` (`src/models/qwen35.cpp`):
//!
//! ```text
//! // [ (n_embd_head * 2) * n_head, n_tokens ]
//! ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, ...);
//! ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
//!     ggml_element_size(Qcur_full) * n_embd_head * 2, ..., 0);
//! ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
//!     ggml_element_size(Qcur_full) * n_embd_head * 2, ...,
//!     ggml_element_size(Qcur_full) * n_embd_head);
//! ```
//!
//! Head `h` of the stored `attn_q` output owns columns `[2·h·hd, 2·h·hd + hd)`
//! as the query and `[2·h·hd + hd, 2·h·hd + 2·hd)` as the gate. [`Qwen35AttentionBlock::new`]
//! regroups the weight's rows once, at construction, to
//! `[query: num_heads·hd | gate: num_heads·hd]`, so the forward splits the
//! projection with a `narrow` at column `num_heads·hd` instead of two
//! strided copies. The block's own `attn_q` is in the regrouped order; the
//! [`Qwen35AttentionWeights`] contract stays the stored, interleaved one.

use crate::error::{Error, Result};
use crate::model::config::Qwen35AttentionConfig;
use crate::nn::{MaybeQuantLinear, MaybeRotatedLinear, QkNorm};
use crate::ops::impl_generic::position::mrope_stream_selector;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Gated GQA attention mixer with per-head q/k RMS norm and IMROPE.
///
/// Forward: `attn_q` → split query/gate per head → `attn_q_norm`; `attn_k` →
/// `attn_k_norm`; `attn_v`; IMROPE on q, k → KV cache append → causal GQA
/// attention → `attn ⊙ sigmoid(gate)` → `attn_output`.
pub struct Qwen35AttentionBlock<R: Runtime> {
    pub(super) cfg: Qwen35AttentionConfig,
    /// Rows regrouped to `[query rows | gate rows]`; see the module doc.
    pub(super) attn_q: MaybeRotatedLinear<R>,
    pub(super) attn_k: MaybeRotatedLinear<R>,
    pub(super) attn_v: MaybeRotatedLinear<R>,
    pub(super) attn_output: MaybeRotatedLinear<R>,
    /// `attn_q_norm` / `attn_k_norm`, `[head_dim]` each.
    pub(super) qk_norm: QkNorm<R>,
    /// One-hot IMROPE stream selector, `[4, 1, rope_dim / 2]`, built once
    /// here from `cfg.rope_sections` and reused by `forward` for both q
    /// and k instead of rebuilding it every call.
    pub(super) mrope_selector: Tensor<R>,
}

/// Weights for [`Qwen35AttentionBlock::new`]. The linear layers arrive
/// built so the Hadamard attach is a constructor-time choice of the loader.
pub struct Qwen35AttentionWeights<R: Runtime> {
    /// `[num_heads * 2 * head_dim, hidden_size]`, query and gate interleaved per head.
    pub attn_q: MaybeRotatedLinear<R>,
    /// `[num_kv_heads * head_dim, hidden_size]`.
    pub attn_k: MaybeRotatedLinear<R>,
    /// `[num_kv_heads * head_dim, hidden_size]`.
    pub attn_v: MaybeRotatedLinear<R>,
    /// `[hidden_size, num_heads * head_dim]`.
    pub attn_output: MaybeRotatedLinear<R>,
    /// `[head_dim]`.
    pub attn_q_norm: Tensor<R>,
    /// `[head_dim]`.
    pub attn_k_norm: Tensor<R>,
}

impl<R: Runtime<DType = DType>> Qwen35AttentionBlock<R> {
    /// Build from a validated config and built modules.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when `cfg.validate()` fails or a tensor shape
    /// disagrees with `cfg`.
    pub fn new(cfg: Qwen35AttentionConfig, weights: Qwen35AttentionWeights<R>) -> Result<Self> {
        cfg.validate()?;
        check_linear(
            weights.attn_q.base(),
            "attn_q",
            cfg.q_gate_dim(),
            cfg.hidden_size,
        )?;
        check_linear(
            weights.attn_k.base(),
            "attn_k",
            cfg.kv_dim(),
            cfg.hidden_size,
        )?;
        check_linear(
            weights.attn_v.base(),
            "attn_v",
            cfg.kv_dim(),
            cfg.hidden_size,
        )?;
        check_linear(
            weights.attn_output.base(),
            "attn_output",
            cfg.hidden_size,
            cfg.q_dim(),
        )?;
        let device = weights.attn_q_norm.device().clone();
        let qk_norm = QkNorm::new(
            weights.attn_q_norm,
            weights.attn_k_norm,
            cfg.head_dim,
            cfg.rms_eps,
        )?;
        let mrope_selector =
            mrope_stream_selector::<R>(cfg.rope_sections, cfg.rope_dim / 2, &device)?;
        // Stored order is `[head][query | gate]`; the forward wants
        // `[query][head] | [gate][head]`.
        let attn_q = weights.attn_q.regroup_rows(cfg.num_heads, 2)?;

        Ok(Self {
            cfg,
            attn_q,
            attn_k: weights.attn_k,
            attn_v: weights.attn_v,
            attn_output: weights.attn_output,
            qk_norm,
            mrope_selector,
        })
    }

    /// Layer config.
    pub fn config(&self) -> &Qwen35AttentionConfig {
        &self.cfg
    }

    /// The per-head q/k norm.
    pub fn qk_norm(&self) -> &QkNorm<R> {
        &self.qk_norm
    }
}

fn check_linear<R: Runtime<DType = DType>>(
    linear: &MaybeQuantLinear<R>,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> Result<()> {
    let shape = linear.shape();
    if shape != [out_features, in_features] {
        return Err(Error::ModelError {
            reason: format!(
                "qwen35_attention.{name}: expected weight [{out_features}, {in_features}], \
                 got {shape:?}"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn cfg() -> Qwen35AttentionConfig {
        Qwen35AttentionConfig {
            hidden_size: 8,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            rope_dim: 4,
            rope_sections: [1, 1, 0, 0],
            rope_theta: 10_000.0,
            rms_eps: 1e-6,
        }
    }

    fn linear(device: &CpuDevice, out: usize, inp: usize) -> MaybeRotatedLinear<CpuRuntime> {
        let w = Tensor::<CpuRuntime>::zeros(&[out, inp], DType::F32, device).unwrap();
        MaybeRotatedLinear::Plain(MaybeQuantLinear::Standard(Linear::new(w, None, false)))
    }

    fn weights(device: &CpuDevice, q_out: usize) -> Qwen35AttentionWeights<CpuRuntime> {
        let ones = |n: usize| Tensor::<CpuRuntime>::ones(&[n], DType::F32, device).unwrap();
        Qwen35AttentionWeights {
            attn_q: linear(device, q_out, 8),
            attn_k: linear(device, 8, 8),
            attn_v: linear(device, 8, 8),
            attn_output: linear(device, 8, 16),
            attn_q_norm: ones(8),
            attn_k_norm: ones(8),
        }
    }

    #[test]
    fn accepts_matching_shapes() {
        let device = CpuDevice::new();
        let block = Qwen35AttentionBlock::new(cfg(), weights(&device, 32)).unwrap();
        assert_eq!(block.config().q_gate_dim(), 32);
        assert_eq!(block.qk_norm().head_dim(), 8);
    }

    #[test]
    fn rejects_unsplit_q_projection() {
        // `attn_q` must carry query AND gate: [num_heads * 2 * head_dim, hidden].
        let device = CpuDevice::new();
        assert!(Qwen35AttentionBlock::new(cfg(), weights(&device, 16)).is_err());
    }

    #[test]
    fn rejects_norm_width_mismatch() {
        let device = CpuDevice::new();
        let mut w = weights(&device, 32);
        w.attn_k_norm = Tensor::<CpuRuntime>::ones(&[4], DType::F32, &device).unwrap();
        assert!(Qwen35AttentionBlock::new(cfg(), w).is_err());
    }
}
