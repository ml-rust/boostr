//! Gated DeltaNet block: weights and construction. The forward is in
//! `forward`, the tracked training forward in `train`, the head-order
//! permutation `group_heads` in `heads`, the trainable switch in
//! `trainable` and parameter access in `params`.
//!
//! # Norm and residual ownership
//!
//! `GdnBlock` is the mixer, the analogue of [`Mamba2`](crate::model::mamba::mamba2::Mamba2)
//! inside the hybrid `SsmBlock`: it takes the
//! `attn_norm`-ed hidden state and returns the `ssm_out` projection. The
//! enclosing layer owns `attn_norm`, the residual add, `post_attention_norm`
//! and the FFN, exactly as `SsmBlock` owns `norm` and the residual around
//! `Mamba2::forward_inference`.
//!
//! # Head order
//!
//! The layer runs in TILED value-head order: value head `h_v` reads key head
//! `h_v % H_k`. The fork's non-fused graph repeats q and k with
//! `ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs)`
//! (`qwen35.cpp`, `build_layer_attn_linear`), a tile along `ne[1]`; the GGUF
//! converter reorders every V-side tensor "from grouped (by K head) to tiled
//! order" (`conversion/qwen.py`, `_reorder_v_heads`) to match. When
//! `ssm_out` was Hadamard-folded the converter keeps its columns grouped
//! and sets `prism.hadamard.gdn_v_grouped`; `group_heads` is the runtime
//! side of that contract.

use super::gate_proj::GateProjections;
use crate::error::{Error, Result};
use crate::model::config::GdnConfig;
use crate::nn::{GateOrder, GatedRmsNorm, MaybeQuantLinear, MaybeRotatedLinear};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Gated DeltaNet mixer.
///
/// Forward: `attn_qkv`/`attn_gate`/`ssm_alpha`/`ssm_beta` → causal conv +
/// SiLU on qkv → split → L2-norm q,k → tiled repeat → delta-rule recurrence
/// → `silu(z) * rms_norm(o)` → optional `group_heads` → `ssm_out`.
pub struct GdnBlock<R: Runtime> {
    pub(super) cfg: GdnConfig,
    pub(super) attn_qkv: MaybeRotatedLinear<R>,
    pub(super) attn_gate: MaybeRotatedLinear<R>,
    /// `ssm_alpha` and `ssm_beta`, one dense linear when both arrive dense.
    pub(super) gates: GateProjections<R>,
    pub(super) ssm_out: MaybeRotatedLinear<R>,
    /// Depthwise conv kernel `[qkv_dim, 1, conv_kernel]`, the layout
    /// `nn::causal_conv1d` takes. No bias: the fork's `ggml_ssm_conv` has none.
    pub(super) conv_weight: Var<R>,
    /// `[value_heads]`. Holds `-exp(A_log)` as loaded, never trainable. A
    /// block without `a_log` multiplies it in as is.
    pub(super) ssm_a: Var<R>,
    /// `[value_heads]`, `ln(-ssm_a)` in the stored dtype. `None` until
    /// [`GdnBlock::with_trainable`] turns training on. When present it is
    /// the decay's source: every forward uses `-exp(a_log)`.
    pub(super) a_log: Option<Var<R>>,
    /// `[value_heads]`, added to `ssm_alpha(x)` before the softplus.
    pub(super) ssm_dt_bias: Var<R>,
    /// `ssm_norm` `[state_size]` per head, `silu(z) * norm(o)`.
    pub(super) norm: GatedRmsNorm<R>,
}

/// Weights for [`GdnBlock::new`]. The linear layers arrive built so the
/// Hadamard attach is a constructor-time choice of the loader.
pub struct GdnWeights<R: Runtime> {
    /// `[qkv_dim, hidden_size]`.
    pub attn_qkv: MaybeRotatedLinear<R>,
    /// `[value_dim, hidden_size]`.
    pub attn_gate: MaybeRotatedLinear<R>,
    /// `[value_heads, hidden_size]`.
    pub ssm_alpha: MaybeQuantLinear<R>,
    /// `[value_heads, hidden_size]`.
    pub ssm_beta: MaybeQuantLinear<R>,
    /// `[hidden_size, value_dim]`.
    pub ssm_out: MaybeRotatedLinear<R>,
    /// `[qkv_dim, conv_kernel]` (GGUF `ssm_conv1d`, ggml `ne = [conv_kernel, qkv_dim]`)
    /// or already `[qkv_dim, 1, conv_kernel]`.
    pub ssm_conv1d: Tensor<R>,
    /// `[value_heads]`, `-exp(A_log)`.
    pub ssm_a: Tensor<R>,
    /// `[value_heads]`.
    pub ssm_dt_bias: Tensor<R>,
    /// `[state_size]`.
    pub ssm_norm: Tensor<R>,
}

impl<R: Runtime<DType = DType>> GdnBlock<R> {
    /// Build from a validated config and built modules.
    ///
    /// Dense `ssm_alpha` and `ssm_beta` are concatenated into one
    /// `[2 * value_heads, hidden_size]` linear here; see `gate_proj`. The
    /// conv kernel, `ssm_a`, `ssm_dt_bias` and `ssm_norm` are frozen;
    /// [`Self::with_trainable`] turns them trainable.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when `cfg.validate()` fails or a tensor shape
    /// disagrees with `cfg`.
    pub fn new(cfg: GdnConfig, weights: GdnWeights<R>) -> Result<Self>
    where
        R::Client: ShapeOps<R>,
    {
        cfg.validate()?;
        let qkv_dim = cfg.qkv_dim();
        let value_dim = cfg.value_dim();
        let heads = cfg.value_heads;

        check_linear(
            weights.attn_qkv.base(),
            "attn_qkv",
            qkv_dim,
            cfg.hidden_size,
        )?;
        check_linear(
            weights.attn_gate.base(),
            "attn_gate",
            value_dim,
            cfg.hidden_size,
        )?;
        check_linear(&weights.ssm_alpha, "ssm_alpha", heads, cfg.hidden_size)?;
        check_linear(&weights.ssm_beta, "ssm_beta", heads, cfg.hidden_size)?;
        check_linear(
            weights.ssm_out.base(),
            "ssm_out",
            cfg.hidden_size,
            value_dim,
        )?;

        let conv_weight = conv_weight_ncl(weights.ssm_conv1d, qkv_dim, cfg.conv_kernel)?;
        check_shape(&weights.ssm_a, "ssm_a", &[heads])?;
        check_shape(&weights.ssm_dt_bias, "ssm_dt_bias", &[heads])?;
        check_shape(&weights.ssm_norm, "ssm_norm", &[cfg.state_size])?;
        let norm = GatedRmsNorm::new(weights.ssm_norm, cfg.rms_eps, GateOrder::NormThenMul, false);
        let gates = GateProjections::new(weights.ssm_alpha, weights.ssm_beta)?;

        Ok(Self {
            cfg,
            attn_qkv: weights.attn_qkv,
            attn_gate: weights.attn_gate,
            gates,
            ssm_out: weights.ssm_out,
            conv_weight: Var::new(conv_weight, false),
            ssm_a: Var::new(weights.ssm_a, false),
            a_log: None,
            ssm_dt_bias: Var::new(weights.ssm_dt_bias, false),
            norm,
        })
    }

    /// Layer config.
    pub fn config(&self) -> &GdnConfig {
        &self.cfg
    }

    /// A block over the same weight storage, with its own projection
    /// objects, so an adapter attaches to it and not to `self`. Every `Var`
    /// keeps its `TensorId`.
    pub fn alias(&self) -> Self {
        Self {
            cfg: self.cfg.clone(),
            attn_qkv: self.attn_qkv.alias(),
            attn_gate: self.attn_gate.alias(),
            gates: self.gates.alias(),
            ssm_out: self.ssm_out.alias(),
            conv_weight: self.conv_weight.alias(),
            ssm_a: self.ssm_a.alias(),
            a_log: self.a_log.as_ref().map(Var::alias),
            ssm_dt_bias: self.ssm_dt_bias.alias(),
            norm: self.norm.alias(),
        }
    }
}

/// `qkv[.., start .. start + len]` reshaped to `shape`.
pub(super) fn slice_heads<R: Runtime>(
    qkv: &Tensor<R>,
    start: usize,
    len: usize,
    shape: &[usize],
) -> Result<Tensor<R>> {
    qkv.narrow(2, start, len)
        .map_err(Error::Numr)?
        .contiguous()?
        .reshape(shape)
        .map_err(Error::Numr)
}

pub(super) fn check_linear<R: Runtime<DType = DType>>(
    linear: &MaybeQuantLinear<R>,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> Result<()> {
    let shape = linear.shape();
    if shape != [out_features, in_features] {
        return Err(Error::ModelError {
            reason: format!(
                "gdn.{name}: expected weight [{out_features}, {in_features}], got {shape:?}"
            ),
        });
    }
    Ok(())
}

pub(super) fn check_shape<R: Runtime>(t: &Tensor<R>, name: &str, want: &[usize]) -> Result<()> {
    if t.shape() != want {
        return Err(Error::ModelError {
            reason: format!("gdn.{name}: expected shape {want:?}, got {:?}", t.shape()),
        });
    }
    Ok(())
}

/// Bring the conv kernel to `[qkv_dim, 1, conv_kernel]`.
pub(super) fn conv_weight_ncl<R: Runtime>(
    w: Tensor<R>,
    qkv_dim: usize,
    kernel: usize,
) -> Result<Tensor<R>> {
    let shape = w.shape();
    let ok = shape == [qkv_dim, kernel] || shape == [qkv_dim, 1, kernel];
    if !ok {
        return Err(Error::ModelError {
            reason: format!(
                "gdn.ssm_conv1d: expected [{qkv_dim}, {kernel}] or [{qkv_dim}, 1, {kernel}], got {shape:?}"
            ),
        });
    }
    w.contiguous()?
        .reshape(&[qkv_dim, 1, kernel])
        .map_err(Error::Numr)
}
