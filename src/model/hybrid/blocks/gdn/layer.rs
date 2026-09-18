//! Gated DeltaNet block: weights, construction, and the head-order
//! permutation `group_heads`. The forward is in `forward`.
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

use crate::error::{Error, Result};
use crate::model::config::GdnConfig;
use crate::nn::{GateOrder, GatedRmsNorm, MaybeQuantLinear, MaybeRotatedLinear};
use numr::dtype::DType;
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
    pub(super) ssm_alpha: MaybeQuantLinear<R>,
    pub(super) ssm_beta: MaybeQuantLinear<R>,
    pub(super) ssm_out: MaybeRotatedLinear<R>,
    /// Depthwise conv kernel `[qkv_dim, 1, conv_kernel]`, the layout
    /// `nn::causal_conv1d` takes. No bias: the fork's `ggml_ssm_conv` has none.
    pub(super) conv_weight: Tensor<R>,
    /// `[value_heads]`. Holds `-exp(A_log)` as stored in the GGUF; the forward
    /// multiplies it in as is.
    pub(super) ssm_a: Tensor<R>,
    /// `[value_heads]`, added to `ssm_alpha(x)` before the softplus.
    pub(super) ssm_dt_bias: Tensor<R>,
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
    /// # Errors
    ///
    /// [`Error::ModelError`] when `cfg.validate()` fails or a tensor shape
    /// disagrees with `cfg`.
    pub fn new(cfg: GdnConfig, weights: GdnWeights<R>) -> Result<Self> {
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

        Ok(Self {
            cfg,
            attn_qkv: weights.attn_qkv,
            attn_gate: weights.attn_gate,
            ssm_alpha: weights.ssm_alpha,
            ssm_beta: weights.ssm_beta,
            ssm_out: weights.ssm_out,
            conv_weight,
            ssm_a: weights.ssm_a,
            ssm_dt_bias: weights.ssm_dt_bias,
            norm,
        })
    }

    /// Layer config.
    pub fn config(&self) -> &GdnConfig {
        &self.cfg
    }
}

/// Reorder `o` `[.., value_dim]` from tiled to grouped value-head order,
/// the input order a Hadamard-folded `ssm_out` expects.
///
/// Port of the `perm_rep` branch of `build_lora_mm` (`llama-graph.cpp`):
///
/// ```text
/// // tiled [hd, nk, rep] -> grouped [hd, rep, nk] feature order
/// x = ggml_reshape_4d(ctx0, x, t.perm_hd, t.perm_nk, t.perm_rep, ne1*ne2*ne3);
/// x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 1, 3));
/// cur_mm = ggml_reshape_4d(ctx0, x, t.perm_hd*t.perm_nk*t.perm_rep, ne1, ne2, ne3);
/// ```
///
/// ggml lists `ne[0]` first, so `[hd, nk, rep]` is row-major
/// `[rep, nk, hd]` and the permute swaps the two head axes. Tiled head
/// `h = i_rep * H_k + i_k` lands at grouped head `h' = i_k * rep + i_rep`.
pub(super) fn group_heads<R: Runtime>(
    o: &Tensor<R>,
    key_heads: usize,
    rep: usize,
    head_dim: usize,
) -> Result<Tensor<R>> {
    let shape = o.shape();
    let Some((&last, lead)) = shape.split_last() else {
        return Err(Error::InvalidArgument {
            arg: "o",
            reason: "expected at least one dim".into(),
        });
    };
    if last != key_heads * rep * head_dim {
        return Err(Error::InvalidArgument {
            arg: "o",
            reason: format!(
                "last dim {last} != key_heads * rep * head_dim ({key_heads} * {rep} * {head_dim})"
            ),
        });
    }
    if rep == 1 {
        return Ok(o.clone());
    }
    let mut tiled = lead.to_vec();
    tiled.extend([rep, key_heads, head_dim]);
    let n = lead.len();
    let mut perm: Vec<usize> = (0..n).collect();
    perm.extend([n + 1, n, n + 2]);

    o.contiguous()?
        .reshape(&tiled)
        .map_err(Error::Numr)?
        .permute(&perm)
        .map_err(Error::Numr)?
        .contiguous()?
        .reshape(shape)
        .map_err(Error::Numr)
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
                "gdn.{name}: expected weight [{out_features}, {in_features}], got {shape:?}"
            ),
        });
    }
    Ok(())
}

fn check_shape<R: Runtime>(t: &Tensor<R>, name: &str, want: &[usize]) -> Result<()> {
    if t.shape() != want {
        return Err(Error::ModelError {
            reason: format!("gdn.{name}: expected shape {want:?}, got {:?}", t.shape()),
        });
    }
    Ok(())
}

/// Bring the conv kernel to `[qkv_dim, 1, conv_kernel]`.
fn conv_weight_ncl<R: Runtime>(w: Tensor<R>, qkv_dim: usize, kernel: usize) -> Result<Tensor<R>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    /// S=2, H_k=2, rep=2. Tiled head `h = i_rep * 2 + i_k` carries values
    /// `[10h, 10h + 1]`. Grouped order lists `(k0, rep0), (k0, rep1),
    /// (k1, rep0), (k1, rep1)` = old heads `0, 2, 1, 3`.
    #[test]
    fn group_heads_ports_the_fork_permutation() {
        let device = CpuDevice::new();
        let o = Tensor::<CpuRuntime>::from_slice(
            &[0.0f32, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
            &[1, 1, 8],
            &device,
        )
        .unwrap();
        let grouped = group_heads(&o, 2, 2, 2).unwrap();
        assert_eq!(grouped.shape(), &[1, 1, 8]);
        assert_eq!(
            grouped.to_vec::<f32>(),
            vec![0.0, 1.0, 20.0, 21.0, 10.0, 11.0, 30.0, 31.0]
        );
    }

    #[test]
    fn group_heads_is_identity_for_rep_one() {
        let device = CpuDevice::new();
        let o =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4], &device).unwrap();
        let same = group_heads(&o, 2, 1, 2).unwrap();
        assert_eq!(same.to_vec::<f32>(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn group_heads_rejects_width_mismatch() {
        let device = CpuDevice::new();
        let o = Tensor::<CpuRuntime>::zeros(&[1, 6], DType::F32, &device).unwrap();
        assert!(group_heads(&o, 2, 2, 2).is_err());
    }
}
