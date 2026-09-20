//! Shape checks shared by the GDN step and chunk paths.

use crate::error::{Error, Result};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Dimensions of one GDN call.
#[derive(Debug, Clone, Copy)]
pub struct GdnDims {
    /// Batch size.
    pub batch: usize,
    /// Tokens per sequence.
    pub seq: usize,
    /// Value head count.
    pub heads: usize,
    /// Key head dimension.
    pub s_k: usize,
    /// Value head dimension.
    pub s_v: usize,
}

fn expect_rank<R: Runtime>(t: &Tensor<R>, arg: &'static str, rank: usize) -> Result<()> {
    if t.ndim() != rank {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!("expected {rank}D, got {}D {:?}", t.ndim(), t.shape()),
        });
    }
    Ok(())
}

fn expect_shape<R: Runtime>(t: &Tensor<R>, arg: &'static str, want: &[usize]) -> Result<()> {
    if t.shape() != want {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!("expected shape {want:?}, got {:?}", t.shape()),
        });
    }
    Ok(())
}

/// Check every operand against the trait layout and return the dims.
///
/// - `q`, `k`: `[batch, seq, H, S_k]`
/// - `v`: `[batch, seq, H, S_v]`
/// - `g`, `beta`: `[batch, seq, H]`
/// - `state`: `[batch, H, S_k, S_v]`
pub fn check_gdn_shapes<R: Runtime>(
    q: &Tensor<R>,
    k: &Tensor<R>,
    v: &Tensor<R>,
    g: &Tensor<R>,
    beta: &Tensor<R>,
    state: &Tensor<R>,
) -> Result<GdnDims> {
    expect_rank(q, "q", 4)?;
    expect_rank(v, "v", 4)?;
    let qs = q.shape();
    let dims = GdnDims {
        batch: qs[0],
        seq: qs[1],
        heads: qs[2],
        s_k: qs[3],
        s_v: v.shape()[3],
    };
    let GdnDims {
        batch,
        seq,
        heads,
        s_k,
        s_v,
    } = dims;
    expect_shape(k, "k", &[batch, seq, heads, s_k])?;
    expect_shape(v, "v", &[batch, seq, heads, s_v])?;
    expect_shape(g, "g", &[batch, seq, heads])?;
    expect_shape(beta, "beta", &[batch, seq, heads])?;
    expect_shape(state, "state", &[batch, heads, s_k, s_v])?;
    if seq == 0 {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: "seq must be >= 1".into(),
        });
    }
    Ok(dims)
}

/// Dimensions of one `gdn_step_from_conv` call.
#[derive(Debug, Clone, Copy)]
pub struct GdnConvDims {
    /// Batch size.
    pub batch: usize,
    /// Tokens per sequence.
    pub seq: usize,
    /// Key head count.
    pub h_k: usize,
    /// Value head count.
    pub h_v: usize,
    /// Key head dimension.
    pub s_k: usize,
    /// Value head dimension.
    pub s_v: usize,
    /// `h_k * s_k`: width of the q block and of the k block.
    pub key_dim: usize,
    /// `h_v * s_v`: width of the v block.
    pub value_dim: usize,
}

/// Check every operand of `gdn_step_from_conv` and return the dims.
///
/// - `qkv`: `[batch, seq, 2 * key_dim + value_dim]`
/// - `alpha_raw`, `beta_raw`: `[batch, seq, H_v]`
/// - `dt_bias`, `ssm_a`: `[H_v]`
/// - `state`: `[batch, H_v, S_k, S_v]`
///
/// `H_v` must be a multiple of `h_k`, `key_dim` a multiple of `h_k` and
/// `value_dim` a multiple of `H_v`.
#[allow(clippy::too_many_arguments)]
pub fn check_gdn_conv_shapes<R: Runtime>(
    qkv: &Tensor<R>,
    alpha_raw: &Tensor<R>,
    beta_raw: &Tensor<R>,
    dt_bias: &Tensor<R>,
    ssm_a: &Tensor<R>,
    state: &Tensor<R>,
    h_k: usize,
    key_dim: usize,
    value_dim: usize,
) -> Result<GdnConvDims> {
    expect_rank(qkv, "qkv", 3)?;
    expect_rank(alpha_raw, "alpha_raw", 3)?;
    expect_rank(state, "state", 4)?;
    let (batch, seq) = (qkv.shape()[0], qkv.shape()[1]);
    let h_v = alpha_raw.shape()[2];
    if h_k == 0 || h_v == 0 || !h_v.is_multiple_of(h_k) {
        return Err(Error::InvalidArgument {
            arg: "alpha_raw",
            reason: format!("value heads {h_v} must be a non-zero multiple of key heads {h_k}"),
        });
    }
    if !key_dim.is_multiple_of(h_k) || !value_dim.is_multiple_of(h_v) {
        return Err(Error::InvalidArgument {
            arg: "qkv",
            reason: format!(
                "key_dim {key_dim} must divide by key heads {h_k} and value_dim {value_dim} by value heads {h_v}"
            ),
        });
    }
    let dims = GdnConvDims {
        batch,
        seq,
        h_k,
        h_v,
        s_k: key_dim / h_k,
        s_v: value_dim / h_v,
        key_dim,
        value_dim,
    };
    expect_shape(qkv, "qkv", &[batch, seq, 2 * key_dim + value_dim])?;
    expect_shape(alpha_raw, "alpha_raw", &[batch, seq, h_v])?;
    expect_shape(beta_raw, "beta_raw", &[batch, seq, h_v])?;
    expect_shape(dt_bias, "dt_bias", &[h_v])?;
    expect_shape(ssm_a, "ssm_a", &[h_v])?;
    expect_shape(state, "state", &[batch, h_v, dims.s_k, dims.s_v])?;
    if seq == 0 {
        return Err(Error::InvalidArgument {
            arg: "qkv",
            reason: "seq must be >= 1".into(),
        });
    }
    Ok(dims)
}
