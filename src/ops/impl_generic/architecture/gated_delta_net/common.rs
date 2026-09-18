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
