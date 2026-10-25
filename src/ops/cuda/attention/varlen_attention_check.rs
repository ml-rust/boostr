//! Host checks of the varlen attention launch arguments.
//!
//! The kernels index every buffer from the shape arguments and read no
//! shape of their own. These checks hold each buffer to the shapes and
//! dtypes of the `VarLenAttentionOps` layout contract before a launch, so a
//! mismatched buffer is an error, never an out-of-bounds access. The
//! launchers pass contiguous copies of strided inputs. The device offsets
//! stay the caller's contract: ascending from 0 to the token count of their
//! tensor.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

/// The shape arguments of one varlen attention call.
#[derive(Debug, Clone, Copy)]
pub(super) struct VarlenDims {
    pub batch_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub head_dim: usize,
}

/// The token counts of Q and of K/V.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Totals {
    pub q: usize,
    pub k: usize,
}

fn expect(arg: &'static str, t: &Tensor<CudaRuntime>, shape: &[usize], dtype: DType) -> Result<()> {
    if t.shape() != shape || t.dtype() != dtype {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!(
                "varlen attention needs a {dtype:?} tensor of shape {shape:?}, got {:?} of \
                 shape {:?}",
                t.dtype(),
                t.shape()
            ),
        });
    }
    Ok(())
}

fn fits_i32(arg: &'static str, value: usize) -> Result<()> {
    if i32::try_from(value).is_err() {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!(
                "varlen attention takes {arg} up to {}, got {value}",
                i32::MAX
            ),
        });
    }
    Ok(())
}

/// Checks the forward arguments and returns the token counts.
///
/// # Errors
///
/// A head count of 0, `num_heads` not a multiple of `num_kv_heads`, a size
/// above `i32::MAX`, or a tensor whose shape or dtype breaks the contract:
/// Q `[Tq, num_heads, head_dim]`, K and V `[Tk, num_kv_heads, head_dim]` in
/// Q's dtype, both offset tensors I32 `[batch_size + 1]`.
pub(super) fn check_fwd_args(
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    cu_seqlens_q: &Tensor<CudaRuntime>,
    cu_seqlens_k: &Tensor<CudaRuntime>,
    d: VarlenDims,
) -> Result<Totals> {
    if d.num_heads == 0 || d.num_kv_heads == 0 || !d.num_heads.is_multiple_of(d.num_kv_heads) {
        return Err(Error::InvalidArgument {
            arg: "num_kv_heads",
            reason: format!(
                "varlen attention needs num_heads a positive multiple of num_kv_heads, got {} \
                 and {}",
                d.num_heads, d.num_kv_heads
            ),
        });
    }
    fits_i32("batch_size", d.batch_size)?;
    fits_i32("num_heads", d.num_heads)?;
    fits_i32("max_seqlen_q", d.max_seqlen_q)?;
    fits_i32("max_seqlen_k", d.max_seqlen_k)?;
    let first = |t: &Tensor<CudaRuntime>| t.shape().first().copied().unwrap_or(0);
    let totals = Totals {
        q: first(q),
        k: first(k),
    };
    let dtype = q.dtype();
    expect("q", q, &[totals.q, d.num_heads, d.head_dim], dtype)?;
    let kv = [totals.k, d.num_kv_heads, d.head_dim];
    expect("k", k, &kv, dtype)?;
    expect("v", v, &kv, dtype)?;
    let offsets = [d.batch_size + 1];
    expect("cu_seqlens_q", cu_seqlens_q, &offsets, DType::I32)?;
    expect("cu_seqlens_k", cu_seqlens_k, &offsets, DType::I32)?;
    Ok(totals)
}

/// Checks the backward arguments and returns the token counts.
///
/// # Errors
///
/// An error of [`check_fwd_args`], a `dout` or `output` of another shape or
/// dtype than Q, or an `lse` that is not an F32 `[Tq, num_heads]` tensor.
#[allow(
    clippy::too_many_arguments,
    reason = "the six forward and gradient tensors, both offset tensors and the dims"
)]
pub(super) fn check_bwd_args(
    dout: &Tensor<CudaRuntime>,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    output: &Tensor<CudaRuntime>,
    lse: &Tensor<CudaRuntime>,
    cu_seqlens_q: &Tensor<CudaRuntime>,
    cu_seqlens_k: &Tensor<CudaRuntime>,
    d: VarlenDims,
) -> Result<Totals> {
    let totals = check_fwd_args(q, k, v, cu_seqlens_q, cu_seqlens_k, d)?;
    let q_shape = [totals.q, d.num_heads, d.head_dim];
    expect("dout", dout, &q_shape, q.dtype())?;
    expect("output", output, &q_shape, q.dtype())?;
    expect("lse", lse, &[totals.q, d.num_heads], DType::F32)?;
    Ok(totals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cuda_setup;

    #[test]
    fn mismatched_buffers_are_refused_before_a_launch() {
        let Some(cuda) = cuda_setup() else {
            return;
        };
        let dev = &cuda.device;
        let t = |shape: &[usize], dtype| Tensor::<CudaRuntime>::zeros(shape, dtype, dev).unwrap();
        let d = VarlenDims {
            batch_size: 2,
            num_heads: 4,
            num_kv_heads: 2,
            max_seqlen_q: 3,
            max_seqlen_k: 3,
            head_dim: 64,
        };
        let (q, kv) = (t(&[5, 4, 64], DType::F32), t(&[5, 2, 64], DType::F32));
        let cu = t(&[3], DType::I32);
        assert_eq!(
            check_fwd_args(&q, &kv, &kv, &cu, &cu, d).unwrap(),
            Totals { q: 5, k: 5 }
        );
        let lse = t(&[5, 4], DType::F32);
        check_bwd_args(&q, &q, &kv, &kv, &q, &lse, &cu, &cu, d).unwrap();

        let short_cu = t(&[2], DType::I32);
        assert!(check_fwd_args(&q, &kv, &kv, &short_cu, &cu, d).is_err());
        let f32_cu = t(&[3], DType::F32);
        assert!(check_fwd_args(&q, &kv, &kv, &cu, &f32_cu, d).is_err());
        let wide_kv = t(&[5, 4, 64], DType::F32);
        assert!(check_fwd_args(&q, &wide_kv, &kv, &cu, &cu, d).is_err());
        let half_v = t(&[5, 2, 64], DType::F16);
        assert!(check_fwd_args(&q, &kv, &half_v, &cu, &cu, d).is_err());
        let bad_heads = VarlenDims {
            num_kv_heads: 3,
            ..d
        };
        assert!(check_fwd_args(&q, &kv, &kv, &cu, &cu, bad_heads).is_err());
        let short_lse = t(&[4, 4], DType::F32);
        assert!(check_bwd_args(&q, &q, &kv, &kv, &q, &short_lse, &cu, &cu, d).is_err());
    }
}
