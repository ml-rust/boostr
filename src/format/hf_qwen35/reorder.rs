//! Value-head reorder of Gated DeltaNet tensors, grouped to tiled order.
//!
//! A Hugging Face Qwen3.5 checkpoint stores the value heads grouped by key
//! head: `[k0v0 .. k0v{rep-1}, k1v0 .. k1v{rep-1}, ..]`. The GDN layer runs in
//! tiled order, where value head `h_v` reads key head `h_v % key_heads`:
//! `[k0v0, k1v0, .., k0v1, k1v1, ..]`. The llama.cpp converter makes the same
//! move in `_LinearAttentionVReorderBase._reorder_v_heads`.
//!
//! [`reorder_qkv_v_rows`] applies it to the V rows of an `in_proj_qkv`-shaped
//! tensor only.

use crate::error::{Error, Result};
use crate::model::config::GdnConfig;
use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// `t` with the value heads along `dim` moved from grouped to tiled order.
///
/// `dim` holds `key_heads * rep * head_dim` entries: `rep` value heads per
/// key head, `head_dim` entries per value head. The axis is viewed as
/// `[key_heads, rep, head_dim]`, the first two axes swap, and the axis is
/// flattened back. Other dims keep their order. The result is contiguous.
///
/// The inverse (tiled to grouped) is the same call with `key_heads` and
/// `rep` swapped.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `dim` is out of range, or when
/// `t.shape()[dim]` is not `key_heads * rep * head_dim`.
pub fn reorder_v_heads<R: Runtime>(
    t: &Tensor<R>,
    dim: usize,
    key_heads: usize,
    rep: usize,
    head_dim: usize,
) -> Result<Tensor<R>> {
    let shape = t.shape().to_vec();
    let Some(&len) = shape.get(dim) else {
        return Err(Error::InvalidArgument {
            arg: "dim",
            reason: format!("dim {dim} is out of range for shape {shape:?}"),
        });
    };
    let want = key_heads * rep * head_dim;
    if len != want || want == 0 {
        return Err(Error::InvalidArgument {
            arg: "t",
            reason: format!(
                "dim {dim} of shape {shape:?} is {len}; key_heads * rep * head_dim \
                 ({key_heads} * {rep} * {head_dim}) is {want}"
            ),
        });
    }
    let mut split = shape[..dim].to_vec();
    split.extend_from_slice(&[key_heads, rep, head_dim]);
    split.extend_from_slice(&shape[dim + 1..]);
    let mut perm: Vec<usize> = (0..split.len()).collect();
    perm.swap(dim, dim + 1);
    Ok(t.contiguous()?
        .reshape(&split)?
        .permute(&perm)?
        .contiguous()?
        .reshape(&shape)?)
}

/// `t` with `[q | k | v]` rows (`gdn.qkv_dim()` of them, `2 * key_dim`
/// query and key rows first) and only the V rows moved from grouped to
/// tiled order, value heads of width `state_size`. Other dims keep their
/// order. Fits `in_proj_qkv`, `conv1d` and a LoRA `lora_B` of
/// `in_proj_qkv`.
///
/// # Errors
///
/// [`Error::ModelError`] when the row count is not `gdn.qkv_dim()`; numr
/// errors propagate.
pub fn reorder_qkv_v_rows<R: Runtime<DType = DType>>(
    t: &Tensor<R>,
    gdn: &GdnConfig,
) -> Result<Tensor<R>>
where
    R::Client: ShapeOps<R>,
{
    let qk = 2 * gdn.key_dim();
    let rows = t.shape().first().copied().unwrap_or(0);
    if rows != gdn.qkv_dim() {
        return Err(Error::ModelError {
            reason: format!(
                "reorder_qkv_v_rows: shape {:?} has {rows} rows; q, k and v need {}",
                t.shape(),
                gdn.qkv_dim()
            ),
        });
    }
    let head = t.narrow(0, 0, qk)?.contiguous()?;
    let tail = reorder_v_heads(
        &t.narrow(0, qk, rows - qk)?,
        0,
        gdn.key_heads,
        gdn.head_repeat(),
        gdn.state_size,
    )?;
    Ok(Tensor::cat(&[&head, &tail], 0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    /// Two key heads, two value heads each, head width 1: key dim 2,
    /// q/k rows `0..4`, v rows `4..8`.
    fn gdn() -> GdnConfig {
        GdnConfig {
            hidden_size: 4,
            conv_kernel: 2,
            state_size: 1,
            key_heads: 2,
            value_heads: 4,
            inner_size: 4,
            rms_eps: 1e-6,
            chunk_size: 64,
            v_grouped: false,
        }
    }

    #[test]
    fn only_the_v_rows_move() {
        let device = CpuDevice::new();
        let rows: Vec<f32> = (0..8).map(|r| r as f32).collect();
        let t = Tensor::<CpuRuntime>::from_slice(&rows, &[8, 1], &device).unwrap();
        let out = reorder_qkv_v_rows(&t, &gdn()).unwrap();
        assert_eq!(
            out.to_vec::<f32>(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 5.0, 7.0]
        );
        let short = Tensor::<CpuRuntime>::from_slice(&rows[..6], &[6, 1], &device).unwrap();
        assert!(reorder_qkv_v_rows(&short, &gdn()).is_err());
    }

    fn iota(shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|i| i as f32).collect();
        Tensor::from_slice(&v, shape, device).unwrap()
    }

    fn values(t: &Tensor<CpuRuntime>) -> Vec<f32> {
        t.contiguous().unwrap().to_vec()
    }

    /// Two key heads, two value heads each, one entry per head:
    /// grouped `[k0v0, k0v1, k1v0, k1v1]` becomes `[k0v0, k1v0, k0v1, k1v1]`.
    #[test]
    fn single_entry_heads_tile() {
        let device = CpuDevice::new();
        let out = reorder_v_heads(&iota(&[4], &device), 0, 2, 2, 1).unwrap();
        assert_eq!(values(&out), vec![0.0, 2.0, 1.0, 3.0]);
    }

    /// Head width 2 moves whole rows of a `[8, 1]` tensor.
    #[test]
    fn rows_move_by_whole_heads() {
        let device = CpuDevice::new();
        let out = reorder_v_heads(&iota(&[8, 1], &device), 0, 2, 2, 2).unwrap();
        assert_eq!(out.shape(), &[8, 1]);
        assert_eq!(values(&out), vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]);
    }

    /// Columns of a `[2, 4]` tensor, head width 1.
    #[test]
    fn columns_reorder_per_row() {
        let device = CpuDevice::new();
        let out = reorder_v_heads(&iota(&[2, 4], &device), 1, 2, 2, 1).unwrap();
        assert_eq!(values(&out), vec![0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]);
    }

    /// Three key heads, two value heads each, head width 2, over a
    /// `[12, 1, 3]` conv kernel.
    #[test]
    fn reorder_then_inverse_is_identity() {
        let device = CpuDevice::new();
        let t = iota(&[12, 1, 3], &device);
        let tiled = reorder_v_heads(&t, 0, 3, 2, 2).unwrap();
        assert_ne!(values(&tiled), values(&t));
        let back = reorder_v_heads(&tiled, 0, 2, 3, 2).unwrap();
        assert_eq!(back.shape(), t.shape());
        assert_eq!(values(&back), values(&t));
    }

    #[test]
    fn one_value_head_per_key_head_keeps_the_order() {
        let device = CpuDevice::new();
        let t = iota(&[6, 2], &device);
        let out = reorder_v_heads(&t, 0, 3, 1, 2).unwrap();
        assert_eq!(values(&out), values(&t));
    }

    #[test]
    fn wrong_length_is_an_error() {
        let device = CpuDevice::new();
        let err = reorder_v_heads(&iota(&[5], &device), 0, 2, 2, 1).unwrap_err();
        assert!(
            err.to_string().contains("key_heads * rep * head_dim"),
            "{err}"
        );
        assert!(reorder_v_heads(&iota(&[4], &device), 1, 2, 2, 1).is_err());
    }
}
