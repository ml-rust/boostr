//! Value-head order of a [`GdnBlock`](super::GdnBlock) output: the
//! tiled-to-grouped permutation `group_heads` and its tracked twin
//! `group_heads_var`. See `layer` for the head-order contract.

use crate::error::{Error, Result};
use crate::nn::var_contiguous;
use numr::autograd::{Var, var_permute, var_reshape};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

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
    let Some((tiled, perm)) = head_layout(shape, key_heads, rep, head_dim)? else {
        return Ok(o.clone());
    };
    o.contiguous()?
        .reshape(&tiled)
        .map_err(Error::Numr)?
        .permute(&perm)
        .map_err(Error::Numr)?
        .contiguous()?
        .reshape(shape)
        .map_err(Error::Numr)
}

/// [`group_heads`] on a tracked `Var`. The gradient flows back through
/// the inverse permutation. `rep == 1` returns an alias of `o`.
pub(super) fn group_heads_var<R: Runtime>(
    o: &Var<R>,
    key_heads: usize,
    rep: usize,
    head_dim: usize,
) -> Result<Var<R>> {
    let shape = o.shape().to_vec();
    let Some((tiled, perm)) = head_layout(&shape, key_heads, rep, head_dim)? else {
        return Ok(o.alias());
    };
    let o = var_contiguous(o)?;
    let o = var_reshape(&o, &tiled).map_err(Error::Numr)?;
    let o = var_permute(&o, &perm).map_err(Error::Numr)?;
    let o = var_contiguous(&o)?;
    var_reshape(&o, &shape).map_err(Error::Numr)
}

/// The tiled shape `[.., rep, key_heads, head_dim]` and the permutation
/// that swaps its two head axes. `None` when `rep == 1`, where the two
/// orders agree.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `shape` is empty or its last dim is not
/// `key_heads * rep * head_dim`.
fn head_layout(
    shape: &[usize],
    key_heads: usize,
    rep: usize,
    head_dim: usize,
) -> Result<Option<(Vec<usize>, Vec<usize>)>> {
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
        return Ok(None);
    }
    let mut tiled = lead.to_vec();
    tiled.extend([rep, key_heads, head_dim]);
    let n = lead.len();
    let mut perm: Vec<usize> = (0..n).collect();
    perm.extend([n + 1, n, n + 2]);
    Ok(Some((tiled, perm)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::autograd::{backward, var_mul, var_sum};
    use numr::dtype::DType;
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
        assert!(group_heads_var(&Var::new(o, true), 2, 2, 2).is_err());
    }

    /// The tracked twin gives the same values, and the gradient of
    /// `sum(group_heads(o) * w)` is `w` moved back to tiled order.
    #[test]
    fn group_heads_var_matches_and_routes_the_gradient_back() {
        let (client, device) = cpu_setup();
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let o = Tensor::<CpuRuntime>::from_slice(&data, &[2, 8], &device).unwrap();
        let want = group_heads(&o, 2, 2, 2).unwrap();
        let ov = Var::new(o, true);
        let grouped = group_heads_var(&ov, 2, 2, 2).unwrap();
        assert_eq!(grouped.shape(), &[2, 8]);
        assert_eq!(grouped.tensor().to_vec::<f32>(), want.to_vec::<f32>());

        let w: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let wt = Tensor::<CpuRuntime>::from_slice(&w, &[2, 8], &device).unwrap();
        let prod = var_mul(&grouped, &Var::new(wt.clone(), false), &client).unwrap();
        let loss = var_sum(&prod, &[0, 1], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();
        let grad = grads.get(ov.id()).unwrap();
        // Grouped and tiled orders swap heads 1 and 2 when H_k = rep = 2,
        // so the permutation is its own inverse here.
        let want_grad = group_heads(&wt, 2, 2, 2).unwrap();
        assert_eq!(
            grad.contiguous().unwrap().to_vec::<f32>(),
            want_grad.to_vec::<f32>()
        );
    }

    #[test]
    fn group_heads_var_keeps_identity_for_rep_one() {
        let device = CpuDevice::new();
        let o = Tensor::<CpuRuntime>::zeros(&[1, 4], DType::F32, &device).unwrap();
        let ov = Var::new(o, true);
        let same = group_heads_var(&ov, 2, 1, 2).unwrap();
        assert_eq!(same.id(), ov.id());
    }
}
