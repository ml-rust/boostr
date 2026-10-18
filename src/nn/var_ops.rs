//! Shared `Var` layout helpers used across attention and SSM blocks.

use numr::autograd::{GradFn, Var};
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// Make a `Var` contiguous (copies data only if the layout is non-contiguous).
///
/// Preserves autograd identity and gradient flow through the layout copy.
pub fn var_contiguous<R: Runtime>(v: &Var<R>) -> crate::error::Result<Var<R>> {
    var_contiguous_numr(v).map_err(crate::error::Error::Numr)
}

/// Same as [`var_contiguous`], in numr's error domain.
///
/// Exists so helpers that must return `numr::error::Result` can preserve
/// gradient flow without duplicating the backward node.
fn var_contiguous_numr<R: Runtime>(v: &Var<R>) -> numr::error::Result<Var<R>> {
    if v.tensor().is_contiguous() {
        return Ok(var_identity(v));
    }

    let tensor = v.tensor().contiguous()?;
    if v.requires_grad() {
        Ok(Var::from_op(
            tensor,
            Arc::new(ContiguousBackward {
                inputs: [v.id()],
                input_grad_fn: v.grad_fn().cloned(),
            }),
        ))
    } else {
        Ok(Var::new(tensor, false))
    }
}

fn var_identity<R: Runtime>(v: &Var<R>) -> Var<R> {
    match (v.requires_grad(), v.grad_fn().cloned()) {
        (true, Some(grad_fn)) => {
            Var::with_id_and_grad_fn(v.tensor().clone(), v.id(), Some(grad_fn))
        }
        (true, None) => Var::with_id(v.tensor().clone(), v.id(), true),
        (false, _) => Var::with_id(v.tensor().clone(), v.id(), false),
    }
}

struct ContiguousBackward<R: Runtime> {
    inputs: [TensorId; 1],
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> GradFn<R> for ContiguousBackward<R> {
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> numr::error::Result<Vec<Option<Tensor<R>>>> {
        // Single input, and the gradient is a refcount clone — nothing to skip.
        Ok(vec![Some(grad_output.clone())])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> numr::error::Result<Vec<Option<Var<R>>>> {
        Ok(vec![Some(grad_output.clone())])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.inputs
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "contiguous"
    }
}

/// Repeat KV heads for GQA: `[B, H_kv, S, D]` -> `[B, H_kv * repeat, S, D]`.
///
/// Every step is tracked. Doing the reshape/broadcast on `x.tensor()` and
/// re-wrapping with `Var::new` produces a LEAF, which severs the graph and
/// leaves `k_proj`/`v_proj` without gradients in every GQA model.
pub fn repeat_kv<R: Runtime>(x: &Var<R>, repeat: usize) -> numr::error::Result<Var<R>>
where
    R::Client: numr::ops::TensorOps<R>,
{
    if repeat == 1 {
        // `alias`, not `clone`: `Var::clone` mints a fresh TensorId, which would
        // orphan the input's gradient/optimizer state on the no-op path.
        return Ok(x.alias());
    }
    let shape = x.shape();
    let [b, h_kv, s, d] = [shape[0], shape[1], shape[2], shape[3]];

    // Contiguous required: reshape needs contiguous layout, and inputs
    // (e.g. V after permute) may be strided.
    let expanded = var_contiguous_numr(x)?;
    let expanded = numr::autograd::var_reshape(&expanded, &[b, h_kv, 1, s, d])?;
    let expanded = numr::autograd::var_broadcast_to(&expanded, &[b, h_kv, repeat, s, d])?;
    let expanded = var_contiguous_numr(&expanded)?;
    numr::autograd::var_reshape(&expanded, &[b, h_kv * repeat, s, d])
}

/// Repeat heads in tiled order: `[b, s, H_k, S]` -> `[b, s, H_v, S]`.
///
/// With `rep = H_v / H_k`, output head `h` reads input head `h % H_k`.
/// [`repeat_kv`] groups copies instead: output head `h` reads `h / rep`.
/// Every step is tracked, so gradient flows back to the input.
///
/// # Errors
///
/// `InvalidArgument` when the input is not 4-D, `H_k` is 0, or `H_v` is not a
/// multiple of `H_k`.
pub fn repeat_heads_tiled<R: Runtime>(x: &Var<R>, h_v: usize) -> numr::error::Result<Var<R>>
where
    R::Client: numr::ops::TensorOps<R>,
{
    let shape = x.shape();
    if shape.len() != 4 {
        return Err(numr::error::Error::InvalidArgument {
            arg: "x",
            reason: format!("expected [b, s, heads, size], got {shape:?}"),
        });
    }
    let [b, s, h_k, size] = [shape[0], shape[1], shape[2], shape[3]];
    if h_k == 0 || !h_v.is_multiple_of(h_k) {
        return Err(numr::error::Error::InvalidArgument {
            arg: "h_v",
            reason: format!("target heads {h_v} must be a multiple of input heads {h_k}"),
        });
    }
    let rep = h_v / h_k;
    if rep == 1 {
        // `alias` keeps the input's TensorId, so gradient state is not orphaned.
        return Ok(x.alias());
    }

    let expanded = var_contiguous_numr(x)?;
    let expanded = numr::autograd::var_reshape(&expanded, &[b, s, 1, h_k, size])?;
    let expanded = numr::autograd::var_broadcast_to(&expanded, &[b, s, rep, h_k, size])?;
    let expanded = var_contiguous_numr(&expanded)?;
    numr::autograd::var_reshape(&expanded, &[b, s, h_v, size])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::autograd::backward;
    use numr::runtime::cpu::CpuRuntime;

    /// GQA head repetition must stay on the autograd graph.
    ///
    /// Regression: this ran reshape/broadcast on `x.tensor()` and re-wrapped the
    /// result with `Var::new`, producing a LEAF. Since every grouped-query
    /// attention block repeats K and V through here, `k_proj` and `v_proj`
    /// received no gradient at all — the model trained, the loss fell, and those
    /// projections stayed frozen at their initial values.
    #[test]
    fn repeat_kv_preserves_gradient_flow() {
        let (client, device) = cpu_setup();

        // [B=1, H_kv=2, S=2, D=2]
        let kv = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &[0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5, 0.1, -0.3],
                &[1, 2, 2, 2],
                &device,
            )
            .unwrap(),
            true,
        );

        let repeated = repeat_kv(&kv, 3).unwrap();
        assert_eq!(repeated.shape(), &[1, 6, 2, 2]);

        let loss = numr::autograd::var_sum(&repeated, &[0, 1, 2, 3], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let grad = grads
            .get(kv.id())
            .expect("repeat_kv must propagate gradient back to the KV input");
        let values: Vec<f32> = grad.contiguous().unwrap().to_vec();

        // Each source element is broadcast to `repeat` outputs, so d(sum)/dx = repeat.
        assert_eq!(values, vec![3.0; 8]);
    }

    /// repeat == 1 short-circuits; it must return the same Var, not a fresh leaf.
    #[test]
    fn repeat_kv_identity_preserves_gradient_flow() {
        let (client, device) = cpu_setup();

        let w = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &device)
                .unwrap(),
            true,
        );
        let upstream = numr::autograd::var_mul(&w, &w, &client).unwrap();

        let repeated = repeat_kv(&upstream, 1).unwrap();
        let loss = numr::autograd::var_sum(&repeated, &[0, 1, 2, 3], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let grad = grads
            .get(w.id())
            .expect("repeat_kv(1) must not detach upstream parameters");
        let values: Vec<f32> = grad.contiguous().unwrap().to_vec();
        assert_eq!(values, vec![2.0, 4.0, 6.0, 8.0]);
    }

    /// Tiled repeat must stay on the autograd graph.
    #[test]
    fn repeat_heads_tiled_preserves_gradient_flow() {
        let (client, device) = cpu_setup();

        // [b=1, s=2, H_k=2, S=2]
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &[0.5f32, -1.0, 2.0, 0.25, -0.75, 1.5, 0.1, -0.3],
                &[1, 2, 2, 2],
                &device,
            )
            .unwrap(),
            true,
        );

        let repeated = repeat_heads_tiled(&x, 6).unwrap();
        assert_eq!(repeated.shape(), &[1, 2, 6, 2]);

        let loss = numr::autograd::var_sum(&repeated, &[0, 1, 2, 3], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let grad = grads
            .get(x.id())
            .expect("repeat_heads_tiled must propagate gradient back to the input");
        let values: Vec<f32> = grad.contiguous().unwrap().to_vec();
        assert_eq!(values, vec![3.0; 8]);
    }

    /// Output head `h` reads input head `h % H_k`, unlike the grouped order.
    #[test]
    fn repeat_heads_tiled_uses_tiled_order() {
        let (_client, device) = cpu_setup();

        // [1, 1, H_k=2, S=2]: head 0 = [1, 2], head 1 = [3, 4].
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &device)
                .unwrap(),
            false,
        );

        let tiled = repeat_heads_tiled(&x, 4).unwrap();
        assert_eq!(tiled.shape(), &[1, 1, 4, 2]);
        let tiled: Vec<f32> = tiled.tensor().contiguous().unwrap().to_vec();
        assert_eq!(tiled, vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);

        // Same data as `[b, H, s, d]`: `repeat_kv` groups heads as [h0, h0, h1, h1].
        let xg = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 1, 2], &device)
                .unwrap(),
            false,
        );
        let grouped = repeat_kv(&xg, 2).unwrap();
        let grouped: Vec<f32> = grouped.tensor().contiguous().unwrap().to_vec();
        assert_eq!(grouped, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
        assert_ne!(tiled, grouped);
    }

    /// rep == 1 returns the same Var, and a non-multiple is an error.
    #[test]
    fn repeat_heads_tiled_identity_and_bad_ratio() {
        let (client, device) = cpu_setup();

        let w = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &device)
                .unwrap(),
            true,
        );
        let upstream = numr::autograd::var_mul(&w, &w, &client).unwrap();

        let same = repeat_heads_tiled(&upstream, 2).unwrap();
        assert_eq!(same.shape(), &[1, 1, 2, 2]);
        let loss = numr::autograd::var_sum(&same, &[0, 1, 2, 3], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();
        let grad = grads
            .get(w.id())
            .expect("rep == 1 must not detach upstream");
        let values: Vec<f32> = grad.contiguous().unwrap().to_vec();
        assert_eq!(values, vec![2.0, 4.0, 6.0, 8.0]);

        assert!(repeat_heads_tiled(&upstream, 3).is_err());
    }
}
