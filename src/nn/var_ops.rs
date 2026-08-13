//! Shared `Var` layout helpers used across attention and SSM blocks.

use numr::autograd::{GradFn, Var};
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// Make a `Var` contiguous (copies data only if the layout is non-contiguous).
///
/// Preserves autograd identity and gradient flow through the layout copy.
pub fn var_contiguous<R: Runtime>(v: &Var<R>) -> crate::error::Result<Var<R>> {
    if v.tensor().is_contiguous() {
        return Ok(var_identity(v));
    }

    let tensor = v.tensor().contiguous().map_err(crate::error::Error::Numr)?;
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
    fn backward(&self, grad_output: &Tensor<R>) -> numr::error::Result<Vec<Option<Tensor<R>>>> {
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
pub fn repeat_kv<R: Runtime>(x: &Var<R>, repeat: usize) -> numr::error::Result<Var<R>> {
    if repeat == 1 {
        return Ok(x.clone());
    }
    let shape = x.shape();
    let [b, h_kv, s, d] = [shape[0], shape[1], shape[2], shape[3]];

    // Contiguous required: reshape needs contiguous layout, and inputs
    // (e.g. V after permute) may be strided.
    let expanded = x.tensor().contiguous()?.reshape(&[b, h_kv, 1, s, d])?;
    let expanded = expanded.broadcast_to(&[b, h_kv, repeat, s, d])?;
    let result = expanded.contiguous()?.reshape(&[b, h_kv * repeat, s, d])?;
    Ok(Var::new(result, x.requires_grad()))
}
