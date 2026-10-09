//! Loading and running the `Linear` and `LayerNorm` layers of a CLAP
//! checkpoint on plain tensors, for inference.

use crate::error::Result;
use crate::nn::{LayerNorm, Linear, VarBuilder};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{NormalizationOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// The `Linear` at `name.weight` (and `name.bias` when `bias`) under `vb`.
pub(super) fn linear<R: Runtime<DType = DType>>(
    vb: &mut VarBuilder<'_, R>,
    name: &str,
    bias: bool,
) -> Result<Linear<R>> {
    let mut l = vb.pp(name);
    let weight = l.take_tensor("weight")?;
    let bias = if bias {
        Some(l.take_tensor("bias")?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias, false))
}

/// The `LayerNorm` at `name.{weight,bias}` under `vb`.
pub(super) fn layer_norm<R: Runtime<DType = DType>>(
    vb: &mut VarBuilder<'_, R>,
    name: &str,
    eps: f64,
) -> Result<LayerNorm<R>> {
    let mut l = vb.pp(name);
    let weight = l.take_tensor("weight")?;
    let bias = l.take_tensor("bias")?;
    Ok(LayerNorm::new(weight, bias, eps as f32, false))
}

/// `x @ Wᵀ + b` over the last axis of `x`.
pub(super) fn dense<R, C>(client: &C, layer: &Linear<R>, x: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    Ok(layer
        .forward(client, &Var::new(x.clone(), false))?
        .tensor()
        .clone())
}

/// LayerNorm over the last axis of `x`.
pub(super) fn norm<R, C>(client: &C, layer: &LayerNorm<R>, x: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + NormalizationOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    Ok(layer
        .forward(client, &Var::new(x.clone(), false))?
        .tensor()
        .clone())
}
