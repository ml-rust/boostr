//! Autograd bridge for the block-quantized `quant_matmul` forward.
//!
//! [`crate::nn::linear::QuantLinear::forward`] computes `quant_matmul(x, W_q)`
//! straight from block-quantized storage — the fast kernel used everywhere at
//! inference. The base weight `W_q` is FROZEN (a GGUF checkpoint is never
//! trained), so it needs no gradient of its own and is never a `Var` input to
//! this node. Only the gradient with respect to the INPUT `x` must flow back,
//! so a LoRA adapter feeding into a quantized projection — or receiving
//! gradient through one — still trains. This is exactly the QLoRA
//! arrangement: forward stays on the fast quantized kernel, backward
//! dequantizes.
//!
//! The dequantize in [`QuantLinearBackward::backward`] runs ONLY when
//! `backward()` is actually called, i.e. only during training. Inference
//! never touches this module: a `requires_grad == false` input keeps using
//! the cheap detached `Var::new(out, false)` path in `MaybeQuantLinear::forward`,
//! so inference memory and speed are unchanged.

use crate::error::Result;
use crate::quant::tensor::QuantTensor;
use crate::quant::traits::DequantOps;
use numr::autograd::{GradFn, Var};
use numr::dtype::DType;
use numr::ops::MatmulOps;
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// Backward node for `quant_matmul(x, W_q)`.
///
/// `W_q` is logically `[N, K] = [out_features, in_features]`; `x` is
/// `[..., K]` and the forward output is `[..., N]`
/// ([`crate::quant::traits::QuantMatmulOps::quant_matmul`]'s own contract).
/// The matmul adjoint w.r.t. the input is `grad_x = grad_out @ W`,
/// contracting `grad_out`'s trailing `N` against `W`'s leading `N` to give
/// `[..., K]` — the same shape as `x`.
///
/// `W_q`'s storage is `Arc`-shared, so holding a copy here (built via
/// [`QuantTensor::from_storage`]) is a refcount bump, not a data copy.
struct QuantLinearBackward<R: Runtime> {
    input_id: [TensorId; 1],
    input_grad_fn: [Option<Arc<dyn GradFn<R>>>; 1],
    weight: QuantTensor<R>,
}

impl<R: Runtime<DType = DType>> GradFn<R> for QuantLinearBackward<R>
where
    R::Client: DequantOps<R> + MatmulOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        needed: &[bool],
    ) -> numr::error::Result<Vec<Option<Tensor<R>>>> {
        if !needed[0] {
            return Ok(vec![None]);
        }

        let client = R::default_client(grad_output.device());

        // `reshape` needs a row-major buffer; the autograd engine hands down
        // whatever layout the consumer's backward produced.
        let grad_output = grad_output.contiguous()?;
        let out_shape = grad_output.shape().to_vec();
        let n = *out_shape.last().ok_or_else(|| {
            numr::error::Error::Internal(
                "QuantLinearBackward: grad_output must be at least 1D".into(),
            )
        })?;
        let leading: usize = out_shape[..out_shape.len() - 1].iter().product();
        let flat_grad = grad_output.reshape(&[leading, n])?;

        // Backward-only dequantize: the fast quantized kernel ran in
        // forward, so this is the one place the full weight is materialized.
        // `dequantize` lives in boostr's error domain; this node lives in
        // numr's (the `GradFn` trait it implements is numr's), so the error
        // is folded across the boundary here rather than plumbing a second
        // error type through the whole autograd engine.
        let w = client
            .dequantize(&self.weight, DType::F32)
            .map_err(|e| numr::error::Error::Internal(format!("QuantLinearBackward: {e}")))?;
        let flat_grad_x = client.matmul(&flat_grad, &w)?;

        let k = *self.weight.shape().get(1).ok_or_else(|| {
            numr::error::Error::Internal(
                "QuantLinearBackward: weight must be 2D [out_features, in_features]".into(),
            )
        })?;
        let mut grad_shape = out_shape;
        let last = grad_shape.len() - 1;
        grad_shape[last] = k;
        let grad_x = flat_grad_x.reshape(&grad_shape)?;

        Ok(vec![Some(grad_x)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> numr::error::Result<Vec<Option<Var<R>>>> {
        // First-order only — wrap the Tensor result in a detached Var.
        // Second-order traversal keeps every node, so ask for every gradient.
        let grads = self.backward_all(grad_output.tensor())?;
        Ok(grads
            .into_iter()
            .map(|g| g.map(|t| Var::new(t, false)))
            .collect())
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_id
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fn.to_vec()
    }

    fn name(&self) -> &'static str {
        "QuantLinearBackward"
    }
}

/// Wrap a `quant_matmul` forward result in a differentiable [`Var`].
///
/// Call this instead of `Var::new(out, false)` whenever `x.requires_grad()`
/// is true. The caller (`MaybeQuantLinear::forward`) keeps the detached path
/// for `requires_grad() == false`, so inference is unaffected — this
/// function is only ever reached during training.
pub fn attach_quant_linear_backward<R>(
    x: &Var<R>,
    output: Tensor<R>,
    weight: &QuantTensor<R>,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    R::Client: DequantOps<R> + MatmulOps<R>,
{
    let weight_ref = QuantTensor::from_storage(
        weight.storage().clone(),
        weight.format(),
        weight.shape(),
        weight.device(),
    )?;

    let grad_fn = QuantLinearBackward {
        input_id: [x.id()],
        input_grad_fn: [x.grad_fn().cloned()],
        weight: weight_ref,
    };
    Ok(Var::from_op(output, Arc::new(grad_fn)))
}
