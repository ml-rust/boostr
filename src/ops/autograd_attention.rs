//! Autograd integration for Flash Attention
//!
//! Wraps FlashAttentionOps (Tensor-level) into Var-level operations
//! for seamless integration with numr's autograd graph.

use crate::error::Result;
use crate::ops::traits::FlashAttentionOps;
use numr::autograd::{GradFn, TensorId, Var};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::sync::Arc;

/// Configuration saved from forward pass for backward.
struct FlashAttentionConfig {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    causal: bool,
    window_size: usize,
}

/// Backward function for Flash Attention.
///
/// Saved state: Q, K, V, output, LSE from forward pass.
/// Computes dQ, dK, dV via `FlashAttentionOps::flash_attention_bwd`.
struct FlashAttentionBackward<R: Runtime> {
    input_ids: [TensorId; 3],      // q, k, v
    saved_tensors: Vec<Tensor<R>>, // [q, k, v, output, lse]
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 3],
    config: FlashAttentionConfig,
}

impl<R: Runtime> GradFn<R> for FlashAttentionBackward<R>
where
    R::Client: FlashAttentionOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> numr::error::Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());
        let q = &self.saved_tensors[0];
        let k = &self.saved_tensors[1];
        let v = &self.saved_tensors[2];
        let output = &self.saved_tensors[3];
        let lse = &self.saved_tensors[4];
        let cfg = &self.config;

        let (dq, dk, dv) = client
            .flash_attention_bwd(
                grad_output,
                q,
                k,
                v,
                output,
                lse,
                cfg.num_heads,
                cfg.num_kv_heads,
                cfg.head_dim,
                cfg.causal,
                cfg.window_size,
            )
            .map_err(|e| {
                numr::error::Error::Internal(format!("flash_attention_bwd failed: {}", e))
            })?;

        Ok(vec![Some(dq), Some(dk), Some(dv)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> numr::error::Result<Vec<Option<Var<R>>>> {
        // First-order only — wrap Tensor results as detached Vars
        let grads = self.backward(grad_output.tensor())?;
        Ok(grads
            .into_iter()
            .map(|g| g.map(|t| Var::new(t, false)))
            .collect())
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.to_vec()
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        &self.saved_tensors
    }

    fn name(&self) -> &'static str {
        "FlashAttentionBackward"
    }
}

/// Flash Attention forward with autograd tracking.
///
/// Wraps `FlashAttentionOps::flash_attention_fwd` into a `Var`-level operation.
/// When any of Q, K, V requires grad, the backward pass is registered.
///
/// Returns `Var<R>` (output only — LSE is internal to the backward).
#[allow(clippy::too_many_arguments)]
pub fn var_flash_attention<R>(
    q: &Var<R>,
    k: &Var<R>,
    v: &Var<R>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    causal: bool,
    window_size: usize,
) -> Result<Var<R>>
where
    R: Runtime,
    R::Client: FlashAttentionOps<R>,
{
    let client = R::default_client(q.tensor().device());

    let (output, lse) = client.flash_attention_fwd(
        q.tensor(),
        k.tensor(),
        v.tensor(),
        num_heads,
        num_kv_heads,
        head_dim,
        causal,
        window_size,
    )?;

    if q.requires_grad() || k.requires_grad() || v.requires_grad() {
        let grad_fn = FlashAttentionBackward {
            input_ids: [q.id(), k.id(), v.id()],
            saved_tensors: vec![
                q.tensor().clone(),
                k.tensor().clone(),
                v.tensor().clone(),
                output.clone(),
                lse,
            ],
            input_grad_fns: [
                q.grad_fn().cloned(),
                k.grad_fn().cloned(),
                v.grad_fn().cloned(),
            ],
            config: FlashAttentionConfig {
                num_heads,
                num_kv_heads,
                head_dim,
                causal,
                window_size,
            },
        };
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}
