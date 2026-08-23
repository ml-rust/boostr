//! Autograd integration for the fused GPU RoPE kernels.
//!
//! The CPU backend builds RoPE out of `var_narrow`/`var_mul`/`var_sub`/
//! `var_add`/`var_cat`, so its graph is built by composition. The CUDA and
//! WebGPU backends run a single fused kernel instead, which produces a bare
//! `Tensor` with no graph attached. This module attaches the missing node.
//!
//! # The adjoint
//!
//! Every RoPE variant applies the same 2x2 rotation to a pair `(a, b)`:
//!
//! ```text
//! out_a = a * cos - b * sin
//! out_b = a * sin + b * cos
//! ```
//!
//! The Jacobian of that pair is the rotation matrix `[[c, -s], [s, c]]`, whose
//! transpose is `[[c, s], [-s, c]]` — the SAME rotation with `sin` negated:
//!
//! ```text
//! da = d_a * cos + d_b * sin
//! db = -d_a * sin + d_b * cos
//! ```
//!
//! So the backward is the forward kernel run on the incoming gradient with a
//! negated sine cache. This holds for all three variants:
//!
//! - **standard** — pairs are `(x[d], x[d + D/2])`.
//! - **interleaved** — pairs are `(x[2d], x[2d+1])`. Only the pairing changes;
//!   the per-pair Jacobian is identical, and the pairing is a permutation whose
//!   adjoint is itself.
//! - **yarn** — the rotation is scaled by `attn_scale`, and the YaRN
//!   `attention_factor` is already folded into the caches by
//!   `RoPE::precompute_freqs`. Both are scalar multiples of a linear map, so
//!   the adjoint keeps the same `attn_scale` and the same (negated) cache.
//!
//! `cos_cache`/`sin_cache` are constants: `RoPE::new` wraps them as
//! `Var::new(t, false)` and `precompute_freqs` builds them from host data, so
//! no gradient flows back to them and this node reports exactly one input.

use crate::error::Result;
use crate::ops::traits::RoPEOps;
use numr::autograd::{GradFn, TensorId, Var};
use numr::ops::ScalarOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::sync::Arc;

/// Which fused RoPE kernel produced the output, so the backward can reuse it.
#[derive(Clone, Copy, Debug)]
pub enum RopeVariant {
    /// Split-half pairing `(x[d], x[d + D/2])`.
    Standard,
    /// Adjacent pairing `(x[2d], x[2d+1])` (GPT-NeoX/Qwen style).
    Interleaved,
    /// Split-half pairing with an output scale.
    Yarn {
        /// Attention scaling factor applied to the rotated output.
        attn_scale: f32,
    },
}

impl RopeVariant {
    fn node_name(&self) -> &'static str {
        match self {
            RopeVariant::Standard => "RoPEBackward",
            RopeVariant::Interleaved => "RoPEInterleavedBackward",
            RopeVariant::Yarn { .. } => "RoPEYarnBackward",
        }
    }
}

/// Backward node for the fused RoPE kernels.
///
/// Saved state: the narrowed/cast cosine cache and the NEGATED sine cache. The
/// negation is materialized into a fresh tensor by [`attach_rope_backward`], so
/// the shared cache owned by [`crate::nn::RoPE`] is never mutated.
struct RopeBackward<R: Runtime> {
    input_ids: [TensorId; 1],
    /// `[cos, neg_sin]`
    saved_tensors: Vec<Tensor<R>>,
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 1],
    variant: RopeVariant,
}

impl<R: Runtime> GradFn<R> for RopeBackward<R>
where
    R::Client: RoPEOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> numr::error::Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());

        // The fused kernels index x, cos, sin and out through raw device
        // pointers with implied row-major strides, so every input must be
        // contiguous. The autograd engine hands down whatever layout the
        // consumer's backward produced, which for a permuted epilogue is a
        // strided view. Normalizing is this node's job, not the caller's.
        // `contiguous` is a refcount clone when the tensor already is.
        let grad = Var::new(grad_output.contiguous()?, false);
        let cos = Var::new(self.saved_tensors[0].contiguous()?, false);
        let neg_sin = Var::new(self.saved_tensors[1].contiguous()?, false);

        // `grad` does not require grad, so these calls take the detached
        // branch of the forward and do not re-enter this node.
        let dx = match self.variant {
            RopeVariant::Standard => client.apply_rope(&grad, &cos, &neg_sin),
            RopeVariant::Interleaved => client.apply_rope_interleaved(&grad, &cos, &neg_sin),
            RopeVariant::Yarn { attn_scale } => {
                client.apply_rope_yarn(&grad, &cos, &neg_sin, attn_scale)
            }
        }
        .map_err(|e| {
            numr::error::Error::Internal(format!("{} failed: {}", self.variant.node_name(), e))
        })?;

        Ok(vec![Some(dx.tensor().clone())])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> numr::error::Result<Vec<Option<Var<R>>>> {
        // First-order only — wrap Tensor results as detached Vars.
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
        self.variant.node_name()
    }
}

/// Wrap a fused RoPE kernel result in a differentiable [`Var`].
///
/// Every fused RoPE forward — CUDA and WebGPU, all three variants — ends here
/// instead of `Var::new(output, false)`, which would sever the graph and leave
/// every parameter upstream of RoPE (`q_proj`, `k_proj`, the embedding table)
/// without a gradient.
///
/// `cos` and `sin` must be the caches AS THE KERNEL SAW THEM — already narrowed
/// to `seq_len` and cast to `x`'s dtype — because the backward reruns the same
/// kernel on a gradient of the same shape and dtype as `x`.
///
/// When `x` needs no gradient this returns a detached leaf, exactly as before,
/// so inference pays nothing.
pub fn attach_rope_backward<R>(
    x: &Var<R>,
    output: Tensor<R>,
    cos: &Tensor<R>,
    sin: &Tensor<R>,
    variant: RopeVariant,
) -> Result<Var<R>>
where
    R: Runtime,
    R::Client: RoPEOps<R> + ScalarOps<R>,
{
    if !x.requires_grad() {
        return Ok(Var::new(output, false));
    }

    let client = R::default_client(x.tensor().device());
    // Allocates a new tensor; the cache owned by `RoPE` is left untouched.
    // Negation is exact in every float format, so this costs no precision.
    let neg_sin = client.mul_scalar(sin, -1.0)?;

    let grad_fn = RopeBackward {
        input_ids: [x.id()],
        saved_tensors: vec![cos.clone(), neg_sin],
        input_grad_fns: [x.grad_fn().cloned()],
        variant,
    };
    Ok(Var::from_op(output, Arc::new(grad_fn)))
}
