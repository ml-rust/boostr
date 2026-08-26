//! SwiGLU MLP for VoxCPM2's shared bidirectional MiniCPM4 transformer
//! layers, used by both `feat_encoder` (`local_encoder`) and the local DiT
//! (`feat_decoder`).
//!
//! Same math as `LlamaMlp`'s dense fallback (`down_proj(silu(gate_proj(x)) *
//! up_proj(x))`, via numr's fused `var_silu_mul` kernel), built on plain
//! [`Linear`] rather than `MaybeQuantLinear`: VoxCPM2's dense safetensors
//! checkpoint is never GGUF-quantized, so that indirection is dropped.
//! `LlamaMlp` itself is `pub(super)` to `model::llama::model` and not
//! reachable from here.

use crate::error::{Error, Result};
use crate::model::traits::ModelClient;
use crate::nn::Linear;
use numr::autograd::{Var, var_silu_mul};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ScalarOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;

/// `gate_proj`/`up_proj`: 1024 -> 4096, `down_proj`: 4096 -> 1024. All
/// bias-free.
pub struct BidirectionalMlp<R: Runtime> {
    pub(crate) gate_proj: Linear<R>,
    pub(crate) up_proj: Linear<R>,
    pub(crate) down_proj: Linear<R>,
}

impl<R: Runtime<DType = DType>> BidirectionalMlp<R> {
    /// `down_proj(silu(gate_proj(x)) * up_proj(x))`
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + UnaryOps<R>,
    {
        let gate = self.gate_proj.forward(client, x)?;
        let up = self.up_proj.forward(client, x)?;
        let hidden = var_silu_mul(&gate, &up, client).map_err(Error::Numr)?;
        self.down_proj.forward(client, &hidden)
    }
}
