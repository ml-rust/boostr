//! SwiGLU MLP for VoxCPM2's shared bidirectional MiniCPM4 transformer
//! layers, used by both `feat_encoder` (`local_encoder`) and the local DiT
//! (`feat_decoder`).
//!
//! Same math as `LlamaMlp`'s dense fallback (`down_proj(silu(gate_proj(x)) *
//! up_proj(x))`, via numr's fused `var_silu_mul` kernel), built on
//! [`MaybeQuantLinear`]: VoxCPM2 ships BOTH dense safetensors and GGUF, and
//! in a GGUF these three projections are block-quantized. The quantized
//! variant multiplies the weight PACKED through `quant_matmul`, so a Q4_K
//! file is not expanded to dense F32 at load; a safetensors checkpoint
//! yields the `Standard` variant and the dense path is unchanged. This block
//! stack is shared, so one conversion here covers BOTH `feat_encoder` and
//! `local_dit`. `LlamaMlp` itself is `pub(super)` to `model::llama::model`
//! and not reachable from here.

use crate::error::{Error, Result};
use crate::model::traits::ModelClient;
use crate::nn::MaybeQuantLinear;
use numr::autograd::{Var, var_silu_mul};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ScalarOps, TensorOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::Runtime;

/// `gate_proj`/`up_proj`: 1024 -> 4096, `down_proj`: 4096 -> 1024. All
/// bias-free.
pub struct BidirectionalMlp<R: Runtime> {
    pub(crate) gate_proj: MaybeQuantLinear<R>,
    pub(crate) up_proj: MaybeQuantLinear<R>,
    pub(crate) down_proj: MaybeQuantLinear<R>,
}

impl<R: Runtime<DType = DType>> BidirectionalMlp<R> {
    /// `down_proj(silu(gate_proj(x)) * up_proj(x))`
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        // `TypeConversionOps` is what `MaybeQuantLinear::forward` adds over a
        // dense `Linear::forward`; `ModelClient` already carries
        // `QuantMatmulOps`.
        C: ModelClient<R> + TypeConversionOps<R>,
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
