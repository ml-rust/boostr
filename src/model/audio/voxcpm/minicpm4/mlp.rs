//! SwiGLU MLP for VoxCPM2's MiniCPM4 decoder layers.
//!
//! `down_proj(silu(gate_proj(x)) * up_proj(x))`, via numr's fused
//! `var_silu_mul` kernel. Built on [`MaybeQuantLinear`]: VoxCPM2 ships BOTH
//! dense safetensors and GGUF, and in a GGUF these three projections are
//! block-quantized. The quantized variant multiplies the weight PACKED
//! through `quant_matmul`, so a Q4_K file is not expanded to dense F32 at
//! load; a safetensors checkpoint yields the `Standard` variant and the
//! dense path is unchanged. Same shape as the `feat_encoder` sibling's MLP,
//! at the decoder's wider dimensions.

use crate::error::{Error, Result};
use crate::model::traits::ModelClient;
use crate::nn::{MaybeQuantLinear, Module, child_params, extend_named};
use numr::autograd::{Var, var_silu_mul};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ScalarOps, TensorOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::Runtime;

/// `gate_proj`/`up_proj`: 2048 -> 6144, `down_proj`: 6144 -> 2048. All
/// bias-free.
pub struct MiniCpm4Mlp<R: Runtime> {
    pub(crate) gate_proj: MaybeQuantLinear<R>,
    pub(crate) up_proj: MaybeQuantLinear<R>,
    pub(crate) down_proj: MaybeQuantLinear<R>,
}

impl<R: Runtime<DType = DType>> MiniCpm4Mlp<R> {
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

/// Names ARE the field names (`gate_proj`, `up_proj`, `down_proj`) — the
/// `mlp` checkpoint segment is added by the owning
/// [`MiniCpm4Layer`](super::layer::MiniCpm4Layer).
impl<R: Runtime> Module<R> for MiniCpm4Mlp<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        let mut params = child_params(&self.gate_proj);
        params.extend(child_params(&self.up_proj));
        params.extend(child_params(&self.down_proj));
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        extend_named(&mut params, "gate_proj", self.gate_proj.named_parameters());
        extend_named(&mut params, "up_proj", self.up_proj.named_parameters());
        extend_named(&mut params, "down_proj", self.down_proj.named_parameters());
        params
    }
}
