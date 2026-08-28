//! SwiGLU MLP for VoxCPM2's MiniCPM4 decoder layers.
//!
//! `down_proj(silu(gate_proj(x)) * up_proj(x))`, via numr's fused
//! `var_silu_mul` kernel. Built on [`MaybeLoraLinear`]: VoxCPM2 ships BOTH
//! dense safetensors and GGUF, and in a GGUF these three projections are
//! block-quantized. The quantized variant multiplies the weight PACKED
//! through `quant_matmul`, so a Q4_K file is not expanded to dense F32 at
//! load; a safetensors checkpoint yields the `Standard` variant and the
//! dense path is unchanged. `MaybeLoraLinear` additionally lets any of the
//! three carry a LoRA adapter. Same shape as the `feat_encoder` sibling's
//! MLP, at the decoder's wider dimensions.

use crate::error::{Error, Result};
use crate::model::traits::ModelClient;
use crate::nn::{
    LoraTargets, MaybeLoraLinear, Module, adapt_if_targeted, child_params, extend_named,
    load_lora_child, push_projection_name,
};
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_silu_mul};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ScalarOps, TensorOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};

/// `gate_proj`/`up_proj`: 2048 -> 6144, `down_proj`: 6144 -> 2048. All
/// bias-free.
pub struct MiniCpm4Mlp<R: Runtime> {
    pub(crate) gate_proj: MaybeLoraLinear<R>,
    pub(crate) up_proj: MaybeLoraLinear<R>,
    pub(crate) down_proj: MaybeLoraLinear<R>,
}

impl<R: Runtime<DType = DType>> MiniCpm4Mlp<R> {
    /// `down_proj(silu(gate_proj(x)) * up_proj(x))`
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        // `TypeConversionOps` is what `MaybeLoraLinear::forward` adds over a
        // dense `Linear::forward`; `ModelClient` already carries
        // `QuantMatmulOps`.
        C: ModelClient<R> + TypeConversionOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + UnaryOps<R>
            + DequantOps<R>,
    {
        let gate = self.gate_proj.forward(client, x)?;
        let up = self.up_proj.forward(client, x)?;
        let hidden = var_silu_mul(&gate, &up, client).map_err(Error::Numr)?;
        self.down_proj.forward(client, &hidden)
    }

    /// Wrap `gate_proj`/`up_proj`/`down_proj` that `targets` names with a
    /// fresh LoRA adapter each, returning how many were adapted. `prefix` is
    /// the dotted path the owning [`super::layer::MiniCpm4Layer`] would pass
    /// to `extend_named` for this block — `"mlp"` — so each projection's
    /// path (via [`LoraTargets::join`]) matches `named_parameters()`'s path
    /// exactly. A leaf step: see
    /// [`MiniCpm4Attention::apply_lora`](super::attention::MiniCpm4Attention::apply_lora)
    /// for why this does not call `LoraTargets::ensure_all_match` itself.
    pub fn apply_lora(
        &mut self,
        targets: &LoraTargets,
        rank: usize,
        alpha: f32,
        device: &R::Device,
        prefix: &str,
    ) -> Result<usize> {
        let mut adapted = adapt_if_targeted(
            &mut self.gate_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "gate_proj",
        )?;
        adapted += adapt_if_targeted(
            &mut self.up_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "up_proj",
        )?;
        adapted += adapt_if_targeted(
            &mut self.down_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "down_proj",
        )?;
        Ok(adapted)
    }

    /// Every dotted projection path [`Self::apply_lora`] would adapt under
    /// `prefix` — `gate_proj`, `up_proj`, `down_proj` — INDEPENDENT of
    /// whether a projection is dense, block-quantized, or
    /// decomposed-quantized. Unlike `named_parameters()`, this never
    /// enumerates empty for a quantized projection: which projections exist
    /// is a STRUCTURAL property of this type, not a function of whether its
    /// weights happen to carry a `Var<R>`. Built with the same
    /// [`crate::nn::push_projection_name`] helper `apply_lora`'s
    /// [`adapt_if_targeted`] calls use, so a path here is never hand-written
    /// separately from the one `apply_lora` matches.
    pub fn lora_projection_names(&self, prefix: &str) -> Vec<String> {
        let mut names = Vec::new();
        push_projection_name(&mut names, prefix, "gate_proj");
        push_projection_name(&mut names, prefix, "up_proj");
        push_projection_name(&mut names, prefix, "down_proj");
        names
    }

    /// Write back updated `gate_proj`/`up_proj`/`down_proj` adapter values
    /// from an optimizer's `params` map, keeping their [`TensorId`]s. See
    /// [`crate::nn::MaybeLoraLinear::load_lora_parameters`] for the
    /// per-projection semantics. No prefix needed — unlike
    /// [`Self::apply_lora`], lookup is by ID.
    pub fn load_lora_parameters(
        &mut self,
        params: &std::collections::HashMap<TensorId, Tensor<R>>,
    ) -> Result<usize> {
        let mut written = load_lora_child(&mut self.gate_proj, params, "gate_proj")?;
        written += load_lora_child(&mut self.up_proj, params, "up_proj")?;
        written += load_lora_child(&mut self.down_proj, params, "down_proj")?;
        Ok(written)
    }

    /// Cheap duplicate that preserves every projection's `Var<R>`
    /// `TensorId`s, for capturing this block by owned value in a `'static`
    /// activation-checkpointing closure — `numr::autograd::checkpoint`'s
    /// closure is `Fn(...) + Send + Sync + 'static`, so a layer cannot be
    /// borrowed into it. Each projection routes through
    /// [`MaybeLoraLinear::alias`], never [`Clone`], so the optimizer, keyed
    /// by `TensorId`, still sees the original parameters' gradients.
    pub fn alias(&self) -> Self {
        Self {
            gate_proj: self.gate_proj.alias(),
            up_proj: self.up_proj.alias(),
            down_proj: self.down_proj.alias(),
        }
    }
}

/// Names ARE the field names (`gate_proj`, `up_proj`, `down_proj`) — the
/// `mlp` checkpoint segment is added by the owning
/// [`MiniCpm4Layer`](super::layer::MiniCpm4Layer).
impl<R: Runtime<DType = DType>> Module<R> for MiniCpm4Mlp<R> {
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
