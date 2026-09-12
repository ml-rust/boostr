//! [`AuxProjections`]: the six root-level projections around `fsq_layer`,
//! including the `stop` classifier chain that shares its input width.

use crate::error::{Error, Result};
use crate::nn::{
    LoraTargets, MaybeLoraLinear, Module, adapt_if_targeted, child_params, extend_named,
    load_lora_child, push_projection_name,
};
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::{Var, var_silu};
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, ScalarOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// The six auxiliary projections around `fsq_layer` that a future
/// `VoxCpm2Model` orchestrator will own: encoder/DiT bridges and the stop
/// classifier. See [`crate::model::audio::voxcpm::fsq::loader`] for the
/// checkpoint key layout each field is loaded from.
///
/// All six are [`MaybeLoraLinear`] for the same reason
/// [`ScalarQuantization`](super::ScalarQuantization)'s pair is: a GGUF stores them block-quantized and
/// they multiply PACKED, while a safetensors checkpoint yields the
/// `Standard` variant and the dense path is unchanged. `MaybeLoraLinear`
/// additionally lets any of the six carry a LoRA adapter.
pub struct AuxProjections<R: Runtime> {
    pub enc_to_lm_proj: MaybeLoraLinear<R>,
    pub lm_to_dit_proj: MaybeLoraLinear<R>,
    pub res_to_dit_proj: MaybeLoraLinear<R>,
    pub fusion_concat_proj: MaybeLoraLinear<R>,
    pub stop_proj: MaybeLoraLinear<R>,
    /// Bias-free: the checkpoint carries no `stop_head.bias` tensor. See
    /// [`crate::model::audio::voxcpm::fsq::loader`] for how this is loaded.
    pub stop_head: MaybeLoraLinear<R>,
}

impl<R: Runtime<DType = DType>> AuxProjections<R> {
    /// `stop_head(silu(stop_proj(hidden)))`: the fixed composition the
    /// reference always runs together to produce stop-token logits.
    pub fn stop<C>(&self, client: &C, hidden: &Var<R>) -> Result<Var<R>>
    where
        // The extra three bounds over a dense `Linear::forward` — see
        // [`ScalarQuantization::forward`].
        C: RuntimeClient<R>
            + TensorOps<R>
            + ActivationOps<R>
            + ScalarOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>,
        R::Client: TensorOps<R> + ActivationOps<R> + ScalarOps<R> + BinaryOps<R> + DequantOps<R>,
    {
        let projected = self.stop_proj.forward(client, hidden)?;
        let activated = var_silu(&projected, client).map_err(Error::Numr)?;
        self.stop_head.forward(client, &activated)
    }

    /// Wrap any of the six projections that `targets` names with a fresh
    /// LoRA adapter, returning how many were adapted. `prefix` is passed
    /// straight through with NO segment appended — these six live at the
    /// checkpoint ROOT with no shared prefix (see the struct doc and
    /// [`Module::named_parameters`] above), so the owning
    /// [`VoxCpm2Model`](crate::model::audio::voxcpm::model::VoxCpm2Model)
    /// calls this the same way it calls `named_parameters` on `aux`: with
    /// whatever prefix IT was itself given, unchanged. A leaf step: no
    /// zero-match check here — see
    /// [`crate::model::audio::voxcpm::minicpm4::MiniCpm4Attention::apply_lora`]'s
    /// doc comment for why.
    pub fn apply_lora(
        &mut self,
        targets: &LoraTargets,
        rank: usize,
        alpha: f32,
        device: &R::Device,
        prefix: &str,
    ) -> Result<usize> {
        let mut adapted = adapt_if_targeted(
            &mut self.enc_to_lm_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "enc_to_lm_proj",
        )?;
        adapted += adapt_if_targeted(
            &mut self.lm_to_dit_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "lm_to_dit_proj",
        )?;
        adapted += adapt_if_targeted(
            &mut self.res_to_dit_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "res_to_dit_proj",
        )?;
        adapted += adapt_if_targeted(
            &mut self.fusion_concat_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "fusion_concat_proj",
        )?;
        adapted += adapt_if_targeted(
            &mut self.stop_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "stop_proj",
        )?;
        adapted += adapt_if_targeted(
            &mut self.stop_head,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "stop_head",
        )?;
        Ok(adapted)
    }

    /// Every dotted projection path [`Self::apply_lora`] would adapt under
    /// `prefix` — the six root-level projections, `prefix` passed straight
    /// through with NO segment appended, exactly as [`Self::apply_lora`]
    /// does — INDEPENDENT of whether any of the six is dense,
    /// block-quantized, or decomposed-quantized. Unlike `named_parameters()`,
    /// this never enumerates empty for a quantized projection: which
    /// projections exist is a STRUCTURAL property of this type, not a
    /// function of whether its weights happen to carry a `Var<R>`. Built
    /// with the same [`crate::nn::push_projection_name`] helper
    /// `apply_lora`'s [`adapt_if_targeted`] calls use, so a path here is
    /// never hand-written separately from the one `apply_lora` matches.
    pub fn lora_projection_names(&self, prefix: &str) -> Vec<String> {
        let mut names = Vec::new();
        push_projection_name(&mut names, prefix, "enc_to_lm_proj");
        push_projection_name(&mut names, prefix, "lm_to_dit_proj");
        push_projection_name(&mut names, prefix, "res_to_dit_proj");
        push_projection_name(&mut names, prefix, "fusion_concat_proj");
        push_projection_name(&mut names, prefix, "stop_proj");
        push_projection_name(&mut names, prefix, "stop_head");
        names
    }

    /// Write back updated adapter values across all six projections from an
    /// optimizer's `params` map, keeping each adapter's [`TensorId`]s. See
    /// [`crate::nn::MaybeLoraLinear::load_lora_parameters`] for the
    /// per-projection semantics. No prefix needed — unlike
    /// [`Self::apply_lora`], lookup is by ID.
    pub fn load_lora_parameters(
        &mut self,
        params: &std::collections::HashMap<TensorId, Tensor<R>>,
    ) -> Result<usize> {
        let mut written = load_lora_child(&mut self.enc_to_lm_proj, params, "enc_to_lm_proj")?;
        written += load_lora_child(&mut self.lm_to_dit_proj, params, "lm_to_dit_proj")?;
        written += load_lora_child(&mut self.res_to_dit_proj, params, "res_to_dit_proj")?;
        written += load_lora_child(&mut self.fusion_concat_proj, params, "fusion_concat_proj")?;
        written += load_lora_child(&mut self.stop_proj, params, "stop_proj")?;
        written += load_lora_child(&mut self.stop_head, params, "stop_head")?;
        Ok(written)
    }

    /// Set every attached adapter's `lora_a`/`lora_b` to `trainable` across
    /// all six projections, returning how many carry an adapter. See
    /// [`crate::nn::LoraLinear::set_trainable`] for why inference must
    /// freeze a file-loaded adapter. Unlike [`Self::apply_lora`], this
    /// touches only ALREADY-adapted projections and needs no `prefix` — it
    /// is a blanket toggle, not a name match.
    pub fn set_lora_trainable(&mut self, trainable: bool) -> usize {
        let mut touched = 0;
        for proj in [
            &mut self.enc_to_lm_proj,
            &mut self.lm_to_dit_proj,
            &mut self.res_to_dit_proj,
            &mut self.fusion_concat_proj,
            &mut self.stop_proj,
            &mut self.stop_head,
        ] {
            if proj.is_adapted() {
                proj.set_trainable(trainable);
                touched += 1;
            }
        }
        touched
    }
}

/// Names ARE the checkpoint root-level keys verbatim (`enc_to_lm_proj`,
/// `lm_to_dit_proj`, `res_to_dit_proj`, `fusion_concat_proj`, `stop_proj`,
/// `stop_head`) — these six live at the checkpoint root with no shared
/// prefix (see [`crate::model::audio::voxcpm::fsq::loader`]), so the
/// top-level [`VoxCpm2Model`](crate::model::audio::voxcpm::model::VoxCpm2Model)
/// composition adds NO prefix here, unlike every other sub-model.
impl<R: Runtime<DType = DType>> Module<R> for AuxProjections<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        let mut params = child_params(&self.enc_to_lm_proj);
        params.extend(child_params(&self.lm_to_dit_proj));
        params.extend(child_params(&self.res_to_dit_proj));
        params.extend(child_params(&self.fusion_concat_proj));
        params.extend(child_params(&self.stop_proj));
        params.extend(child_params(&self.stop_head));
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        extend_named(
            &mut params,
            "enc_to_lm_proj",
            self.enc_to_lm_proj.named_parameters(),
        );
        extend_named(
            &mut params,
            "lm_to_dit_proj",
            self.lm_to_dit_proj.named_parameters(),
        );
        extend_named(
            &mut params,
            "res_to_dit_proj",
            self.res_to_dit_proj.named_parameters(),
        );
        extend_named(
            &mut params,
            "fusion_concat_proj",
            self.fusion_concat_proj.named_parameters(),
        );
        extend_named(&mut params, "stop_proj", self.stop_proj.named_parameters());
        extend_named(&mut params, "stop_head", self.stop_head.named_parameters());
        params
    }
}
