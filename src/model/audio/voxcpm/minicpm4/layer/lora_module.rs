use super::MiniCpm4Layer;
use crate::error::Result;
use crate::nn::{LoraTargets, Module, child_params, extend_named};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};

impl<R: Runtime<DType = DType>> MiniCpm4Layer<R> {
    /// Delegate to [`MiniCpm4Attention::apply_lora`](super::super::attention::MiniCpm4Attention::apply_lora)
    /// and [`MiniCpm4Mlp::apply_lora`](super::super::mlp::MiniCpm4Mlp::apply_lora),
    /// summing their counts. `prefix` is the dotted path the owning
    /// [`super::super::model::MiniCpm4Model`] would pass to `extend_named` for this
    /// layer — `"layers.{i}"` — extended here by `"self_attn"`/`"mlp"`
    /// exactly as `Module::named_parameters` extends it above. No zero-match
    /// check here: see `MiniCpm4Attention::apply_lora`'s doc comment.
    pub fn apply_lora(
        &mut self,
        targets: &LoraTargets,
        rank: usize,
        alpha: f32,
        device: &R::Device,
        prefix: &str,
    ) -> Result<usize> {
        let mut adapted = self.self_attn.apply_lora(
            targets,
            rank,
            alpha,
            device,
            &LoraTargets::join(prefix, "self_attn"),
        )?;
        adapted += self.mlp.apply_lora(
            targets,
            rank,
            alpha,
            device,
            &LoraTargets::join(prefix, "mlp"),
        )?;
        Ok(adapted)
    }

    /// Every dotted projection path [`Self::apply_lora`] would adapt under
    /// `prefix`, delegating to `MiniCpm4Attention::lora_projection_names`
    /// and `MiniCpm4Mlp::lora_projection_names` at the SAME
    /// `"self_attn"`/`"mlp"`-joined prefixes [`Self::apply_lora`] passes
    /// them, so a path here is never built by separately hand-written logic.
    pub fn lora_projection_names(&self, prefix: &str) -> Vec<String> {
        let mut names = self
            .self_attn
            .lora_projection_names(&LoraTargets::join(prefix, "self_attn"));
        names.extend(
            self.mlp
                .lora_projection_names(&LoraTargets::join(prefix, "mlp")),
        );
        names
    }

    /// Delegate to `MiniCpm4Attention::load_lora_parameters` and
    /// `MiniCpm4Mlp::load_lora_parameters`, summing their counts. No prefix
    /// needed — unlike [`Self::apply_lora`], lookup is by ID, not by dotted
    /// path.
    pub fn load_lora_parameters(
        &mut self,
        params: &std::collections::HashMap<TensorId, Tensor<R>>,
    ) -> Result<usize> {
        let mut written = self.self_attn.load_lora_parameters(params)?;
        written += self.mlp.load_lora_parameters(params)?;
        Ok(written)
    }

    /// Delegate to `MiniCpm4Attention::set_lora_trainable` and
    /// `MiniCpm4Mlp::set_lora_trainable`, summing their counts. No
    /// `targets`/`prefix` needed — unlike [`Self::apply_lora`], this is a
    /// blanket toggle over whatever is already adapted.
    pub fn set_lora_trainable(&mut self, trainable: bool) -> usize {
        self.self_attn.set_lora_trainable(trainable) + self.mlp.set_lora_trainable(trainable)
    }
}

/// Names ARE the field names (`input_layernorm`, `self_attn.*`,
/// `post_attention_layernorm`, `mlp.*`) — this matches the
/// `{prefix}.layers.{i}.*` checkpoint layout
/// ([`crate::model::audio::voxcpm::minicpm4::loader`]) exactly, so the
/// owning [`MiniCpm4Model`](super::super::model::MiniCpm4Model) need only prefix
/// by `layers.{i}` to reach the full checkpoint key.
impl<R: Runtime<DType = DType>> Module<R> for MiniCpm4Layer<R> {
    fn parameters(&self) -> Vec<&numr::autograd::Var<R>> {
        let mut params = child_params(&self.input_layernorm);
        params.extend(child_params(&self.self_attn));
        params.extend(child_params(&self.post_attention_layernorm));
        params.extend(child_params(&self.mlp));
        params
    }

    fn named_parameters(&self) -> Vec<(String, &numr::autograd::Var<R>)> {
        let mut params = Vec::new();
        extend_named(
            &mut params,
            "input_layernorm",
            self.input_layernorm.named_parameters(),
        );
        extend_named(&mut params, "self_attn", self.self_attn.named_parameters());
        extend_named(
            &mut params,
            "post_attention_layernorm",
            self.post_attention_layernorm.named_parameters(),
        );
        extend_named(&mut params, "mlp", self.mlp.named_parameters());
        params
    }
}
