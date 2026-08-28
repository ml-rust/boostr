//! Single pre-norm decoder layer for VoxCPM2's MiniCPM4.
//!
//! `RmsNorm` -> causal attention -> add, `RmsNorm` -> SwiGLU MLP -> add.
//!
//! Residuals are PLAIN adds: `use_mup` is `false` on this checkpoint, so no
//! muP `scale_depth/sqrt(num_layers)` factor is applied (unlike some
//! MiniCPM-lineage ports that assume it always is). `scale_emb` (12.0) is
//! inactive for the same reason — the reference applies it only under muP —
//! and neither knob has an inert branch here;
//! [`MiniCpm4Config`](crate::model::audio::voxcpm::minicpm4::MiniCpm4Config)
//! rejects a `use_mup=true` checkpoint outright rather than letting this
//! layer compute a different model in silence.

use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::model::audio::voxcpm::minicpm4::attention::MiniCpm4Attention;
use crate::model::audio::voxcpm::minicpm4::mlp::MiniCpm4Mlp;
use crate::model::traits::ModelClient;
use crate::nn::{LoraTargets, Module, RmsNorm, RoPE, child_params, extend_named};
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_add};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};

pub struct MiniCpm4Layer<R: Runtime> {
    pub(crate) input_layernorm: RmsNorm<R>,
    pub(crate) self_attn: MiniCpm4Attention<R>,
    pub(crate) post_attention_layernorm: RmsNorm<R>,
    pub(crate) mlp: MiniCpm4Mlp<R>,
}

impl<R: Runtime<DType = DType>> MiniCpm4Layer<R> {
    /// `x: [batch, seq, hidden]` -> `[batch, seq, hidden]`.
    ///
    /// `rope` is `None` only for a NoPE (`no_rope`) stack, which has no table;
    /// [`MiniCpm4Attention`] rejects a `None` it is not entitled to.
    pub fn forward<C>(&self, client: &C, x: &Var<R>, rope: Option<&RoPE<R>>) -> Result<Var<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        let normed = self.input_layernorm.forward(client, x)?;
        let attn_out = self.self_attn.forward(client, &normed, rope)?;
        let h = var_add(x, &attn_out, client).map_err(Error::Numr)?;

        let normed = self.post_attention_layernorm.forward(client, &h)?;
        let mlp_out = self.mlp.forward(client, &normed)?;
        var_add(&h, &mlp_out, client).map_err(Error::Numr)
    }

    /// KV-cached variant of [`forward`](Self::forward): `x: [batch, seq,
    /// hidden]` covering absolute positions `position..position + seq` ->
    /// `[batch, seq, hidden]`.
    ///
    /// Identical residual/norm/MLP structure — only attention differs, reading
    /// and extending `kv_cache` instead of recomputing the whole prefix.
    pub fn forward_cached<C>(
        &self,
        client: &C,
        x: &Var<R>,
        rope: Option<&RoPE<R>>,
        kv_cache: &mut KvCache<R>,
        position: usize,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        let normed = self.input_layernorm.forward(client, x)?;
        let attn_out = self
            .self_attn
            .forward_cached(client, &normed, rope, kv_cache, position)?;
        let h = var_add(x, &attn_out, client).map_err(Error::Numr)?;

        let normed = self.post_attention_layernorm.forward(client, &h)?;
        let mlp_out = self.mlp.forward(client, &normed)?;
        var_add(&h, &mlp_out, client).map_err(Error::Numr)
    }

    /// Delegate to [`MiniCpm4Attention::apply_lora`](super::attention::MiniCpm4Attention::apply_lora)
    /// and [`MiniCpm4Mlp::apply_lora`](super::mlp::MiniCpm4Mlp::apply_lora),
    /// summing their counts. `prefix` is the dotted path the owning
    /// [`super::model::MiniCpm4Model`] would pass to `extend_named` for this
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
}

/// Names ARE the field names (`input_layernorm`, `self_attn.*`,
/// `post_attention_layernorm`, `mlp.*`) — this matches the
/// `{prefix}.layers.{i}.*` checkpoint layout
/// ([`crate::model::audio::voxcpm::minicpm4::loader`]) exactly, so the
/// owning [`MiniCpm4Model`](super::model::MiniCpm4Model) need only prefix
/// by `layers.{i}` to reach the full checkpoint key.
impl<R: Runtime<DType = DType>> Module<R> for MiniCpm4Layer<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        let mut params = child_params(&self.input_layernorm);
        params.extend(child_params(&self.self_attn));
        params.extend(child_params(&self.post_attention_layernorm));
        params.extend(child_params(&self.mlp));
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
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
