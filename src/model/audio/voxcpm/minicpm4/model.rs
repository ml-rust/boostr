//! `MiniCpm4Model` — VoxCPM2's MiniCPM4 decoder-only transformer (`base_lm`),
//! full-sequence causal forward, inference only.
//!
//! ```text
//! inputs_embeds [B, S, hidden]
//!   -> 28x pre-norm decoder layer (causal GQA 16/2, head_dim 128, SwiGLU)
//!   -> final RmsNorm                                   [B, S, hidden]
//! ```
//!
//! Two things this model deliberately does NOT do:
//!
//! - **No `lm_head`.** The checkpoint has none. [`MiniCpm4Model::forward`]
//!   returns hidden states, never logits.
//! - **No KV cache on this path.** [`forward`](MiniCpm4Model::forward)
//!   recomputes every position on every call. The incremental (KV-cached)
//!   decode path — `new_kv_cache` / `prefill` / `decode_step` — lives in the
//!   sibling [`decode`](crate::model::audio::voxcpm::minicpm4::decode) module
//!   and leaves this one untouched.
//!
//! - **RoPE is OPTIONAL.** `residual_lm` runs NoPE (`no_rope`), so `rope` is
//!   `None` there and every attention block skips the rotation on both the
//!   full-sequence and the KV-cached path. Nothing takes its place.
//!
//! [`forward`](MiniCpm4Model::forward) takes pre-computed `inputs_embeds`
//! rather than token ids, matching the real pipeline (which feeds a combined
//! text+audio embedding). The `embed_tokens` table is exposed separately as
//! [`MiniCpm4Model::embed`] and is OPTIONAL: VoxCPM2's `residual_lm` is this
//! same architecture with `vocab_size` 0 and no table at all.
//!
//! Built from plain [`Var<R>`]-wrapped weights (`requires_grad = false`)
//! rather than autograd-tracked training params — same inference-only posture
//! as the `local_encoder` and AudioVAE siblings.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::minicpm4::layer::MiniCpm4Layer;
use crate::model::traits::ModelClient;
use crate::nn::{
    LoraTargets, MaybeQuantEmbedding, Module, RmsNorm, RoPE, child_params, extend_named,
};
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};

pub struct MiniCpm4Model<R: Runtime> {
    /// `[vocab_size, hidden_size]` lookup table. `None` when the config
    /// carries `vocab_size == 0` (`residual_lm`), which has no such tensor in
    /// the checkpoint. May be block-quantized (packed on device, dequantized
    /// only for the rows a forward pass gathers) or standard, depending on
    /// what the checkpoint stores — see [`MaybeQuantEmbedding`].
    pub(crate) embed_tokens: Option<MaybeQuantEmbedding<R>>,
    pub(crate) layers: Vec<MiniCpm4Layer<R>>,
    pub(crate) norm: RmsNorm<R>,
    /// Precomputed cos/sin tables. `None` when the config carries
    /// `no_rope` (`residual_lm`): that stack runs NoPE, so the loader builds no
    /// table rather than a `max_position_embeddings`-row cache nothing reads.
    pub(crate) rope: Option<RoPE<R>>,
    pub(crate) hidden_size: usize,
    /// Run every layer through
    /// [`MiniCpm4Layer::forward_checkpointed`] instead of
    /// [`MiniCpm4Layer::forward`]. `false` by default, so inference pays
    /// nothing — see [`Self::set_activation_checkpointing`].
    pub(crate) activation_checkpointing: bool,
}

impl<R: Runtime<DType = DType>> MiniCpm4Model<R> {
    /// Number of decoder layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Model width.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Whether this instantiation owns an `embed_tokens` table.
    pub fn has_embedding(&self) -> bool {
        self.embed_tokens.is_some()
    }

    /// Turn activation checkpointing on or off for every layer in this stack.
    ///
    /// `on` trades ~33% extra compute for dropping each layer's
    /// intermediates during the forward pass and recomputing them during
    /// backward, which is what caps training VRAM. Default is `off`, so an
    /// inference path pays nothing.
    ///
    /// Set it only for training: the KV-cached decode path
    /// ([`decode`](crate::model::audio::voxcpm::minicpm4::decode)) never
    /// reads this flag, since a recomputed segment must not touch the cache.
    pub fn set_activation_checkpointing(&mut self, on: bool) {
        self.activation_checkpointing = on;
    }

    /// Whether this stack runs its layers with activation checkpointing.
    pub fn activation_checkpointing(&self) -> bool {
        self.activation_checkpointing
    }

    /// Whether this instantiation rotates Q/K.
    ///
    /// `false` for a NoPE (`no_rope`) stack, which owns no RoPE table. Every
    /// attention block in such a stack carries the matching `no_rope` flag, so
    /// no path here can reach a rotation with nothing to rotate by.
    pub fn uses_rope(&self) -> bool {
        self.rope.is_some()
    }

    /// `embed_tokens` lookup: integer `ids` `[...]` -> `[..., hidden_size]`.
    ///
    /// The result is NOT scaled: `scale_emb` (12.0) applies only under muP,
    /// which is off on this checkpoint, so the reference leaves the lookup
    /// untouched and so does this.
    ///
    /// Errors when the config had `vocab_size == 0` — a `residual_lm`-shaped
    /// instantiation is fed pre-computed embeddings and has no table to read.
    pub fn embed<C>(&self, client: &C, ids: &Tensor<R>) -> Result<Var<R>>
    where
        C: ModelClient<R> + DequantOps<R>,
        R::Client: IndexingOps<R>,
    {
        let table = self
            .embed_tokens
            .as_ref()
            .ok_or_else(|| Error::InvalidArgument {
                arg: "ids",
                reason: "this MiniCPM4 instantiation has no embed_tokens table \
                         (config vocab_size == 0); pass inputs_embeds to forward instead"
                    .to_string(),
            })?;
        table.forward(client, ids)
    }

    /// Full-sequence causal forward over pre-computed embeddings.
    ///
    /// `inputs_embeds: [batch, seq, hidden_size]` -> `[batch, seq,
    /// hidden_size]` after the final `norm`. `seq` may not exceed the RoPE
    /// cache length the loader built (`max_position_embeddings`); a NoPE
    /// (`no_rope`) stack has no such cache and no such bound.
    ///
    /// When [`set_activation_checkpointing`](Self::set_activation_checkpointing)
    /// is on, every layer runs through
    /// [`MiniCpm4Layer::forward_checkpointed`] — same ops, same order, same
    /// output values, at ~33% extra compute. Off (the default), the walk is
    /// exactly what it was before that flag existed.
    pub fn forward<C>(&self, client: &C, inputs_embeds: &Var<R>) -> Result<Var<R>>
    where
        // `'static` is what `forward_checkpointed` adds: the closure numr
        // stores for the backward recompute owns the client.
        C: ModelClient<R> + TypeConversionOps<R> + 'static,
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
        let shape = inputs_embeds.shape().to_vec();
        if shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg: "inputs_embeds",
                reason: format!(
                    "expected 3D [batch, seq, hidden_size], got {}D",
                    shape.len()
                ),
            });
        }
        if shape[2] != self.hidden_size {
            return Err(Error::InvalidArgument {
                arg: "inputs_embeds",
                reason: format!(
                    "expected hidden_size {}, got {}",
                    self.hidden_size, shape[2]
                ),
            });
        }

        // `alias`, never `clone`: `Var::clone` mints a fresh `TensorId`,
        // which orphans a caller's gradient when `inputs_embeds` is a leaf.
        let mut h = inputs_embeds.alias();
        for layer in &self.layers {
            h = if self.activation_checkpointing {
                layer.forward_checkpointed(client, &h, self.rope.as_ref())?
            } else {
                layer.forward(client, &h, self.rope.as_ref())?
            };
        }
        self.norm.forward(client, &h)
    }

    /// Wrap every layer's targeted projections with a fresh LoRA adapter,
    /// returning the total adapted. `prefix` mirrors
    /// `Module::named_parameters` below exactly: each layer is joined at
    /// `"layers.{i}"`. `embed_tokens`/`norm` carry no [`crate::nn::MaybeLoraLinear`]
    /// projections, so neither is touched here.
    ///
    /// This is the entry point for adapting this sub-model DIRECTLY (VoxCPM2
    /// instantiates this type twice — `base_lm`/`residual_lm` — and a caller
    /// may adapt either on its own), so it validates every target up front
    /// with [`LoraTargets::ensure_all_match`] against this tree's OWN full
    /// candidate set — [`Self::lora_projection_names`], NOT
    /// `self.named_parameters()` — before delegating to
    /// [`Self::apply_lora_unchecked`].
    ///
    /// The candidate set MUST be structural (every projection this model
    /// COULD adapt), not parameter-derived (every projection that HAPPENS
    /// to carry a dense `Var<R>` right now): on a GGUF checkpoint every
    /// `MiniCpm4Attention`/`MiniCpm4Mlp` projection is block-quantized, so
    /// `named_parameters()` returns EMPTY for all of them and a valid
    /// `q_proj`/`v_proj` target would be rejected as matching nothing —
    /// exactly the QLoRA-unusable bug this candidate source fixes.
    pub fn apply_lora(
        &mut self,
        targets: &LoraTargets,
        rank: usize,
        alpha: f32,
        device: &R::Device,
        prefix: &str,
    ) -> Result<usize> {
        let candidates = self.lora_projection_names(prefix);
        targets.ensure_all_match(&candidates)?;
        self.apply_lora_unchecked(targets, rank, alpha, device, prefix)
    }

    /// Every dotted projection path [`Self::apply_lora`] would adapt under
    /// `prefix` — INDEPENDENT of whether each layer's projections are
    /// dense, block-quantized, or decomposed-quantized. `embed_tokens`/
    /// `norm` carry no [`crate::nn::MaybeLoraLinear`] projections, so
    /// neither contributes a name, matching [`Self::apply_lora_unchecked`]'s
    /// walk exactly: each layer is joined at the SAME `"layers.{i}"` prefix
    /// [`Self::apply_lora_unchecked`] passes to
    /// [`crate::model::audio::voxcpm::minicpm4::layer::MiniCpm4Layer::apply_lora`],
    /// so a path here is never built by separately hand-written logic.
    pub fn lora_projection_names(&self, prefix: &str) -> Vec<String> {
        let mut names = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            names.extend(
                layer.lora_projection_names(&LoraTargets::join(prefix, &format!("layers.{i}"))),
            );
        }
        names
    }

    /// Same walk as [`Self::apply_lora`] but skips
    /// [`LoraTargets::ensure_all_match`]. Exists for a parent
    /// (`VoxCpm2Model`) that has already validated `targets` against the
    /// WHOLE model: re-validating here against only this subtree would
    /// reject a target that lives in a sibling (`feat_encoder`,
    /// `feat_decoder`, `aux`), even though it is perfectly valid at root.
    pub(crate) fn apply_lora_unchecked(
        &mut self,
        targets: &LoraTargets,
        rank: usize,
        alpha: f32,
        device: &R::Device,
        prefix: &str,
    ) -> Result<usize> {
        let mut adapted = 0;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            adapted += layer.apply_lora(
                targets,
                rank,
                alpha,
                device,
                &LoraTargets::join(prefix, &format!("layers.{i}")),
            )?;
        }
        Ok(adapted)
    }

    /// Write back updated adapter values across every layer from an
    /// optimizer's `params` map, keeping every adapter's [`TensorId`]s. See
    /// [`crate::nn::MaybeLoraLinear::load_lora_parameters`] for the
    /// per-projection semantics. No prefix or target validation needed here
    /// — unlike [`Self::apply_lora`], lookup is by ID, not by dotted path,
    /// so there is no zero-match trap to guard against.
    pub fn load_lora_parameters(
        &mut self,
        params: &std::collections::HashMap<TensorId, Tensor<R>>,
    ) -> Result<usize> {
        let mut written = 0;
        for layer in self.layers.iter_mut() {
            written += layer.load_lora_parameters(params)?;
        }
        Ok(written)
    }
}

/// Names ARE the field names (`embed_tokens`, `layers.{i}.*`, `norm`) —
/// this matches the `{prefix}.*` checkpoint layout
/// ([`crate::model::audio::voxcpm::minicpm4::loader`]) exactly, so
/// [`VoxCpm2Model`](crate::model::audio::voxcpm::model::VoxCpm2Model) need
/// only prefix by `base_lm`/`residual_lm` to reach the full checkpoint key.
/// `embed_tokens` is absent entirely on a `residual_lm` instantiation
/// (`vocab_size == 0`) — correctly contributing nothing rather than a
/// zero-filled placeholder. `rope` carries no `Var<R>` (a precomputed,
/// non-learned cos/sin cache, like every other `RoPE` table in this crate)
/// and is correctly absent below.
impl<R: Runtime<DType = DType>> Module<R> for MiniCpm4Model<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        let mut params = self
            .embed_tokens
            .as_ref()
            .map(|embed_tokens| child_params(embed_tokens))
            .unwrap_or_default();
        for layer in &self.layers {
            params.extend(child_params(layer));
        }
        params.extend(child_params(&self.norm));
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        if let Some(embed_tokens) = &self.embed_tokens {
            extend_named(&mut params, "embed_tokens", embed_tokens.named_parameters());
        }
        for (i, layer) in self.layers.iter().enumerate() {
            extend_named(
                &mut params,
                &format!("layers.{i}"),
                layer.named_parameters(),
            );
        }
        extend_named(&mut params, "norm", self.norm.named_parameters());
        params
    }
}

#[cfg(test)]
pub(crate) mod tests;
