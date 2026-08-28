//! Non-causal (bidirectional) GQA attention for VoxCPM2's shared MiniCPM4
//! block stack, used by both `feat_encoder` (`local_encoder`) and the local
//! DiT (`feat_decoder`).
//!
//! Every other transformer block in this crate runs through
//! `crate::model::attention_core`, which hardcodes causal(+window) masking —
//! this is the one bidirectional transformer stack in VoxCPM2, attending its
//! sequence with no mask at all (`feat_encoder`'s fixed 5-position
//! `[CLS, p0, p1, p2, p3]`; `feat_decoder`'s own assembled sequence). This
//! differs from `minicpm4`'s causal blocks, which mask via
//! `attention_core` and cannot be reused here. That orchestration is written
//! by hand here; every primitive it calls (`Linear`, `RoPE`/`apply_rope`,
//! `multi_head_attention_impl`, `repeat_kv`, `var_contiguous`) is reused
//! as-is. `LlamaAttention` itself is `pub(super)` to `model::llama::model`
//! and not reachable from here, and its `forward` methods are
//! unconditionally causal regardless.
//!
//! `head_dim` (128) is independent of `hidden_size / num_heads` (1024/16 =
//! 64) here — read from config (`kv_channels`), never derived, and passed
//! straight to the projections and the RoPE cache. `num_kv_heads` (GQA, 2
//! heads) is inherited from `lm_config`, not derivable from the per-caller
//! encoder/decoder config alone.

use crate::error::{Error, Result};
use crate::model::traits::ModelClient;
use crate::nn::{
    LoraTargets, MaybeLoraLinear, Module, RoPE, adapt_if_targeted, child_params, extend_named,
    load_lora_child, push_projection_name, repeat_kv, var_contiguous,
};
use crate::ops::impl_generic::attention::multi_head_attention_impl;
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_permute, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};

/// `q_proj`: 1024 -> 2048 (16 heads), `k_proj`/`v_proj`: 1024 -> 256 (2
/// heads, GQA group size 8), `o_proj`: 2048 -> 1024. All bias-free.
///
/// The projections are [`MaybeLoraLinear`], not plain `Linear`, for the
/// same reason `MiniCpm4Attention`'s are: a GGUF checkpoint stores them
/// block-quantized, and the quantized variant multiplies through
/// `quant_matmul` with the weight left PACKED instead of expanded to dense
/// F32 at load. This block stack is shared, so one conversion here covers
/// BOTH `feat_encoder` and `local_dit`. A safetensors checkpoint yields the
/// `Standard` variant and runs exactly the dense path it always did.
/// `MaybeLoraLinear` additionally lets any of the four carry a LoRA adapter.
pub struct BidirectionalAttention<R: Runtime> {
    pub(crate) q_proj: MaybeLoraLinear<R>,
    pub(crate) k_proj: MaybeLoraLinear<R>,
    pub(crate) v_proj: MaybeLoraLinear<R>,
    pub(crate) o_proj: MaybeLoraLinear<R>,
    pub(crate) num_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
}

impl<R: Runtime<DType = DType>> BidirectionalAttention<R> {
    /// Bidirectional GQA attention over `x: [N, S, hidden]` (`N = B*T`,
    /// `S = num_positions` — fixed at 5 for `feat_encoder`; caller-defined
    /// for `feat_decoder`). No mask: every position is valid and attends
    /// every other, including itself. Softmax scale is
    /// `1/sqrt(head_dim)`, derived from `q`'s actual last dimension by
    /// `multi_head_attention_impl`.
    pub fn forward<C>(&self, client: &C, x: &Var<R>, rope: &RoPE<R>) -> Result<Var<R>>
    where
        // `TypeConversionOps` is what `MaybeLoraLinear::forward` adds over a
        // dense `Linear::forward`: its decomposed-quant arm casts activations
        // to F32. `ModelClient` already carries `QuantMatmulOps`.
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
        let shape = x.shape().to_vec();
        let (batch, seq_len) = (shape[0], shape[1]);

        let q = self.q_proj.forward(client, x)?;
        let k = self.k_proj.forward(client, x)?;
        let v = self.v_proj.forward(client, x)?;

        // [N, S, H*D] -> [N, S, H, D] -> [N, H, S, D]
        let q = var_reshape(&q, &[batch, seq_len, self.num_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let k = var_reshape(&k, &[batch, seq_len, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let v = var_reshape(&v, &[batch, seq_len, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;

        let q = var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let k = var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let v = var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?;

        // Contiguous Q/K: the fused RoPE kernel assumes contiguous layout.
        // V: `repeat_kv` requires contiguous input too.
        let q = var_contiguous(&q)?;
        let k = var_contiguous(&k)?;
        let v = var_contiguous(&v)?;

        let q = client.apply_rope(&q, rope.cos_cache(), rope.sin_cache())?;
        let k = client.apply_rope(&k, rope.cos_cache(), rope.sin_cache())?;

        // GQA: repeat the 2 KV heads to 16 before the dense attention kernel.
        let repeat = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, repeat).map_err(Error::Numr)?;
        let v = repeat_kv(&v, repeat).map_err(Error::Numr)?;

        // No mask: bidirectional, all 5 positions always valid.
        let attn_out = multi_head_attention_impl(client, &q, &k, &v, None, self.num_heads)?;

        // [N, H, S, D] -> [N, S, H, D] -> [N, S, H*D]
        let attn_out = var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out)?;
        let attn_out = var_reshape(&attn_out, &[batch, seq_len, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        self.o_proj.forward(client, &attn_out)
    }

    /// Wrap `q_proj`/`k_proj`/`v_proj`/`o_proj` that `targets` names with a
    /// fresh LoRA adapter each, returning how many were adapted. `prefix` is
    /// the dotted path the owning [`BidirectionalLayer`](super::layer::BidirectionalLayer)
    /// would pass to `extend_named` for this block — `"self_attn"` — so each
    /// projection's path (via [`LoraTargets::join`]) matches
    /// `named_parameters()`'s path exactly. A leaf step: no zero-match check
    /// here — see
    /// [`MiniCpm4Attention::apply_lora`](crate::model::audio::voxcpm::minicpm4::MiniCpm4Attention::apply_lora)'s
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
            &mut self.q_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "q_proj",
        )?;
        adapted += adapt_if_targeted(
            &mut self.k_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "k_proj",
        )?;
        adapted += adapt_if_targeted(
            &mut self.v_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "v_proj",
        )?;
        adapted += adapt_if_targeted(
            &mut self.o_proj,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "o_proj",
        )?;
        Ok(adapted)
    }

    /// Every dotted projection path [`Self::apply_lora`] would adapt under
    /// `prefix` — `q_proj`, `k_proj`, `v_proj`, `o_proj` — INDEPENDENT of
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
        push_projection_name(&mut names, prefix, "q_proj");
        push_projection_name(&mut names, prefix, "k_proj");
        push_projection_name(&mut names, prefix, "v_proj");
        push_projection_name(&mut names, prefix, "o_proj");
        names
    }

    /// Write back updated `q_proj`/`k_proj`/`v_proj`/`o_proj` adapter values
    /// from an optimizer's `params` map, keeping their [`TensorId`]s. See
    /// [`crate::nn::MaybeLoraLinear::load_lora_parameters`] for the
    /// per-projection semantics. No prefix needed — unlike
    /// [`Self::apply_lora`], lookup is by ID.
    pub fn load_lora_parameters(
        &mut self,
        params: &std::collections::HashMap<TensorId, Tensor<R>>,
    ) -> Result<usize> {
        let mut written = load_lora_child(&mut self.q_proj, params, "q_proj")?;
        written += load_lora_child(&mut self.k_proj, params, "k_proj")?;
        written += load_lora_child(&mut self.v_proj, params, "v_proj")?;
        written += load_lora_child(&mut self.o_proj, params, "o_proj")?;
        Ok(written)
    }
}

/// Names ARE the field names (`q_proj`, `k_proj`, `v_proj`, `o_proj`) —
/// the `self_attn` checkpoint segment is added by the owning
/// [`BidirectionalLayer`](super::layer::BidirectionalLayer). Every
/// projection may enumerate empty when block-quantized — see
/// [`MaybeLoraLinear::parameters`](crate::nn::maybe_lora::MaybeLoraLinear).
impl<R: Runtime<DType = DType>> Module<R> for BidirectionalAttention<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        let mut params = child_params(&self.q_proj);
        params.extend(child_params(&self.k_proj));
        params.extend(child_params(&self.v_proj));
        params.extend(child_params(&self.o_proj));
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        extend_named(&mut params, "q_proj", self.q_proj.named_parameters());
        extend_named(&mut params, "k_proj", self.k_proj.named_parameters());
        extend_named(&mut params, "v_proj", self.v_proj.named_parameters());
        extend_named(&mut params, "o_proj", self.o_proj.named_parameters());
        params
    }
}
