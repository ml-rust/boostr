//! Causal GQA attention for VoxCPM2's MiniCPM4 decoder.
//!
//! Unlike the `feat_encoder` sibling in this module — the one bidirectional
//! transformer in the VoxCPM2 stack, which hand-writes its own unmasked
//! orchestration — this block is a plain causal decoder, so its full-sequence
//! [`MiniCpm4Attention::forward`] runs the shared [`attention_core_masked`]
//! sequence that every other causal block in the crate uses (`LlamaAttention`
//! included). That helper owns reshape/permute, contiguity, RoPE, the GQA head
//! repeat, and the causal mask; nothing here re-derives any of it.
//!
//! Causality is not a flag there: `attention_core_masked` always builds a
//! causal mask. That is the full-sequence forward, so without it every
//! position would attend to FUTURE positions while every shape stayed valid.
//!
//! The KV-cached [`MiniCpm4Attention::forward_cached`] instead calls the flash
//! kernel directly, as `LlamaAttention::forward_with_kv_cache` does.
//!
//! Both entry points take the RoPE tables as `Option`: VoxCPM2's
//! `residual_lm` is this same block with `no_rope` set, and its loader builds
//! no table at all. `no_rope` is NoPE — the rotation is dropped and NOTHING
//! replaces it (no ALiBi, no learned positions), so position reaches the block
//! only through causal ordering. A `None` table with `no_rope` unset is an
//! error, never a silent skip.
//!
//! `head_dim` (128) is read from config, never derived from
//! `hidden_size / num_heads` — see
//! [`MiniCpm4Config::head_dim`](crate::model::audio::voxcpm::minicpm4::MiniCpm4Config::head_dim).

use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::model::attention_core::{AttentionCoreSpec, AttentionKernel, attention_core_masked};
use crate::model::traits::ModelClient;
use crate::nn::var_ops::var_contiguous;
use crate::nn::{
    LoraTargets, MaybeLoraLinear, MaybeQuantLinear, Module, RoPE, adapt_if_targeted, child_params,
    extend_named, load_lora_child, push_projection_name,
};
use crate::quant::traits::DequantOps;
use guards::{missing_rope, require_preallocated_cache};
use numr::autograd::{Var, var_narrow, var_permute, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};

/// `q_proj`: 2048 -> 2048 (16 heads x 128), `k_proj`/`v_proj`: 2048 -> 256
/// (2 heads x 128, GQA group size 8), `o_proj`: 2048 -> 2048. All bias-free.
///
/// The projections are [`MaybeLoraLinear`], not plain `Linear`: a GGUF
/// checkpoint stores them block-quantized, and this is the bulk of VoxCPM2's
/// weights. The quantized variant multiplies through `quant_matmul` with the
/// weight left PACKED, so a Q4_K file costs Q4_K-sized memory instead of
/// being expanded to dense F32 at load. A safetensors checkpoint yields the
/// `Standard` variant and runs exactly the dense path it always did.
/// `MaybeLoraLinear` additionally lets any of the four carry a LoRA adapter.
pub struct MiniCpm4Attention<R: Runtime> {
    pub(crate) q_proj: MaybeLoraLinear<R>,
    pub(crate) k_proj: MaybeLoraLinear<R>,
    pub(crate) v_proj: MaybeLoraLinear<R>,
    pub(crate) o_proj: MaybeLoraLinear<R>,
    pub(crate) num_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    /// NoPE: skip the rotary embedding on BOTH paths (`residual_lm`).
    ///
    /// `false` for `base_lm`, where every code path runs exactly as it did
    /// before this flag existed.
    pub(crate) no_rope: bool,
}

impl<R: Runtime<DType = DType>> MiniCpm4Attention<R> {
    /// Borrowed view of this block's attention parameters, for
    /// [`attention_core_masked`].
    ///
    /// No Q/K per-head norm and no ALiBi on this checkpoint, and
    /// `sliding_window: 0` — MiniCPM4's VoxCPM2 configuration attends the
    /// full prefix, so windowing is disabled rather than left unset.
    ///
    /// `skip_rope` carries `no_rope` through. It is INDEPENDENT of
    /// `use_alibi`, which stays `false` here: ALiBi would add a distance bias
    /// that `residual_lm` does not have.
    fn core_spec(&self) -> AttentionCoreSpec<'_, R> {
        AttentionCoreSpec {
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            q_norm: None,
            k_norm: None,
            use_alibi: false,
            skip_rope: self.no_rope,
            sliding_window: 0,
            // Only `forward` reads this field. `attention_core_flash` builds a
            // backward node that resolves its client from the runtime, so it
            // requires `R::Client: FlashAttentionOps<R>` — a bound this block's
            // where-clause does not carry. `forward_cached` is unaffected: it
            // calls `flash_attention_fwd` on `client` (already a
            // `FlashAttentionOps<R>` via `ModelClient<R>`), with no backward.
            kernel: AttentionKernel::Masked,
        }
    }

    /// Dtype and device the K/V this block writes will actually have.
    ///
    /// Read off `k_proj` because it is that projection's OUTPUT that lands in
    /// the cache. A quantized projection cannot answer this from its weight:
    /// `quant_matmul` consumes F32 and emits F32 whatever block format the
    /// weight is packed in, so the cached keys are F32 there — the packed
    /// weight has no element dtype to copy.
    pub(crate) fn kv_dtype_device(&self) -> Result<(DType, &R::Device)> {
        match self.k_proj.base() {
            MaybeQuantLinear::Standard(linear) => {
                let w = linear.weight().tensor();
                Ok((w.dtype(), w.device()))
            }
            MaybeQuantLinear::Quantized(qlinear) => Ok((DType::F32, qlinear.weight().device())),
            MaybeQuantLinear::DecomposedQuant(_) => Err(Error::ModelError {
                reason: "MiniCPM4 k_proj: no VoxCPM2 checkpoint loads decomposed-quantized \
                         (AWQ/GPTQ) weights, so the KV cache dtype is undefined here"
                    .to_string(),
            }),
        }
    }

    /// Causal GQA attention over `x: [batch, seq, hidden]`, returning
    /// `[batch, seq, hidden]`.
    ///
    /// Softmax scale is `1/sqrt(head_dim)`, derived from `q`'s actual last
    /// dimension inside `multi_head_attention_impl`.
    ///
    /// `rope` may be `None` only when `no_rope` is set; otherwise it is an
    /// [`Error::InvalidArgument`], because running unrotated would stay
    /// shape-valid while computing a different model.
    pub fn forward<C>(&self, client: &C, x: &Var<R>, rope: Option<&RoPE<R>>) -> Result<Var<R>>
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
        let q = self.q_proj.forward(client, x)?;
        let k = self.k_proj.forward(client, x)?;
        let v = self.v_proj.forward(client, x)?;

        let (cos, sin) = match (self.no_rope, rope) {
            (true, _) => (None, None),
            (false, Some(rope)) => (Some(rope.cos_cache()), Some(rope.sin_cache())),
            (false, None) => return Err(missing_rope()),
        };

        let attn_out = attention_core_masked(client, &q, &k, &v, cos, sin, &self.core_spec())?;

        self.o_proj.forward(client, &attn_out)
    }

    /// KV-cached causal GQA attention over `x: [batch, seq, hidden]` covering
    /// absolute positions `position..position + seq`, returning
    /// `[batch, seq, hidden]`.
    ///
    /// Serves BOTH cached shapes: prefill passes the whole prefix at
    /// `position == 0`, a decode step passes `seq == 1` at the next position.
    ///
    /// This does not call [`attention_core_masked`]: that entry point rotates
    /// the RAW K it is given, so handing it the cache would re-apply RoPE to
    /// keys already rotated when they were written.
    /// The prologue is otherwise the same one that function documents —
    /// reshape/permute, contiguous Q/K, (no Q/K norm here), RoPE. The attention
    /// is the flash kernel, reading the cache in place and masking internally:
    /// nothing here materializes a mask, repeats KV heads, or copies history.
    ///
    /// # Masking
    ///
    /// `causal` is `seq > 1`. A prefill chunk needs it, because each query row
    /// must reject the rows after it. A decode step (`seq == 1`) does not: its
    /// one query row is the last position, so a causal mask would be all zeros.
    ///
    /// The admitted key set is the reference's. `build_attention_mask` in
    /// `ops::impl_generic::attention::flash_standard` offsets keys by
    /// `seq_len_k - seq_len_q` — the offset the shared `prefill_attention_mask`
    /// builder used — so query row `i` admits exactly keys `0..=position + i`.
    /// The reference instead masks with `arange(max_length) <= position_id`
    /// across the FULL preallocated width: same keys, plus `-inf` on zeroed
    /// never-written slots that contribute nothing to the softmax either way,
    /// and `kv_seq_len` stops the kernel reading those slots at all. The
    /// equivalence holds ONLY because the caller writes positions in order from
    /// 0, which [`MiniCpm4Model::decode_step`] enforces by requiring
    /// `position == kv_cache.seq_len()`.
    ///
    /// [`MiniCpm4Model::decode_step`]:
    ///     crate::model::audio::voxcpm::minicpm4::MiniCpm4Model::decode_step
    /// `rope` may be `None` only when `no_rope` is set; otherwise it is an
    /// [`Error::InvalidArgument`]. The decode path never dereferences an
    /// absent table.
    pub fn forward_cached<C>(
        &self,
        client: &C,
        x: &Var<R>,
        rope: Option<&RoPE<R>>,
        kv_cache: &mut KvCache<R>,
        position: usize,
    ) -> Result<Var<R>>
    where
        // `TypeConversionOps` for the same reason `forward` needs it.
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
        if shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!("expected 3D [batch, seq, hidden], got {}D", shape.len()),
            });
        }
        let (batch, seq) = (shape[0], shape[1]);

        let q = self.q_proj.forward(client, x)?;
        let k = self.k_proj.forward(client, x)?;
        let v = self.v_proj.forward(client, x)?;

        // [B, S, H*D] -> [B, S, H, D] -> [B, H, S, D]
        let q =
            var_reshape(&q, &[batch, seq, self.num_heads, self.head_dim]).map_err(Error::Numr)?;
        let k = var_reshape(&k, &[batch, seq, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let v = var_reshape(&v, &[batch, seq, self.num_kv_heads, self.head_dim])
            .map_err(Error::Numr)?;
        let q = var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let k = var_permute(&k, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let v = var_permute(&v, &[0, 2, 1, 3]).map_err(Error::Numr)?;

        // The fused RoPE kernel assumes contiguous layout.
        let q = var_contiguous(&q)?;
        let k = var_contiguous(&k)?;

        // Same precomputed cos/sin tables the full-sequence path uses, sliced
        // at the absolute positions this call covers. Building a second table
        // here is how the two paths would drift.
        //
        // NoPE skips the whole block: this is the SECOND site that applies
        // RoPE (the full-sequence path applies it inside
        // `attention_core_masked`), and honouring `no_rope` in only one of
        // them would leave the two paths computing different models.
        let (q, k) = match (self.no_rope, rope) {
            (true, _) => (q, k),
            (false, Some(rope)) => {
                // `position` rotates the query; `kv_cache` stores it at its own
                // `seq_len`. They MUST agree, or the query is rotated for one
                // absolute position and attended at another — shapes stay
                // valid and the output is silently wrong.
                //
                // Checked here, not at the top, because this is the ONLY site
                // that reads `position`. A NoPE stack never rotates, so its
                // `position` is genuinely unused and stays unconstrained —
                // which is what
                // `nope_output_is_independent_of_absolute_position` asserts by
                // passing a `position` a cache-aligned call could not use.
                let written = kv_cache.seq_len();
                if position != written {
                    return Err(Error::InferenceError {
                        reason: format!(
                            "RoPE position {position} disagrees with the KV cache's next \
                             write index {written}: the query would be rotated for position \
                             {position} but stored and attended at {written}. Pass \
                             position == kv_cache.seq_len(), as MiniCpm4Model::decode_step \
                             does."
                        ),
                    });
                }
                let cos = var_narrow(rope.cos_cache(), 0, position, seq).map_err(Error::Numr)?;
                let sin = var_narrow(rope.sin_cache(), 0, position, seq).map_err(Error::Numr)?;
                let q = client.apply_rope(&q, &cos, &sin)?;
                let k = client.apply_rope(&k, &cos, &sin)?;
                (q, k)
            }
            (false, None) => return Err(missing_rope()),
        };

        let v = var_contiguous(&v)?;

        // Post-RoPE K/V land in the cache, as the reference writes rotated keys
        // at index `position_id`. `update_fused` writes the new slots IN PLACE
        // through `KvCacheOps::kv_cache_update`; `update` would `slice_assign`,
        // which is functional — a second buffer of the whole preallocated
        // cache, allocated and copied per layer per step.
        require_preallocated_cache(kv_cache, seq)?;
        kv_cache.update_fused(k.tensor(), v.tensor(), client)?;

        // The kernel broadcasts the GQA heads itself, so the raw cache buffers
        // go in untouched, bounded to the written slots by `kv_seq_len`.
        let (out, _lse) = client.flash_attention_fwd(
            q.tensor(),
            kv_cache.k_cache_raw(),
            kv_cache.v_cache_raw(),
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            // Prefill chunk masks; a single decode row has nothing to reject.
            seq > 1,
            // The disabled-window sentinel `core_spec` declares.
            self.core_spec().sliding_window,
            Some(kv_cache.seq_len()),
        )?;
        let attn_out = Var::new(out, false);

        // [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        let attn_out = var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out)?;
        let attn_out = var_reshape(&attn_out, &[batch, seq, self.num_heads * self.head_dim])
            .map_err(Error::Numr)?;

        self.o_proj.forward(client, &attn_out)
    }

    /// Wrap `q_proj`/`k_proj`/`v_proj`/`o_proj` that `targets` names with a
    /// fresh LoRA adapter each, returning how many were adapted.
    ///
    /// `prefix` is the dotted path the OWNING [`MiniCpm4Layer`] would pass
    /// to [`crate::nn::extend_named`] for this block — `"self_attn"` under
    /// `MiniCpm4Layer::named_parameters`'s own convention — so each
    /// projection's full path here (via [`LoraTargets::join`]) matches
    /// `named_parameters()`'s path for that same projection exactly.
    ///
    /// This is a LEAF step in the bottom-up composition: it does NOT call
    /// [`LoraTargets::ensure_all_match`] itself, only the model-level entry
    /// points do (see their doc comments) — a target this block has no
    /// projection for (e.g. `gate_proj`) is not an error here, only if it
    /// matches nothing ANYWHERE in the tree the caller actually entered on.
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
    /// decomposed-quantized. This is what fixes the QLoRA validation bug: a
    /// GGUF-loaded `MiniCpm4Attention` has `named_parameters()` return
    /// EMPTY for every projection here (block-quantized storage has no
    /// `Var<R>`), so validating against `named_parameters()` rejects a
    /// perfectly valid `q_proj`/`v_proj` target on a quantized checkpoint.
    /// Which projections exist is a STRUCTURAL property of this type, not a
    /// function of whether its weights happen to be dense. Built with the
    /// same [`crate::nn::push_projection_name`] helper `apply_lora`'s
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

    /// Cheap duplicate that preserves every projection's `Var<R>`
    /// `TensorId`s, for capturing this block by owned value in a `'static`
    /// activation-checkpointing closure — `numr::autograd::checkpoint`'s
    /// closure is `Fn(...) + Send + Sync + 'static`, so a layer cannot be
    /// borrowed into it. Each projection routes through
    /// [`MaybeLoraLinear::alias`], never [`Clone`], so the optimizer, keyed
    /// by `TensorId`, still sees the original parameters' gradients.
    pub fn alias(&self) -> Self {
        Self {
            q_proj: self.q_proj.alias(),
            k_proj: self.k_proj.alias(),
            v_proj: self.v_proj.alias(),
            o_proj: self.o_proj.alias(),
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            no_rope: self.no_rope,
        }
    }
}

/// Names ARE the field names (`q_proj`, `k_proj`, `v_proj`, `o_proj`) —
/// the `self_attn` checkpoint segment is added by the owning
/// [`MiniCpm4Layer`](super::layer::MiniCpm4Layer). `no_rope` carries no
/// `Var<R>` (it is a `bool`), so it is correctly absent from every
/// collection below.
impl<R: Runtime<DType = DType>> Module<R> for MiniCpm4Attention<R> {
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

mod guards;

#[cfg(test)]
mod tests;
