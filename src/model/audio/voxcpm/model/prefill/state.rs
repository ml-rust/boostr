//! The prefill's input row and its output state.

use crate::error::{Error, Result};
use crate::inference::LayeredKvCache;
use crate::model::audio::voxcpm::minicpm4::LanePositions;
use numr::autograd::Var;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// One request of a prefill batch: what the single-row `prefill` takes.
pub struct PrefillRow<'a, R: Runtime> {
    /// [`encode_reference`](super::super::VoxCpm2Model::encode_reference)'s
    /// output, `[T_ref, patch_size, feat_dim]`. `None` selects no-reference
    /// (zero-shot) mode: the whole reference prefix, delimiters and
    /// zero-patch bookends included, is dropped and `S = text_token_ids.len()`.
    ///
    /// No-reference is a type-level state, never an empty tensor. A `Some`
    /// holding `T_ref == 0` is rejected.
    pub ref_feat: Option<&'a Tensor<R>>,
    /// The already tokenized prompt; must end with
    /// [`AUDIO_START_ID`](super::super::config::AUDIO_START_ID). boostr does
    /// not tokenize here. A continuation row carries `prompt_text ++
    /// target_text` here.
    pub text_token_ids: &'a [u32],
    /// [`encode_prompt`](super::super::VoxCpm2Model::encode_prompt)'s
    /// output, `[T_prompt, patch_size, feat_dim]`: the continuation ("Hi-Fi")
    /// tail, appended after the text as audio positions. `None` selects no
    /// tail. With `ref_feat` also `Some` the row is ref+continuation, with
    /// `ref_feat` `None` it is continuation-only.
    ///
    /// No tail is a type-level state, never an empty tensor. A `Some`
    /// holding `T_prompt == 0` is rejected.
    pub prompt_feat: Option<&'a Tensor<R>>,
}

// Manual, not derived: a derive would demand `R: Clone`/`R: Copy`, and the
// row holds only borrows.
impl<R: Runtime> Clone for PrefillRow<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Runtime> Copy for PrefillRow<'_, R> {}

impl<R: Runtime> PrefillRow<'_, R> {
    /// Reference patch count, after checking `ref_feat`'s shape against the
    /// model's patch geometry. `0` for no-reference mode.
    pub(super) fn t_ref(&self, patch_size: usize, feat_dim: usize) -> Result<usize> {
        patch_count(self.ref_feat, "ref_feat", "T_ref", patch_size, feat_dim)
    }

    /// Continuation prompt patch count, after checking `prompt_feat`'s shape
    /// against the model's patch geometry. `0` for no continuation tail.
    pub(super) fn t_prompt(&self, patch_size: usize, feat_dim: usize) -> Result<usize> {
        patch_count(
            self.prompt_feat,
            "prompt_feat",
            "T_prompt",
            patch_size,
            feat_dim,
        )
    }
}

/// Leading-axis length of an optional `[T, patch_size, feat_dim]` feature
/// tensor, `0` for `None`. `arg` and `axis` name the tensor and its length in
/// the error.
fn patch_count<R: Runtime>(
    feat: Option<&Tensor<R>>,
    arg: &'static str,
    axis: &str,
    patch_size: usize,
    feat_dim: usize,
) -> Result<usize> {
    let Some(feat) = feat else {
        return Ok(0);
    };
    let shape = feat.shape();
    if shape.len() != 3 || shape[1] != patch_size || shape[2] != feat_dim {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!(
                "expected [{axis}, {patch_size}, {feat_dim}], got {shape:?}; \
                 pass the model's own encode output"
            ),
        });
    }
    // An absent span is `None`, never a zero-length tensor: a `Some` here
    // must carry at least one patch.
    if shape[0] == 0 {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!(
                "expected at least 1 patch, got {axis} == 0; pass None to omit the span"
            ),
        });
    }
    Ok(shape[0])
}

/// Everything the per-patch sampling loop needs from the prefill.
pub struct PrefillState<R: Runtime> {
    /// `[B, lm_hidden]` — the LAST row of the post-blend `enc_outputs`.
    /// Un-fsq'd for a row whose last position is text, fsq'd for a row with
    /// a continuation tail, whose last position is audio. Under left padding
    /// row `S_max - 1` is the last real position of every batch row.
    pub lm_hidden: Var<R>,
    /// `[B, lm_hidden]` — the LAST row of `residual_lm`'s output.
    pub residual_hidden: Var<R>,
    /// `base_lm`'s cache, primed to `current_length == S_max`.
    pub base_cache: LayeredKvCache<R>,
    /// `residual_lm`'s cache, primed to the SAME `current_length`.
    pub residual_cache: LayeredKvCache<R>,
    /// Row `b`'s next write position, its key range, and the batch's left
    /// padding (`lanes.pad().starts[b] = S_max - S_b`, on the device for the
    /// attention kernels and on the host for the per-row RoPE positions).
    /// Every row starts at `S_max`. ONE table serves both caches: the decode
    /// loop writes both at the same position and advances the table once,
    /// after both have stepped.
    pub lanes: LanePositions<R>,
    /// Rows ALLOCATED, `B >= 1`: what the tensors and both caches here
    /// hold. A prefill's own rows never move. A slotted generation state
    /// drives this handoff forward in place and widens it as it seats more
    /// slots, so it keeps this in step with the rows it allocated.
    pub batch: usize,
    /// Present only when the prefill ran via a `_capturing` entry point;
    /// always `None` on the plain paths, which allocate nothing extra.
    pub intermediates: Option<PrefillIntermediates<R>>,
    /// `[B, patch_size, feat_dim]` in the LM dtype: each row's last raw
    /// `audio_feat` patch, the DiT condition of its first generated patch.
    /// Zeros for a row without a continuation tail, the prompt's last patch
    /// for a row with one.
    ///
    /// `Some` on every prefill output. [`GenerateState::start`] and
    /// [`GenerateState::admit`] read it; `start` takes it, leaving `None` in
    /// the running state. `None` conditions every row on zeros.
    ///
    /// [`GenerateState::start`]: crate::model::audio::voxcpm::model::GenerateState::start
    /// [`GenerateState::admit`]: crate::model::audio::voxcpm::model::GenerateState::admit
    pub prefix_feat_cond: Option<Var<R>>,
    /// Per row, the decode context of a continuation row: the last
    /// `min(STREAMING_PREFIX_LEN - 1, audio positions)` audio patches of its
    /// sequence, each `[1, patch_size, feat_dim]` in the LM dtype, in
    /// sequence order. Empty for a row without a continuation tail. The
    /// decode prepends them to the row's generated patches and trims their
    /// samples off the front.
    ///
    /// One entry per prefilled row. `start` takes the vector, leaving it
    /// empty in the running state; an empty vector means no row has context.
    pub context: Vec<Vec<Var<R>>>,
}

/// Full-sequence intermediates, for the gate example to compare against the
/// reference. Each is `[B, S_max, _]`.
pub struct PrefillIntermediates<R: Runtime> {
    /// `text_mask * text_embed + audio_mask * feat_embed`, `[B, S_max, 2048]`.
    pub combined_embed: Var<R>,
    /// `base_lm`'s output AFTER the fsq blend, `[B, S_max, 2048]`.
    pub enc_outputs: Var<R>,
    /// `enc_to_lm_proj(feat_encoder(audio_feat))`, UNMASKED, `[B, S_max, 2048]`.
    pub feat_embed: Var<R>,
    /// `fusion_concat_proj(...)`, i.e. what `residual_lm` was prefilled with,
    /// `[B, S_max, 2048]`.
    pub residual_enc_inputs: Var<R>,
}
