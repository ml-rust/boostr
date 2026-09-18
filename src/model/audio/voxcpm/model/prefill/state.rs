//! The prefill's input row and its output state.

use crate::error::{Error, Result};
use crate::inference::LayeredKvCache;
use crate::model::audio::voxcpm::minicpm4::LeftPad;
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
    /// not tokenize here.
    pub text_token_ids: &'a [u32],
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
        let Some(ref_feat) = self.ref_feat else {
            return Ok(0);
        };
        let ref_shape = ref_feat.shape().to_vec();
        if ref_shape.len() != 3 || ref_shape[1] != patch_size || ref_shape[2] != feat_dim {
            return Err(Error::InvalidArgument {
                arg: "ref_feat",
                reason: format!("expected [T_ref, {patch_size}, {feat_dim}], got {ref_shape:?}"),
            });
        }
        // No-reference is `None`, never a zero-length tensor: a `Some` here
        // must carry at least one patch.
        if ref_shape[0] == 0 {
            return Err(Error::InvalidArgument {
                arg: "ref_feat",
                reason: "expected at least 1 reference patch, got 0; pass None for \
                         no-reference (zero-shot) mode"
                    .to_string(),
            });
        }
        Ok(ref_shape[0])
    }
}

/// Everything the per-patch sampling loop needs from the prefill.
pub struct PrefillState<R: Runtime> {
    /// `[B, lm_hidden]` — the LAST row of the post-blend `enc_outputs`,
    /// un-fsq'd because it is a text position. Under left padding row
    /// `S_max - 1` is the last real position of every batch row.
    pub lm_hidden: Var<R>,
    /// `[B, lm_hidden]` — the LAST row of `residual_lm`'s output.
    pub residual_hidden: Var<R>,
    /// `base_lm`'s cache, primed to `current_length == position`.
    pub base_cache: LayeredKvCache<R>,
    /// `residual_lm`'s cache, primed to the SAME `current_length`.
    pub residual_cache: LayeredKvCache<R>,
    /// `S_max`. The single shared position counter for the later decode
    /// loop: both caches advance together from here.
    pub position: usize,
    /// Rows in the batch, `B >= 1`.
    pub batch: usize,
    /// Per-row left padding, `kv_start.starts[b] = S_max - S_b`, on the
    /// device for the attention kernels and on the host for the per-row RoPE
    /// positions. `None` when no row is padded (every `B == 1` prefill), in
    /// which case every attention call is the unpadded one.
    pub kv_start: Option<LeftPad<R>>,
    /// Present only when the prefill ran via a `_capturing` entry point;
    /// always `None` on the plain paths, which allocate nothing extra.
    pub intermediates: Option<PrefillIntermediates<R>>,
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
