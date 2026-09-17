//! Unit B of the VoxCPM2 end-to-end orchestrator: the per-patch generation
//! loop that drives a finished [`PrefillState`] — one row or a left-padded
//! batch — until every row has hit a stop token or its cap.
//!
//! ```text
//! per iteration i, over every row of the batch at once:
//!   1. mu         = cat(lm_to_dit_proj(lm_hidden), res_to_dit_proj(residual_hidden))
//!   2. pred_feat  = feat_decoder(mu, cond = prefix_feat_cond^T)
//!   3. curr_embed = enc_to_lm_proj(feat_encoder(pred_feat.unsqueeze(1)))
//!   4. patches.push(pred_feat); prefix_feat_cond = pred_feat
//!   5. stop?      = argmax(stop(lm_hidden)) == 1   <- the CURRENT lm_hidden
//!   6. lm_hidden  = fsq(base_lm.decode_step(curr_embed, base_cache, position))
//!   7. residual_hidden = residual_lm.decode_step(
//!          fusion_concat_proj(cat(lm_hidden, curr_embed)), residual_cache, position)
//!   8. position  += 1
//! ```
//!
//! The VAE decode and the wav wrapper are a LATER unit: this module returns
//! patches and nothing else.
//!
//! # Traps this module exists to get right
//!
//! - **`prefix_feat_cond` starts as the ZERO text-pad patch**, `[1,
//!   patch_size, feat_dim]` of zeros — NOT the last patch of the reference
//!   audio. [`GenerateState::start`] builds it; [`PrefillState`] never
//!   feeds it.
//! - **The initial `lm_hidden` is UN-fsq'd.** It is the post-blend
//!   `enc_outputs` row at a TEXT position, which the prefill's blend left
//!   alone. From step 6 onward every `lm_hidden` IS fsq'd. Do not normalise
//!   the two: re-fsq'ing the prefill's row, and dropping the fsq in step 6,
//!   both stay shape-valid.
//! - **Step 7 consumes the POST-fsq `lm_hidden`**, and the concat order is
//!   `(lm_hidden, curr_embed)`. That concat is `2 * lm_hidden` wide, so
//!   swapping the halves is shape-valid and computes a different model.
//! - **ONE shared `position` counter drives BOTH caches and EVERY row.**
//!   The prefill primed both to `S_max` and both advance by exactly one per
//!   iteration. A finished row keeps stepping — its patches are computed and
//!   discarded — because the caches have no per-row compaction; only its
//!   `patch_len` and `outcomes` entries stand still.
//!   [`MiniCpm4Model::decode_step`] rejects `position != cache.seq_len()`, so
//!   a drift errors rather than corrupts — that check is why this counter is
//!   not duplicated per cache.
//! - **The stop guard is STRICTLY greater**: `i > min_len`, so a stop token
//!   at `i <= min_len` is ignored and at least `min_len + 2` patches are
//!   always emitted (`i = 0..=min_len` push unconditionally, and `i = min_len
//!   + 1` pushes before its check can fire).
//! - **The `max_len` exit is silent in the reference.** Here it is
//!   [`GenerateOutcome::MaxLen`], distinct from
//!   [`GenerateOutcome::StopToken`], so a caller can tell a finished
//!   utterance from a truncated one — per row, in
//!   [`GenerateState::outcomes`].
//! - **A stop that finishes the LAST open row skips steps 6-8**, exactly as
//!   the reference's `break` does for one row. A stop that leaves other
//!   rows open does not: the step runs through for everyone.
//! - **`step_with_noise` ignores `temperature`.** Scaling the noise draw is
//!   the drawing wrapper's job, the same split `solve_euler`/`sample` uses.
//!   A caller injecting `z` owns its scale.
//!
//! # The one device read
//!
//! Step 5 turns two logits per row into control flow, so it cannot stay on
//! device. The read is `argmax` ON DEVICE plus ONE `to_vec` of the resulting
//! `[batch]` index: eight bytes per row per patch, never the logits
//! themselves. See `validate::stop_predicted`.
//!
//! # Layout
//!
//! Split to stay under this repo's 500-line file limit: `types` (the public
//! types and their construction), `validate` (shape checks and the stop
//! decision, shared by every entry point), `step` (one iteration and the
//! noise draw), `run` (the whole-run driver), `capture` (the capturing
//! variant and its shared inner body) and `teacher_forced` (the batched
//! training-time counterpart).

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::fsq::{AuxProjections, ScalarQuantization};
use crate::model::audio::voxcpm::local_dit::{CfmOptions, LocalDit};
use crate::model::audio::voxcpm::local_encoder::LocalEncoder;
use crate::model::audio::voxcpm::minicpm4::MiniCpm4Model;
use crate::model::audio::voxcpm::model::config::VoxCpm2Config;
use crate::model::audio::voxcpm::model::loader::VoxCpm2Model;
use crate::model::audio::voxcpm::model::prefill::PrefillState;
use crate::model::traits::ModelClient;
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_mul_scalar};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, RandomOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

mod capture;
mod run;
mod step;
mod teacher_forced;
mod types;
mod validate;

#[cfg(test)]
pub(crate) mod test_support;

pub use capture::StepIntermediates;
pub use teacher_forced::TeacherForcedConditioning;
pub(crate) use types::STOP_CLASS;
pub use types::{
    GenerateOptions, GenerateOutcome, GenerateState, PatchGenerator, RowOptions, StepOutcome,
};

use validate::{check_patch, check_row, stop_predicted};
