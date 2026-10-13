//! Training stage of the VoxCPM2 end-to-end orchestrator: wire teacher-forced
//! conditioning ([`PatchGenerator::teacher_forced_conditioning`]) into a
//! conditional flow-matching (CFM) training loss, differentiable back to
//! the LoRA adapters via [`numr::autograd::backward`].
//!
//! ```text
//! cond      = teacher_forced_conditioning(prefill, target_patches)   [T, ...]
//! x_t       = flow_matching_interpolate(noise, target_patches, t)    [T, patch_size, feat_dim]
//! v_pred    = feat_decoder.forward(x_t, cond.mu, t, cond.cond, 0)    [T, patch_size, feat_dim]
//! loss      = flow_matching_loss(v_pred, noise, target_patches)      scalar
//! ```
//!
//! `losses`: optimizer split, the stop term. `diff_loss`: shared layout, why `dt` is zero.
//!
//! - `dropout`: conditioning dropout (`training_cfg_rate`) and its draw
//! - `cfm`: `cfm_loss_with_noise` and the seeded `cfm_loss` wrapper
//! - `diff_loss`: `cfm_loss_from_conditioning`, the differentiable DiT pass
//! - `stop`: the stop-classifier term
//! - `losses`: [`TrainLosses`] and the combined `train_losses*` entry points
//!
//! [`PatchGenerator::teacher_forced_conditioning`]: super::PatchGenerator::teacher_forced_conditioning

mod cfm;
mod diff_loss;
mod dropout;
mod losses;
mod stop;

pub use losses::TrainLosses;
pub use stop::stop_loss_from_logits;
