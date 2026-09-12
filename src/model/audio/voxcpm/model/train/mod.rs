//! Unit D of the VoxCPM2 end-to-end orchestrator: wire teacher-forced
//! conditioning ([`PatchGenerator::teacher_forced_conditioning`]) into a
//! conditional flow-matching (CFM) training loss, differentiable back to
//! the LoRA adapters via [`numr::autograd::backward`].
//!
//! ```text
//! cond      = teacher_forced_conditioning(prefill, target_patches)   [T, ...]
//! x_t       = flow_matching_interpolate(noise, target_patches, t)    [T, patch_size, feat_dim]
//! v_pred    = feat_decoder.forward(x_t^T, cond.mu, t, cond.cond^T, 0) [T, feat_dim, patch_size]
//! loss      = flow_matching_loss(v_pred^T, noise, target_patches)    scalar
//! ```
//!
//! The optimizer step itself (building a `HashMap<TensorId, Tensor<R>>` from
//! [`crate::nn::Module::trainable_parameters`] and calling
//! [`crate::trainer::simple::SimpleTrainer::step`]) is the CALLER's job, the
//! same way `crate::trainer` already works for every other model in this
//! crate — nothing here is VoxCPM2-specific about running an optimizer.
//!
//! The `stop`/`losses` modules add the SECOND term the reference VoxCPM
//! fine-tuning guide trains, `loss/stop`, and [`PatchGenerator::train_losses_with_noise`]/
//! [`PatchGenerator::train_losses`] combine it with the `loss/diff` term
//! from ONE shared [`PatchGenerator::teacher_forced_conditioning`] call.
//!
//! # Why the DiT's `x`/`cond` need a transpose and `target_patches`/`noise`
//! do not
//!
//! [`LocalDit::forward`](crate::model::audio::voxcpm::local_dit::LocalDit::forward)
//! is pinned to `[batch, feat_dim, patch_size]` for both `x` and `cond`
//! (`local_dit/dit.rs`'s own doc comment: `in_proj` transposes to `[batch,
//! patch_size, feat_dim]` internally and transposes the output back). Every
//! OTHER tensor in this module — [`TeacherForcedConditioning::cond`](super::TeacherForcedConditioning::cond),
//! `target_patches`, `noise`, and therefore `flow_matching_interpolate`'s
//! output `x_t` — lives in the opposite layout, `[T, patch_size, feat_dim]`,
//! because that is what [`PatchGenerator::teacher_forced_conditioning`]
//! and the per-patch loop's own `prefix_feat_cond`/emitted patches both use.
//! So `x_t` and `cond.cond` are transposed going INTO the estimator, and its
//! output is transposed back before `flow_matching_loss` compares it against
//! `noise`/`target_patches` in THEIR native layout. Skipping either
//! transpose is shape-valid (both axes are frequently the same order of
//! magnitude in a small fixture) and silently trains against the wrong
//! axis — see `local_dit/dit.rs`'s own module docs for why this exact trap
//! is called out there too.
//!
//! # Why `dt` is zero
//!
//! [`LocalDit::forward`](crate::model::audio::voxcpm::local_dit::LocalDit::forward)'s
//! `dt` argument is the MEAN-VELOCITY delta, live only when
//! `LocalDitConfig::mean_mode` is set — false on this checkpoint,
//! per that field's own doc comment. The inference sampler
//! (`local_dit/sampler`'s `dt_in`) always feeds zeros for the identical
//! reason, and this training step matches it: `dt` is NOT the flow
//! timestep `t` (that is a live, per-sample input) and is not free to drop,
//! since `SinusoidalPosEmb(0)` is not the zero vector and `delta_time_mlp`
//! still contributes a real bias.
//!
//! - `dropout`: conditioning dropout (`training_cfg_rate`) and its draw
//! - `cfm`: `cfm_loss_with_noise` and the seeded `cfm_loss` wrapper
//! - `diff_loss`: `cfm_loss_from_conditioning`, the differentiable DiT pass
//! - `stop`: the stop-classifier term
//! - `losses`: [`TrainLosses`] and the combined `train_losses*` entry points
//!
//! [`PatchGenerator::teacher_forced_conditioning`]: super::PatchGenerator::teacher_forced_conditioning
//! [`PatchGenerator::train_losses_with_noise`]: super::PatchGenerator::train_losses_with_noise
//! [`PatchGenerator::train_losses`]: super::PatchGenerator::train_losses

mod cfm;
mod diff_loss;
mod dropout;
mod losses;
mod stop;

pub use losses::TrainLosses;
pub use stop::stop_loss_from_logits;
