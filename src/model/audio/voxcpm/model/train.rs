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
//! The sibling [`stop`] module adds the SECOND term upstream's fine-tuning
//! guide trains, `loss/stop`, and [`PatchGenerator::train_losses_with_noise`]/
//! [`PatchGenerator::train_losses`] combine it with this file's `loss/diff`
//! from ONE shared [`PatchGenerator::teacher_forced_conditioning`] call.
//!
//! # Why the DiT's `x`/`cond` need a transpose and `target_patches`/`noise`
//! do not
//!
//! [`LocalDit::forward`](crate::model::audio::voxcpm::local_dit::LocalDit::forward)
//! is pinned to `[batch, feat_dim, patch_size]` for both `x` and `cond`
//! (`local_dit/dit.rs`'s own doc comment: `in_proj` transposes to `[batch,
//! patch_size, feat_dim]` internally and transposes the output back). Every
//! OTHER tensor in this file — [`TeacherForcedConditioning::cond`](super::TeacherForcedConditioning::cond),
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
//! (`local_dit/sampler.rs`'s `dt_in`) always feeds zeros for the identical
//! reason, and this training step matches it: `dt` is NOT the flow
//! timestep `t` (that is a live, per-sample input) and is not free to drop,
//! since `SinusoidalPosEmb(0)` is not the zero vector and `delta_time_mlp`
//! still contributes a real bias.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::model::generate::{PatchGenerator, TeacherForcedConditioning};
use crate::model::audio::voxcpm::model::prefill::PrefillState;
use crate::model::traits::ModelClient;
use crate::nn::{flow_matching_interpolate, flow_matching_loss, var_contiguous};
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_transpose};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, RandomOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

mod stop;
pub use stop::{TrainLosses, stop_loss_from_logits};

/// Training-time conditioning dropout (upstream's `training_cfg_rate`):
/// when `drop_cond`, replace `cond.mu` with a zero tensor of the same
/// shape/dtype/device, everything else passed through unchanged. `pub(crate)`
/// so [`PatchGenerator::train_losses_with_noise`] applies it once, upstream
/// of BOTH the diff and stop terms, instead of each computing its own copy.
/// See [`PatchGenerator::cfm_loss_with_noise`]'s `drop_cond` doc for why
/// `mu` alone is the right tensor to zero.
pub(crate) fn apply_cond_dropout<R: Runtime<DType = DType>>(
    cond: TeacherForcedConditioning<R>,
    drop_cond: bool,
) -> Result<TeacherForcedConditioning<R>> {
    if !drop_cond {
        return Ok(cond);
    }
    let mu_zero = Var::new(
        Tensor::<R>::zeros(
            cond.mu.shape(),
            cond.mu.tensor().dtype(),
            cond.mu.tensor().device(),
        )
        .map_err(Error::Numr)?,
        false,
    );
    Ok(TeacherForcedConditioning {
        mu: mu_zero,
        ..cond
    })
}

/// Draws the conditioning-dropout bool for one training step from
/// `seed.wrapping_add(2)` — a THIRD independent stream alongside `t`
/// (`seed`) and `noise` (`seed + 1`), reusing [`RandomOps::rand_seeded`]
/// rather than a new RNG. `rate` is assumed already validated to `[0.0,
/// 1.0]` by the caller ([`check_training_cfg_rate`]).
fn draw_drop_cond<C, R>(client: &C, seed: u64, rate: f64) -> Result<bool>
where
    R: Runtime<DType = DType>,
    C: RandomOps<R>,
{
    if rate <= 0.0 {
        return Ok(false);
    }
    if rate >= 1.0 {
        return Ok(true);
    }
    let draw = client.rand_seeded(&[1], DType::F32, seed.wrapping_add(2))?;
    let value = draw.item::<f32>().map_err(Error::Numr)? as f64;
    Ok(value < rate)
}

/// Validates `training_cfg_rate` is in `[0.0, 1.0]` — a rate above 1 would
/// silently always-drop instead of erroring, and a negative rate is
/// meaningless. See [`PatchGenerator::cfm_loss`]'s doc comment for why 0 is
/// accepted but discouraged (upstream's default is 0.1).
fn check_training_cfg_rate(rate: f64) -> Result<()> {
    if !(0.0..=1.0).contains(&rate) {
        return Err(Error::InvalidArgument {
            arg: "training_cfg_rate",
            reason: format!("expected a value in [0.0, 1.0], got {rate}"),
        });
    }
    Ok(())
}

impl<R: Runtime<DType = DType>> PatchGenerator<'_, R> {
    /// One CFM training step's loss, with `t` and `noise` supplied by the
    /// caller — the training-step counterpart of
    /// [`Self::step_with_noise`](super::PatchGenerator::step_with_noise):
    /// deterministic, so a caller can pin both draws and reproduce a step
    /// bit for bit (e.g. to overfit one fixed batch across many optimizer
    /// steps without the target itself drifting).
    ///
    /// - `target_patches`: `[T >= 1, patch_size, feat_dim]` ground truth,
    ///   teacher-forced through [`Self::teacher_forced_conditioning`]. See
    ///   that method's own doc comment for `prefill`'s requirements
    ///   (`prefill.intermediates` must be `Some` whenever `prefill.position
    ///   > 0`).
    /// - `t`: `[T]`, one flow timestep per patch, expected in `[0, 1]` (not
    ///   validated — an out-of-range `t` is a caller bug, not a shape
    ///   error, and [`flow_matching_interpolate`]'s formula is well-defined
    ///   for any `t`).
    /// - `noise`: `[T, patch_size, feat_dim]`, matching `target_patches`.
    /// - `drop_cond`: training-time conditioning dropout (upstream's
    ///   `training_cfg_rate`). When true, `cond.mu` is replaced with a zero
    ///   tensor of the same shape/dtype/device BEFORE either loss term is
    ///   computed — the same construction
    ///   [`LocalDit::solve_euler`](crate::model::audio::voxcpm::local_dit::LocalDit::solve_euler)
    ///   uses to build its unconditional half at inference (`sampler.rs`'s
    ///   `mu_zero`: the CFG-doubled batch differs in `mu` ALONE, `cond` is
    ///   duplicated unchanged). Zeroing `mu` some fraction of training steps
    ///   teaches the model to produce a sane, TEXT-INDEPENDENT prediction
    ///   when `mu` is absent, which is what makes classifier-free guidance
    ///   at inference actually work; upstream's FAQ calls skipping this "the
    ///   most common fine-tuning failure mode" (text gets ignored) and says
    ///   explicitly not to train with it always off. `cond`, `x_t`/`noise`
    ///   and `t` are untouched — only `mu` defines the unconditional branch.
    ///
    /// Returns a scalar `Var<R>` whose graph reaches every adapter
    /// `apply_lora` attached under `feat_encoder`, `base_lm`,
    /// `residual_lm`, `feat_decoder`, `fsq` or `aux` — `target_patches`,
    /// `t` and `noise` themselves carry no grad; they are the target, not a
    /// trained input. Differentiate with [`numr::autograd::backward`].
    ///
    /// Errors on a shape-mismatched `target_patches`/`t`/`noise` (never a
    /// panic), and propagates every error
    /// [`Self::teacher_forced_conditioning`] or [`LocalDit::forward`] would
    /// raise.
    ///
    /// [`LocalDit::forward`]:
    ///     crate::model::audio::voxcpm::local_dit::LocalDit::forward
    pub fn cfm_loss_with_noise<C>(
        &self,
        client: &C,
        prefill: &PrefillState<R>,
        target_patches: &Tensor<R>,
        t: &Tensor<R>,
        noise: &Tensor<R>,
        drop_cond: bool,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R>,
        R::Client: ModelClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + TypeConversionOps<R>
            + DequantOps<R>,
    {
        let shape = target_patches.shape().to_vec();
        if shape.len() != 3 || shape[0] == 0 {
            return Err(Error::InvalidArgument {
                arg: "target_patches",
                reason: format!("expected rank-3 [T >= 1, patch_size, feat_dim], got {shape:?}"),
            });
        }
        let tcount = shape[0];
        if t.shape() != [tcount] {
            return Err(Error::InvalidArgument {
                arg: "t",
                reason: format!("expected [{tcount}], got {:?}", t.shape()),
            });
        }
        if noise.shape() != shape.as_slice() {
            return Err(Error::InvalidArgument {
                arg: "noise",
                reason: format!("expected {shape:?}, got {:?}", noise.shape()),
            });
        }

        // Every remaining shape check ([`Self::teacher_forced_conditioning`]'s
        // `patch_size`/`feat_dim` check, `flow_matching_interpolate`'s
        // noise/data match, `LocalDit::forward`'s own checks) is already
        // enforced by the callee — not re-implemented here.
        let cond = self.teacher_forced_conditioning(client, prefill, target_patches)?;
        let cond = apply_cond_dropout(cond, drop_cond)?;
        self.cfm_loss_from_conditioning(client, &cond, target_patches, t, noise, tcount)
    }

    /// The diffusion half of [`Self::cfm_loss_with_noise`], factored out so
    /// [`Self::train_losses_with_noise`] can share ONE
    /// [`Self::teacher_forced_conditioning`] call with the stop loss instead
    /// of paying for `base_lm`/`residual_lm`'s full-sequence forward twice.
    /// `tcount` is `target_patches.shape()[0]`, already validated by the
    /// caller. `pub(crate)`, not private: the sibling `train::stop` module's
    /// `Self::train_losses_with_noise` shares this exact diffusion
    /// computation instead of recomputing it.
    pub(crate) fn cfm_loss_from_conditioning<C>(
        &self,
        client: &C,
        cond: &TeacherForcedConditioning<R>,
        target_patches: &Tensor<R>,
        t: &Tensor<R>,
        noise: &Tensor<R>,
        tcount: usize,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R>,
        R::Client: ModelClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + TypeConversionOps<R>
            + DequantOps<R>,
    {
        let dtype = cond.mu.tensor().dtype();
        let device = cond.mu.tensor().device();
        let target_var = Var::new(target_patches.to_dtype(dtype)?, false);
        let noise_var = Var::new(noise.to_dtype(dtype)?, false);
        let t_var = Var::new(t.to_dtype(dtype)?, false);

        // The CFM probability path, still in [T, patch_size, feat_dim] —
        // the layout `target_patches`/`noise` arrived in.
        let x_t = flow_matching_interpolate(client, &noise_var, &target_var, &t_var)?;

        // Into the DiT's [T, feat_dim, patch_size] layout — see the module
        // docs for why both `x_t` and `cond.cond` are transposed here and
        // the estimator's output is transposed back below.
        let x_t_dit = var_contiguous(&var_transpose(&x_t)?)?;
        let cond_dit = var_contiguous(&var_transpose(&cond.cond)?)?;

        // `dt` is the mean-velocity delta, always zero on this checkpoint
        // (`mean_mode` is false) — see the module docs.
        let dt_zero = Var::new(Tensor::<R>::zeros(&[tcount], dtype, device)?, false);

        let v_pred_dit = self
            .feat_decoder
            .forward(client, &x_t_dit, &cond.mu, &t_var, &cond_dit, &dt_zero)?;
        // Back to [T, patch_size, feat_dim] to compare against
        // `noise`/`target_patches` in their native layout.
        let v_pred = var_contiguous(&var_transpose(&v_pred_dit)?)?;

        flow_matching_loss(client, &v_pred, &noise_var, &target_var)
    }

    /// [`Self::cfm_loss_with_noise`], drawing `t` and `noise` itself.
    ///
    /// `t` is `client.rand_seeded(seed)` — uniform `[0, 1)` — and `noise` is
    /// `client.randn_seeded(seed + 1)`, mirroring how
    /// [`LocalDit::sample`](crate::model::audio::voxcpm::local_dit::LocalDit::sample)
    /// draws its own seeded noise: one seed reproduces the whole step, and
    /// the `+ 1` offset keeps the timestep draw and the noise draw on
    /// independent streams instead of the same one. `randn_seeded`/
    /// `rand_seeded` are reproducible per backend only — see
    /// [`numr::ops::RandomOps::randn_seeded`] for why a CPU run and a CUDA
    /// run of one seed draw differently.
    ///
    /// `training_cfg_rate` is upstream's `training_cfg_rate` — the per-step
    /// probability of conditioning dropout (see
    /// [`Self::cfm_loss_with_noise`]'s `drop_cond` doc), drawn from
    /// `seed.wrapping_add(2)`: a third stream independent of `t`/`noise`.
    /// Upstream defaults this to 0.1 and its FAQ calls 0 "the most common
    /// fine-tuning failure mode" (the model learns to ignore the text).
    /// Must be in `[0.0, 1.0]` — an out-of-range rate is an
    /// [`Error::InvalidArgument`], not a silent always-drop.
    ///
    /// Errors on a shape-mismatched `target_patches` — same as
    /// [`Self::cfm_loss_with_noise`] once the draw shapes are derived from
    /// it, so a caller cannot see a panic from either path.
    pub fn cfm_loss<C>(
        &self,
        client: &C,
        prefill: &PrefillState<R>,
        target_patches: &Tensor<R>,
        seed: u64,
        training_cfg_rate: f64,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R> + RandomOps<R>,
        R::Client: ModelClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + TypeConversionOps<R>
            + DequantOps<R>,
    {
        check_training_cfg_rate(training_cfg_rate)?;
        let shape = target_patches.shape();
        if shape.len() != 3 || shape[0] == 0 {
            return Err(Error::InvalidArgument {
                arg: "target_patches",
                reason: format!("expected rank-3 [T >= 1, patch_size, feat_dim], got {shape:?}"),
            });
        }
        let tcount = shape[0];
        let dtype = target_patches.dtype();

        let t = client.rand_seeded(&[tcount], dtype, seed)?;
        let noise = client.randn_seeded(shape, dtype, seed.wrapping_add(1))?;
        let drop_cond = draw_drop_cond::<C, R>(client, seed, training_cfg_rate)?;
        self.cfm_loss_with_noise(client, prefill, target_patches, &t, &noise, drop_cond)
    }
}

#[cfg(test)]
mod tests;
