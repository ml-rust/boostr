//! The stop-classifier training term and the combined `loss/diff` +
//! `loss/stop` entry point — a sibling of [`super`]'s CFM (`loss/diff`)
//! loss, split into its own file to keep `train.rs` under the 500-line
//! model-file limit.
//!
//! Upstream's fine-tuning guide trains BOTH terms (`lambdas: {loss/diff:
//! 1.0, loss/stop: 1.0}`) and its own FAQ names runaway generation
//! ("generation doesn't stop") as a top failure mode, recommending a higher
//! `loss/stop` weight when it happens. Training on [`super::PatchGenerator::cfm_loss`]
//! alone — this crate's previous state — never trains the stop head at all:
//! `stop_proj`/`stop_head` sit OUTSIDE that loss's graph (see
//! `fsq/layer.rs`'s straight-through-estimator doc comment for the measured
//! "zero gradient" finding), so a model fine-tuned that way keeps whatever
//! stop behavior it started with.
//!
//! # Why the stop-head input is `TeacherForcedConditioning::lm_hidden`, not
//! a fresh `base_lm` forward
//!
//! `generate.rs`'s per-patch loop step 5 reads `aux.stop(client,
//! &state.prefill.lm_hidden)` — the CURRENT `lm_hidden`, i.e. the hidden
//! state from BEFORE that iteration's steps 6-7 overwrite it for the next
//! one. That is the SAME shifted value step 1 feeds `lm_to_dit_proj` to
//! build `mu`'s LM half (`teacher_forced.rs`'s own "The shift" section).
//! Since [`super::teacher_forced_conditioning`] already computes that
//! shifted value once (as `lm_shifted`) to build `mu`, it is exposed on
//! [`TeacherForcedConditioning::lm_hidden`] and consumed here rather than
//! re-running `base_lm`/`residual_lm`'s full-sequence forward a second
//! time.

use super::{
    Error, ModelClient, PatchGenerator, PrefillState, Result, TeacherForcedConditioning,
    apply_cond_dropout, check_training_cfg_rate, draw_drop_cond,
};
use crate::model::audio::voxcpm::model::generate::STOP_CLASS;
use crate::nn::cross_entropy_loss;
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_add, var_mul_scalar};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, RandomOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// The diffusion loss (`loss/diff`) and stop loss (`loss/stop`) from one
/// training step, plus their weighted sum — mirrors upstream's own
/// TensorBoard scalars so a caller can log all three the same way. See
/// [`PatchGenerator::train_losses_with_noise`].
pub struct TrainLosses<R: Runtime> {
    pub diff: Var<R>,
    pub stop: Var<R>,
    pub total: Var<R>,
}

/// Per-patch stop-classifier target: class 0 ("continue") for every patch
/// except the LAST, class 1 ([`STOP_CLASS`], "stop") for the final one.
///
/// This mirrors [`PatchGenerator::generate`](super::super::generate::PatchGenerator::generate)'s
/// own stop check (`generate.rs`'s module doc, step 5): the reference loop
/// only fires `STOP_CLASS` past `min_len`, and under teacher forcing the
/// LAST ground-truth patch IS where generation is supposed to end. Training
/// the head to fire there — and nowhere else — is what makes inference stop
/// at the right length instead of running on. See this module's own doc
/// comment for the runaway-generation failure mode this term exists to fix.
fn stop_targets<R: Runtime<DType = DType>>(tcount: usize, device: &R::Device) -> Result<Tensor<R>> {
    let mut targets = vec![0i64; tcount];
    if let Some(last) = targets.last_mut() {
        *last = STOP_CLASS;
    }
    Tensor::<R>::from_slice(&targets, &[tcount], device).map_err(Error::Numr)
}

/// 2-class cross-entropy over per-patch stop logits, target built by
/// [`stop_targets`]. `logits` is `[..., T, 2]` (T = the product of every
/// dimension but the last); a rank-3 `[1, T, 2]` from
/// [`AuxProjections::stop`](crate::model::audio::voxcpm::fsq::AuxProjections::stop)
/// and a rank-2 `[T, 2]` built directly (as tests do, to exercise the
/// target construction without running the model) both work.
pub fn stop_loss_from_logits<R, C>(client: &C, logits: &Var<R>) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R> + TypeConversionOps<R>,
    R::Client: TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ReduceOps<R>
        + IndexingOps<R>
        + DequantOps<R>,
{
    let shape = logits.shape();
    if shape.len() < 2 {
        return Err(Error::InvalidArgument {
            arg: "logits",
            reason: format!("expected at least rank-2 [..., T, 2], got {shape:?}"),
        });
    }
    let tcount: usize = shape[..shape.len() - 1].iter().product();
    let device = logits.tensor().device();
    let targets = stop_targets::<R>(tcount, device)?;
    cross_entropy_loss(client, logits, &targets)
}

impl<R: Runtime<DType = DType>> PatchGenerator<'_, R> {
    /// The stop-classifier training term. `cond.lm_hidden` (`[1, T,
    /// lm_hidden]`) is run through `aux.stop` — the SAME composition and the
    /// SAME per-patch input `generate.rs`'s step 5 reads. See this module's
    /// doc comment for why `cond.lm_hidden` is the right input.
    ///
    /// Returns a scalar `Var<R>` whose graph reaches `stop_proj`/`stop_head`
    /// (and everything upstream of `cond.lm_hidden`) — the CFM loss alone
    /// does not; see this module's doc comment for the measured "zero
    /// gradient" finding that motivated this method.
    pub fn stop_loss<C>(&self, client: &C, cond: &TeacherForcedConditioning<R>) -> Result<Var<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R>,
        R::Client: TensorOps<R>
            + ActivationOps<R>
            + ScalarOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + DequantOps<R>,
    {
        let logits = self.aux.stop(client, &cond.lm_hidden)?;
        stop_loss_from_logits(client, &logits)
    }

    /// [`Self::cfm_loss_with_noise`] and [`Self::stop_loss`] from ONE shared
    /// [`Self::teacher_forced_conditioning`] call, combined as `lambda_diff *
    /// diff + lambda_stop * stop` — the two terms upstream's fine-tuning
    /// guide logs separately as `loss/diff` and `loss/stop`. Passing
    /// `lambda_diff = 1.0, lambda_stop = 1.0` reproduces upstream's own
    /// default `lambdas:` block; upstream's FAQ recommends raising
    /// `lambda_stop` specifically when generation runs away (the model never
    /// emits a stop token), which is why both weights are caller-supplied
    /// rather than baked in.
    ///
    /// `lambda_stop = 0.0` makes `total` numerically equal
    /// `lambda_diff * diff` (`stop` is still computed and returned, just
    /// weighted out of `total`) — see `train/tests.rs` for the check that
    /// pins this against [`Self::cfm_loss_with_noise`] directly.
    ///
    /// `drop_cond` (upstream's `training_cfg_rate` draw) is applied to
    /// `cond` ONCE, right after [`Self::teacher_forced_conditioning`]
    /// returns, so BOTH `diff` and `stop` see the same conditioning object
    /// — see [`super::apply_cond_dropout`] and
    /// [`Self::cfm_loss_with_noise`]'s `drop_cond` doc for why only `mu` is
    /// zeroed. `stop` reads `cond.lm_hidden`, not `cond.mu`, so it is
    /// numerically UNAFFECTED by `drop_cond` either way — the dropout is
    /// deliberately scoped to the diffusion term alone, matching upstream.
    #[allow(clippy::too_many_arguments)]
    pub fn train_losses_with_noise<C>(
        &self,
        client: &C,
        prefill: &PrefillState<R>,
        target_patches: &Tensor<R>,
        t: &Tensor<R>,
        noise: &Tensor<R>,
        lambda_diff: f64,
        lambda_stop: f64,
        drop_cond: bool,
    ) -> Result<TrainLosses<R>>
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

        // ONE forward through `teacher_forced_conditioning` — shared by
        // both terms, so this pays for `base_lm`/`residual_lm`'s
        // full-sequence forward exactly once, the same as
        // `cfm_loss_with_noise` alone would.
        let cond = self.teacher_forced_conditioning(client, prefill, target_patches)?;
        let cond = apply_cond_dropout(cond, drop_cond)?;

        let diff =
            self.cfm_loss_from_conditioning(client, &cond, target_patches, t, noise, tcount)?;
        let stop = self.stop_loss(client, &cond)?;

        let diff_scaled = var_mul_scalar(&diff, lambda_diff, client)?;
        let stop_scaled = var_mul_scalar(&stop, lambda_stop, client)?;
        let total = var_add(&diff_scaled, &stop_scaled, client)?;

        Ok(TrainLosses { diff, stop, total })
    }

    /// [`Self::train_losses_with_noise`], drawing `t` and `noise` itself —
    /// the combined-loss counterpart of [`Self::cfm_loss`], same seeded-draw
    /// convention (`t` from `seed`, `noise` from `seed + 1`).
    ///
    /// `training_cfg_rate` is upstream's per-step conditioning-dropout
    /// probability, drawn from `seed.wrapping_add(2)` — see
    /// [`super::PatchGenerator::cfm_loss`]'s doc for the default (0.1) and
    /// why 0 is discouraged. Must be in `[0.0, 1.0]`, else
    /// [`Error::InvalidArgument`].
    #[allow(clippy::too_many_arguments)]
    pub fn train_losses<C>(
        &self,
        client: &C,
        prefill: &PrefillState<R>,
        target_patches: &Tensor<R>,
        seed: u64,
        lambda_diff: f64,
        lambda_stop: f64,
        training_cfg_rate: f64,
    ) -> Result<TrainLosses<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R> + RandomOps<R>,
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
        self.train_losses_with_noise(
            client,
            prefill,
            target_patches,
            &t,
            &noise,
            lambda_diff,
            lambda_stop,
            drop_cond,
        )
    }
}
