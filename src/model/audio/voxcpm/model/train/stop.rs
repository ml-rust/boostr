//! The stop-classifier training term (`loss/stop`).
//!
//! See the sibling `losses` module for why the reference VoxCPM fine-tuning
//! guide trains this term alongside `loss/diff`, and for the entry points
//! that combine the two.
//!
//! # Why the stop-head input is `TeacherForcedConditioning::lm_hidden`, not
//! a fresh `base_lm` forward
//!
//! `generate.rs`'s per-patch loop step 5 reads `aux.stop(client,
//! &state.prefill.lm_hidden)` — the CURRENT `lm_hidden`, i.e. the hidden
//! state from BEFORE that iteration's steps 6-7 overwrite it for the next
//! one. That is the SAME shifted value step 1 feeds `lm_to_dit_proj` to
//! build `mu`'s LM half (`teacher_forced.rs`'s own "The shift" section).
//! Since [`PatchGenerator::teacher_forced_conditioning`] already computes that
//! shifted value once (as `lm_shifted`) to build `mu`, it is exposed on
//! [`TeacherForcedConditioning::lm_hidden`] and consumed here rather than
//! re-running `base_lm`/`residual_lm`'s full-sequence forward a second
//! time.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::model::generate::{
    PatchGenerator, STOP_CLASS, TeacherForcedConditioning,
};
use crate::model::traits::ModelClient;
use crate::nn::cross_entropy_loss;
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, IndexingOps, ReduceOps, ScalarOps, TensorOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

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
}

#[cfg(test)]
mod tests {
    use super::super::cfm::tests::{T, target_patches};
    use super::*;
    use crate::model::audio::voxcpm::model::generate::tests::support::{fixture, state};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn stop_loss_is_finite_and_positive() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let generator = fx.generator();
        let st = state(&fx, &device);

        let target = target_patches(0.4, &device);
        let cond = generator
            .teacher_forced_conditioning(&client, &st.prefill, &target)
            .expect("teacher_forced_conditioning");
        let loss = generator.stop_loss(&client, &cond).expect("stop_loss");
        let val = loss.tensor().to_vec::<f32>()[0];
        assert!(val.is_finite(), "stop loss must be finite, got {val}");
        assert!(val > 0.0, "stop loss must be positive, got {val}");
    }

    /// Pins the target construction itself, not the model: `logits` are built
    /// by hand, so this catches a wrong-position stop target (e.g. class 1 on
    /// EVERY patch, or on patch 0) even if the model side of `stop_loss` is
    /// correct.
    #[test]
    fn stop_target_is_the_last_patch_only() {
        let (client, device) = cpu_setup();

        // Confidently "continue" (class 0) on every one of the T = 3 patches,
        // including the last — the runaway-generation failure mode this loss
        // exists to penalise.
        #[rustfmt::skip]
        let all_continue = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &[5.0f32, -5.0,
                  5.0, -5.0,
                  5.0, -5.0],
                &[T, 2],
                &device,
            )
            .expect("logits"),
            false,
        );
        // Confidently "continue" for every patch except the last, "stop" on the
        // last — exactly the target `stop_targets` builds.
        #[rustfmt::skip]
        let stop_on_last = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &[5.0f32, -5.0,
                  5.0, -5.0,
                  -5.0, 5.0],
                &[T, 2],
                &device,
            )
            .expect("logits"),
            false,
        );

        let loss_all_continue =
            stop_loss_from_logits(&client, &all_continue).expect("stop_loss_from_logits");
        let loss_stop_on_last =
            stop_loss_from_logits(&client, &stop_on_last).expect("stop_loss_from_logits");
        let val_all_continue = loss_all_continue.tensor().to_vec::<f32>()[0];
        let val_stop_on_last = loss_stop_on_last.tensor().to_vec::<f32>()[0];

        assert!(
            val_all_continue > val_stop_on_last,
            "predicting \"continue\" on the LAST patch too must score worse than \
             predicting \"stop\" only there: all_continue={val_all_continue} \
             stop_on_last={val_stop_on_last}"
        );
        assert!(
            val_stop_on_last < 0.1,
            "a confident, correctly-placed stop prediction should be near zero, \
             got {val_stop_on_last}"
        );
    }
}
