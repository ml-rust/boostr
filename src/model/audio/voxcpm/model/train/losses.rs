//! The combined `loss/diff` + `loss/stop` entry points.
//!
//! The reference VoxCPM fine-tuning guide trains BOTH terms (`lambdas: {loss/diff:
//! 1.0, loss/stop: 1.0}`) and its own FAQ names runaway generation
//! ("generation doesn't stop") as a top failure mode, recommending a higher
//! `loss/stop` weight when it happens. Training on [`PatchGenerator::cfm_loss`]
//! alone — this crate's previous state — never trains the stop head at all:
//! `stop_proj`/`stop_head` sit OUTSIDE that loss's graph (see
//! `fsq/layer`'s straight-through-estimator doc comment for the measured
//! "zero gradient" finding), so a model fine-tuned that way keeps whatever
//! stop behavior it started with.

use super::dropout::{apply_cond_dropout, check_training_cfg_rate, draw_drop_cond};
use crate::error::{Error, Result};
use crate::model::audio::voxcpm::model::generate::PatchGenerator;
use crate::model::audio::voxcpm::model::prefill::PrefillState;
use crate::model::traits::ModelClient;
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
/// training step, plus their weighted sum — mirrors the reference VoxCPM
/// implementation's own TensorBoard scalars so a caller can log all three the same way. See
/// [`PatchGenerator::train_losses_with_noise`].
pub struct TrainLosses<R: Runtime> {
    pub diff: Var<R>,
    pub stop: Var<R>,
    pub total: Var<R>,
}

impl<R: Runtime<DType = DType>> PatchGenerator<'_, R> {
    /// [`Self::cfm_loss_with_noise`] and [`Self::stop_loss`] from ONE shared
    /// [`Self::teacher_forced_conditioning`] call, combined as `lambda_diff *
    /// diff + lambda_stop * stop` — the two terms the reference VoxCPM
    /// fine-tuning guide logs separately as `loss/diff` and `loss/stop`. Passing
    /// `lambda_diff = 1.0, lambda_stop = 1.0` reproduces the reference VoxCPM
    /// implementation's own default `lambdas:` block; its FAQ recommends raising
    /// `lambda_stop` specifically when generation runs away (the model never
    /// emits a stop token), which is why both weights are caller-supplied
    /// rather than baked in.
    ///
    /// `lambda_stop = 0.0` makes `total` numerically equal
    /// `lambda_diff * diff` (`stop` is still computed and returned, just
    /// weighted out of `total`) — see `train/tests.rs` for the check that
    /// pins this against [`Self::cfm_loss_with_noise`] directly.
    ///
    /// `drop_cond` (the reference VoxCPM implementation's `training_cfg_rate` draw) is applied to
    /// `cond` ONCE, right after [`Self::teacher_forced_conditioning`]
    /// returns, so BOTH `diff` and `stop` see the same conditioning object
    /// — see `dropout::apply_cond_dropout` and
    /// [`Self::cfm_loss_with_noise`]'s `drop_cond` doc for why only `mu` is
    /// zeroed. `stop` reads `cond.lm_hidden`, not `cond.mu`, so it is
    /// numerically UNAFFECTED by `drop_cond` either way — the dropout is
    /// deliberately scoped to the diffusion term alone, matching the
    /// reference VoxCPM implementation.
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
        C: ModelClient<R> + TypeConversionOps<R> + 'static,
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
    /// `training_cfg_rate` is the reference VoxCPM implementation's per-step
    /// conditioning-dropout probability, drawn from `seed.wrapping_add(2)` — see
    /// [`crate::model::audio::voxcpm::model::PatchGenerator::cfm_loss`]'s doc for the default (0.1) and
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
        C: ModelClient<R> + TypeConversionOps<R> + RandomOps<R> + 'static,
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

#[cfg(test)]
mod tests {
    //! [`stop_loss_reaches_stop_head_but_diff_alone_does_not`] adapts
    //! `stop_proj`/`stop_head` specifically to demonstrate the fact the
    //! `diff_loss` tests' doc comment states: `cfm_loss` alone leaves them
    //! with NO gradient entry, and [`PatchGenerator::train_losses_with_noise`]'s
    //! `total` is what puts them on the graph.

    use super::super::cfm::tests::{T, target_patches};
    use super::*;
    use crate::model::audio::voxcpm::model::generate::tests::support::{fixture, state};
    use crate::nn::{LoraTargets, Module};
    use crate::test_utils::cpu_setup;
    use numr::autograd::backward;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::TensorId;

    /// The finding this whole unit exists to fix: `cfm_loss` alone never calls
    /// `aux.stop`, so `stop_proj`/`stop_head` get NO gradient entry from it —
    /// and `train_losses_with_noise`'s `total` is what puts them on the graph.
    #[test]
    fn stop_loss_reaches_stop_head_but_diff_alone_does_not() {
        let (client, device) = cpu_setup();
        let mut fx = fixture(false, &device);

        let rank = 2;
        let alpha = 4.0;
        // Adapt the stop chain AND one projection that IS inside the CFM graph.
        // Without the second, the diffusion loss would have no trainable input at
        // all and `backward` would fail outright rather than returning an empty
        // gradient — which proves the same point, but by erroring instead of by
        // measuring. Adapting `lm_to_dit_proj` too keeps the diff-only backward
        // well-formed, so "the stop head got nothing" is an OBSERVATION rather
        // than an exception.
        let stop_targets_list = LoraTargets::new(["stop_proj", "stop_head"]);
        fx.aux
            .apply_lora(&stop_targets_list, rank, alpha, &device, "")
            .expect("apply_lora aux stop chain");
        fx.aux
            .apply_lora(
                &LoraTargets::new(["lm_to_dit_proj"]),
                rank,
                alpha,
                &device,
                "",
            )
            .expect("apply_lora aux lm_to_dit_proj");

        // Only the STOP-chain adapters are under test; `lm_to_dit_proj`'s exist
        // solely to keep the diff-only graph alive.
        let mut adapters: Vec<(String, TensorId)> = Vec::new();
        for (name, var) in Module::named_parameters(&fx.aux) {
            let is_stop_chain = name.contains("stop_proj") || name.contains("stop_head");
            if var.requires_grad()
                && is_stop_chain
                && (name.ends_with("lora_a") || name.ends_with("lora_b"))
            {
                adapters.push((name, var.id()));
            }
        }
        assert!(
            !adapters.is_empty(),
            "stop_proj/stop_head must have matched the target list above"
        );

        let st = state(&fx, &device);
        let target = target_patches(0.6, &device);
        let noise = target_patches(1.9, &device);
        let ts = Tensor::<CpuRuntime>::from_slice(&[0.2f32, 0.5, 0.8], &[T], &device).expect("t");

        // The CFM loss ALONE: the stop head sits outside its graph, so every
        // adapter above must be either absent from the grad store or all-zero.
        let diff_only = {
            let generator = fx.generator();
            generator
                .cfm_loss_with_noise(&client, &st.prefill, &target, &ts, &noise, false)
                .expect("cfm_loss_with_noise")
        };
        let diff_grads = backward(&diff_only, &client).expect("backward diff_only");
        for (name, id) in &adapters {
            let has_signal = diff_grads
                .get(*id)
                .map(|grad| {
                    let values: Vec<f32> = grad.contiguous().expect("contiguous").to_vec();
                    values.iter().any(|v| *v != 0.0)
                })
                .unwrap_or(false);
            assert!(
                !has_signal,
                "adapter {name} got a nonzero gradient from cfm_loss ALONE — the \
                 stop head should be off that graph entirely"
            );
        }

        // The combined loss: `total` must reach every one of those adapters
        // with a nonzero gradient.
        let losses = {
            let generator = fx.generator();
            generator
                .train_losses_with_noise(
                    &client,
                    &st.prefill,
                    &target,
                    &ts,
                    &noise,
                    1.0,
                    1.0,
                    false,
                )
                .expect("train_losses_with_noise")
        };
        let total_grads = backward(&losses.total, &client).expect("backward total");
        let mut nonzero_b = 0usize;
        for (name, id) in &adapters {
            let grad = total_grads.get(*id).unwrap_or_else(|| {
                panic!("adapter {name} has no gradient at all from the combined loss")
            });
            let values: Vec<f32> = grad.contiguous().expect("contiguous").to_vec();
            let any_nonzero = values.iter().any(|v| *v != 0.0);
            if name.ends_with("lora_b") {
                assert!(
                    any_nonzero,
                    "adapter {name} has an all-zero gradient from the combined loss"
                );
                nonzero_b += 1;
            }
            // `lora_a`'s gradient is exactly zero at LoRA init regardless of
            // which loss produced it (dL/dA is proportional to B, and B starts
            // at zero) — see `gradients_reach_every_lora_adapter`'s identical
            // reasoning. Only `lora_b` is asserted nonzero here.
        }
        assert!(nonzero_b > 0, "no stop-chain lora_b adapter was checked");
    }

    /// `lambda_stop = 0.0` must make `total` numerically equal `lambda_diff *
    /// diff` — the same value `cfm_loss_with_noise` returns on its own, since
    /// `lambda_diff = 1.0` here.
    #[test]
    fn lambda_stop_zero_matches_diff_loss_alone() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let generator = fx.generator();
        let st = state(&fx, &device);

        let target = target_patches(0.6, &device);
        let noise = target_patches(1.9, &device);
        let ts = Tensor::<CpuRuntime>::from_slice(&[0.2f32, 0.5, 0.8], &[T], &device).expect("t");

        let diff_alone = generator
            .cfm_loss_with_noise(&client, &st.prefill, &target, &ts, &noise, false)
            .expect("cfm_loss_with_noise");
        let diff_alone_val = diff_alone.tensor().to_vec::<f32>()[0];

        let losses = generator
            .train_losses_with_noise(&client, &st.prefill, &target, &ts, &noise, 1.0, 0.0, false)
            .expect("train_losses_with_noise");
        let total_val = losses.total.tensor().to_vec::<f32>()[0];
        let diff_val = losses.diff.tensor().to_vec::<f32>()[0];

        assert!(
            (total_val - diff_alone_val).abs() < 1e-5,
            "lambda_stop = 0.0 must leave total == diff alone: total={total_val} \
             diff_alone={diff_alone_val}"
        );
        assert!(
            (diff_val - diff_alone_val).abs() < 1e-5,
            "TrainLosses::diff must equal cfm_loss_with_noise's own value: \
             diff={diff_val} diff_alone={diff_alone_val}"
        );
    }
}
