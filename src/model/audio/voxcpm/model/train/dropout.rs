//! Training-time conditioning dropout (the reference VoxCPM implementation's
//! `training_cfg_rate`): the zeroing itself, the seeded per-step draw, and
//! the rate validation.

use crate::error::{Error, Result};
use crate::model::audio::voxcpm::model::generate::TeacherForcedConditioning;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::RandomOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Training-time conditioning dropout (the reference VoxCPM implementation's
/// `training_cfg_rate`):
/// when `drop_cond`, replace `cond.mu` with a zero tensor of the same
/// shape/dtype/device, everything else passed through unchanged. Shared so
/// [`PatchGenerator::train_losses_with_noise`] applies it once, upstream
/// of BOTH the diff and stop terms, instead of each computing its own copy.
/// See [`PatchGenerator::cfm_loss_with_noise`]'s `drop_cond` doc for why
/// `mu` alone is the right tensor to zero.
///
/// [`PatchGenerator::train_losses_with_noise`]: crate::model::audio::voxcpm::model::PatchGenerator::train_losses_with_noise
/// [`PatchGenerator::cfm_loss_with_noise`]: crate::model::audio::voxcpm::model::PatchGenerator::cfm_loss_with_noise
pub(super) fn apply_cond_dropout<R: Runtime<DType = DType>>(
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
pub(super) fn draw_drop_cond<C, R>(client: &C, seed: u64, rate: f64) -> Result<bool>
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
/// accepted but discouraged (the reference VoxCPM implementation's default is
/// 0.1).
///
/// [`PatchGenerator::cfm_loss`]: crate::model::audio::voxcpm::model::PatchGenerator::cfm_loss
pub(super) fn check_training_cfg_rate(rate: f64) -> Result<()> {
    if !(0.0..=1.0).contains(&rate) {
        return Err(Error::InvalidArgument {
            arg: "training_cfg_rate",
            reason: format!("expected a value in [0.0, 1.0], got {rate}"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::cfm::tests::{T, target_patches};
    use crate::model::audio::voxcpm::model::generate::tests::support::{fixture, state};
    use crate::test_utils::cpu_setup;
    use numr::ops::RandomOps;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    /// `drop_cond = true` must produce a finite loss that DIFFERS from
    /// `drop_cond = false` on the identical `t`/`noise` — otherwise the
    /// dropout parameter is wired in but never actually changes anything.
    #[test]
    fn drop_cond_true_changes_the_diff_loss() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let generator = fx.generator();
        let st = state(&fx, &device);

        let target = target_patches(0.6, &device);
        let noise = target_patches(1.9, &device);
        let ts = Tensor::<CpuRuntime>::from_slice(&[0.2f32, 0.5, 0.8], &[T], &device).expect("t");

        let loss_kept = generator
            .cfm_loss_with_noise(&client, &st.prefill, &target, &ts, &noise, false)
            .expect("cfm_loss_with_noise drop_cond=false");
        let loss_dropped = generator
            .cfm_loss_with_noise(&client, &st.prefill, &target, &ts, &noise, true)
            .expect("cfm_loss_with_noise drop_cond=true");

        let val_kept = loss_kept.tensor().to_vec::<f32>()[0];
        let val_dropped = loss_dropped.tensor().to_vec::<f32>()[0];
        assert!(
            val_dropped.is_finite(),
            "drop_cond=true loss must be finite, got {val_dropped}"
        );
        assert!(
            (val_dropped - val_kept).abs() > 1e-6,
            "drop_cond=true must change the diff loss: kept={val_kept} dropped={val_dropped}"
        );
    }

    /// `drop_cond = false` must be bit-identical to the pre-dropout behaviour:
    /// [`PatchGenerator::teacher_forced_conditioning`] followed by
    /// [`PatchGenerator::cfm_loss_from_conditioning`] directly, with no dropout
    /// applied anywhere. Pins that the default path is untouched by this unit.
    ///
    /// [`PatchGenerator::teacher_forced_conditioning`]: crate::model::audio::voxcpm::model::PatchGenerator::teacher_forced_conditioning
    /// [`PatchGenerator::cfm_loss_from_conditioning`]: crate::model::audio::voxcpm::model::PatchGenerator::cfm_loss_from_conditioning
    #[test]
    fn drop_cond_false_matches_conditioning_computed_directly() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let generator = fx.generator();
        let st = state(&fx, &device);

        let target = target_patches(0.6, &device);
        let noise = target_patches(1.9, &device);
        let ts = Tensor::<CpuRuntime>::from_slice(&[0.2f32, 0.5, 0.8], &[T], &device).expect("t");

        let via_wrapper = generator
            .cfm_loss_with_noise(&client, &st.prefill, &target, &ts, &noise, false)
            .expect("cfm_loss_with_noise");

        let cond = generator
            .teacher_forced_conditioning(&client, &st.prefill, &target)
            .expect("teacher_forced_conditioning");
        let direct = generator
            .cfm_loss_from_conditioning(&client, &cond, &target, &ts, &noise, T)
            .expect("cfm_loss_from_conditioning");

        let val_wrapper = via_wrapper.tensor().to_vec::<f32>()[0];
        let val_direct = direct.tensor().to_vec::<f32>()[0];
        assert_eq!(
            val_wrapper.to_bits(),
            val_direct.to_bits(),
            "drop_cond=false must be bit-identical to the undropped direct computation: \
             wrapper={val_wrapper} direct={val_direct}"
        );
    }

    /// The stop loss reads `cond.lm_hidden`, not `cond.mu`, so `drop_cond` must
    /// leave it numerically IDENTICAL — the dropout must not leak into the stop
    /// term.
    #[test]
    fn drop_cond_does_not_change_the_stop_loss() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let generator = fx.generator();
        let st = state(&fx, &device);

        let target = target_patches(0.6, &device);
        let noise = target_patches(1.9, &device);
        let ts = Tensor::<CpuRuntime>::from_slice(&[0.2f32, 0.5, 0.8], &[T], &device).expect("t");

        let kept = generator
            .train_losses_with_noise(&client, &st.prefill, &target, &ts, &noise, 1.0, 1.0, false)
            .expect("train_losses_with_noise drop_cond=false");
        let dropped = generator
            .train_losses_with_noise(&client, &st.prefill, &target, &ts, &noise, 1.0, 1.0, true)
            .expect("train_losses_with_noise drop_cond=true");

        let stop_kept = kept.stop.tensor().to_vec::<f32>()[0];
        let stop_dropped = dropped.stop.tensor().to_vec::<f32>()[0];
        assert_eq!(
            stop_kept.to_bits(),
            stop_dropped.to_bits(),
            "loss/stop must be identical regardless of drop_cond: kept={stop_kept} \
             dropped={stop_dropped}"
        );
    }

    /// `training_cfg_rate = 0.0` must never drop and `1.0` must always drop,
    /// checked through the seeded wrapper across several seeds by comparing
    /// against the deterministic form with the matching `t`/`noise` draws and
    /// an explicit `drop_cond`.
    #[test]
    fn training_cfg_rate_boundaries_are_deterministic() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let generator = fx.generator();
        let st = state(&fx, &device);

        let target = target_patches(0.6, &device);
        let dtype = target.dtype();
        let shape = target.shape();

        for seed in [1u64, 42, 100, 9_999] {
            let t_draw = client
                .rand_seeded(&[T], dtype, seed)
                .expect("rand_seeded t");
            let noise_draw = client
                .randn_seeded(shape, dtype, seed.wrapping_add(1))
                .expect("randn_seeded noise");

            let never_dropped = generator
                .cfm_loss(&client, &st.prefill, &target, seed, 0.0)
                .expect("cfm_loss rate=0.0");
            let never_dropped_direct = generator
                .cfm_loss_with_noise(&client, &st.prefill, &target, &t_draw, &noise_draw, false)
                .expect("cfm_loss_with_noise drop_cond=false");
            assert_eq!(
                never_dropped.tensor().to_vec::<f32>()[0].to_bits(),
                never_dropped_direct.tensor().to_vec::<f32>()[0].to_bits(),
                "training_cfg_rate=0.0 must never drop (seed={seed})"
            );

            let always_dropped = generator
                .cfm_loss(&client, &st.prefill, &target, seed, 1.0)
                .expect("cfm_loss rate=1.0");
            let always_dropped_direct = generator
                .cfm_loss_with_noise(&client, &st.prefill, &target, &t_draw, &noise_draw, true)
                .expect("cfm_loss_with_noise drop_cond=true");
            assert_eq!(
                always_dropped.tensor().to_vec::<f32>()[0].to_bits(),
                always_dropped_direct.tensor().to_vec::<f32>()[0].to_bits(),
                "training_cfg_rate=1.0 must always drop (seed={seed})"
            );
        }
    }

    /// An out-of-range `training_cfg_rate` must error, not panic — from both
    /// [`PatchGenerator::cfm_loss`] and [`PatchGenerator::train_losses`].
    ///
    /// [`PatchGenerator::cfm_loss`]: crate::model::audio::voxcpm::model::PatchGenerator::cfm_loss
    /// [`PatchGenerator::train_losses`]: crate::model::audio::voxcpm::model::PatchGenerator::train_losses
    #[test]
    fn training_cfg_rate_out_of_range_errors() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let generator = fx.generator();
        let st = state(&fx, &device);

        let target = target_patches(0.6, &device);

        for bad_rate in [-0.1, 1.5] {
            assert!(
                generator
                    .cfm_loss(&client, &st.prefill, &target, 1, bad_rate)
                    .is_err(),
                "cfm_loss must reject training_cfg_rate={bad_rate}, not panic"
            );
            assert!(
                generator
                    .train_losses(&client, &st.prefill, &target, 1, 1.0, 1.0, bad_rate)
                    .is_err(),
                "train_losses must reject training_cfg_rate={bad_rate}, not panic"
            );
        }
    }
}
