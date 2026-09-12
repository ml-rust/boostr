//! `cfm_loss_with_noise` (caller-supplied `t`/`noise`) and the seeded
//! `cfm_loss` wrapper.

use super::dropout::{apply_cond_dropout, check_training_cfg_rate, draw_drop_cond};
use crate::error::{Error, Result};
use crate::model::audio::voxcpm::model::generate::PatchGenerator;
use crate::model::audio::voxcpm::model::prefill::PrefillState;
use crate::model::traits::ModelClient;
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, RandomOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> PatchGenerator<'_, R> {
    /// One CFM training step's loss, with `t` and `noise` supplied by the
    /// caller — the training-step counterpart of
    /// [`Self::step_with_noise`](crate::model::audio::voxcpm::model::PatchGenerator::step_with_noise):
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
    /// - `drop_cond`: training-time conditioning dropout (the reference
    ///   VoxCPM implementation's `training_cfg_rate`). When true, `cond.mu` is replaced with a zero
    ///   tensor of the same shape/dtype/device BEFORE either loss term is
    ///   computed — the same construction
    ///   [`LocalDit::solve_euler`](crate::model::audio::voxcpm::local_dit::LocalDit::solve_euler)
    ///   uses to build its unconditional half at inference (`sampler`'s
    ///   `mu_zero`: the CFG-doubled batch differs in `mu` ALONE, `cond` is
    ///   duplicated unchanged). Zeroing `mu` some fraction of training steps
    ///   teaches the model to produce a sane, TEXT-INDEPENDENT prediction
    ///   when `mu` is absent, which is what makes classifier-free guidance
    ///   at inference actually work; the reference VoxCPM FAQ calls skipping
    ///   this "the most common fine-tuning failure mode" (text gets ignored) and says
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
    /// [`flow_matching_interpolate`]: crate::nn::flow_matching_interpolate
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

        // Every remaining shape check ([`Self::teacher_forced_conditioning`]'s
        // `patch_size`/`feat_dim` check, `flow_matching_interpolate`'s
        // noise/data match, `LocalDit::forward`'s own checks) is already
        // enforced by the callee — not re-implemented here.
        let cond = self.teacher_forced_conditioning(client, prefill, target_patches)?;
        let cond = apply_cond_dropout(cond, drop_cond)?;
        self.cfm_loss_from_conditioning(client, &cond, target_patches, t, noise, tcount)
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
    /// `training_cfg_rate` is the reference VoxCPM implementation's
    /// `training_cfg_rate` — the per-step
    /// probability of conditioning dropout (see
    /// [`Self::cfm_loss_with_noise`]'s `drop_cond` doc), drawn from
    /// `seed.wrapping_add(2)`: a third stream independent of `t`/`noise`.
    /// The reference implementation defaults this to 0.1 and its FAQ calls 0 "the most common
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
        self.cfm_loss_with_noise(client, prefill, target_patches, &t, &noise, drop_cond)
    }
}

#[cfg(test)]
pub(super) mod tests {
    //! Reuses `generate/tests/support.rs`'s `Fixture` — the exact same tiny
    //! sub-models the teacher-forced tests exercise — rather than building a
    //! second fixture. That module's items are `pub(crate)` specifically so
    //! this sibling of `generate` can reach them directly.

    use super::*;
    use crate::model::audio::voxcpm::local_dit::tests::{FEAT_DIM, PATCH_SIZE, t};
    use crate::model::audio::voxcpm::model::generate::tests::support::{fixture, state};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    /// `T = 3` — enough that the teacher-forced shift actually reads a batched
    /// row (see `teacher_forced.rs`'s module docs), so `base_lm`, `residual_lm`,
    /// `fsq` and `enc_to_lm_proj`/`fusion_concat_proj` all sit on a live
    /// gradient path instead of computing an output nothing downstream reads.
    pub(in super::super) const T: usize = 3;

    pub(in super::super) fn target_patches(seed: f32, device: &CpuDevice) -> Tensor<CpuRuntime> {
        t(&[T, PATCH_SIZE, FEAT_DIM], seed, device)
    }

    #[test]
    fn cfm_loss_with_noise_is_finite_and_positive() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let generator = fx.generator();
        let st = state(&fx, &device);

        let target = target_patches(0.4, &device);
        let noise = target_patches(1.7, &device);
        let ts = Tensor::<CpuRuntime>::from_slice(&[0.2f32, 0.5, 0.8], &[T], &device).expect("t");

        let loss = generator
            .cfm_loss_with_noise(&client, &st.prefill, &target, &ts, &noise, false)
            .expect("cfm_loss_with_noise");
        let val = loss.tensor().to_vec::<f32>()[0];
        assert!(val.is_finite(), "loss must be finite, got {val}");
        assert!(
            val > 0.0,
            "loss must be positive for mismatched prediction/target, got {val}"
        );
    }

    #[test]
    fn cfm_loss_seeded_wrapper_is_finite_and_positive() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let generator = fx.generator();
        let st = state(&fx, &device);

        let target = target_patches(2.3, &device);
        let loss = generator
            .cfm_loss(&client, &st.prefill, &target, 42, 0.0)
            .expect("cfm_loss");
        let val = loss.tensor().to_vec::<f32>()[0];
        assert!(val.is_finite(), "loss must be finite, got {val}");
        assert!(val > 0.0, "loss must be positive, got {val}");
    }

    #[test]
    fn cfm_loss_with_noise_rejects_bad_shapes() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let generator = fx.generator();
        let st = state(&fx, &device);

        let target = target_patches(0.6, &device);
        let noise = target_patches(1.9, &device);
        let ts = Tensor::<CpuRuntime>::from_slice(&[0.2f32, 0.5, 0.8], &[T], &device).expect("t");

        let bad_target =
            Tensor::<CpuRuntime>::zeros(&[T, PATCH_SIZE + 1, FEAT_DIM], DType::F32, &device)
                .expect("zeros");
        assert!(
            generator
                .cfm_loss_with_noise(&client, &st.prefill, &bad_target, &ts, &noise, false)
                .is_err(),
            "a shape-mismatched target_patches must error, not panic"
        );

        let bad_t = Tensor::<CpuRuntime>::zeros(&[T + 1], DType::F32, &device).expect("zeros");
        assert!(
            generator
                .cfm_loss_with_noise(&client, &st.prefill, &target, &bad_t, &noise, false)
                .is_err(),
            "a shape-mismatched t must error, not panic"
        );

        let bad_noise =
            Tensor::<CpuRuntime>::zeros(&[T, PATCH_SIZE, FEAT_DIM + 1], DType::F32, &device)
                .expect("zeros");
        assert!(
            generator
                .cfm_loss_with_noise(&client, &st.prefill, &target, &ts, &bad_noise, false)
                .is_err(),
            "a shape-mismatched noise must error, not panic"
        );

        let empty = Tensor::<CpuRuntime>::zeros(&[0, PATCH_SIZE, FEAT_DIM], DType::F32, &device)
            .expect("zeros");
        let empty_t = Tensor::<CpuRuntime>::zeros(&[0], DType::F32, &device).expect("zeros");
        let empty_noise =
            Tensor::<CpuRuntime>::zeros(&[0, PATCH_SIZE, FEAT_DIM], DType::F32, &device)
                .expect("zeros");
        assert!(
            generator
                .cfm_loss_with_noise(&client, &st.prefill, &empty, &empty_t, &empty_noise, false)
                .is_err(),
            "T = 0 must error, not panic"
        );
    }

    #[test]
    fn cfm_loss_seeded_wrapper_rejects_bad_target_shape() {
        let (client, device) = cpu_setup();
        let fx = fixture(false, &device);
        let generator = fx.generator();
        let st = state(&fx, &device);

        let bad = Tensor::<CpuRuntime>::zeros(&[T, PATCH_SIZE, FEAT_DIM], DType::F32, &device)
            .expect("zeros")
            .reshape(&[T * PATCH_SIZE, FEAT_DIM])
            .expect("reshape to wrong rank");
        assert!(
            generator
                .cfm_loss(&client, &st.prefill, &bad, 7, 0.0)
                .is_err(),
            "a rank-2 target_patches must error, not panic"
        );
    }
}
