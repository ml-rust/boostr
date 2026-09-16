//! `LocalDit::solve_euler_graphed`: the inference entry that replays the
//! Euler loop as one CUDA graph per patch where it can, and runs the eager
//! [`LocalDit::solve_euler`] loop everywhere else.
//!
//! Eager is chosen when any of these holds:
//!
//! - the runtime is not CUDA (or the crate is built without `cuda`)
//! - `trajectory` is requested: the graph exposes only the final `x`
//! - an input carries `requires_grad`: the graph records no autograd tape
//! - the schedule has no post-warmup step: nothing to capture
//! - capture or replay fails: logged once per process, then eager

use super::schedule::euler_steps;
use crate::error::{Error, Result};
use crate::model::audio::voxcpm::local_dit::loader::LocalDit;
use crate::model::traits::ModelClient;
use crate::ops::FlashAttentionOps;
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> LocalDit<R> {
    /// [`solve_euler`](Self::solve_euler) with CUDA graph replay.
    ///
    /// Same arguments and same result, bit for bit. On a CUDA runtime the
    /// post-warmup loop is captured once per distinct
    /// `(batch, schedule, cfg_value, use_cfg_zero_star, dtype, mu width)`
    /// and replayed on every later call. See the module doc for when the
    /// eager loop runs instead.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_euler_graphed<C>(
        &self,
        client: &C,
        z: &Var<R>,
        t_span: &[f32],
        mu: &Var<R>,
        cond: &Var<R>,
        cfg_value: f32,
        use_cfg_zero_star: bool,
        trajectory: Option<&mut Vec<Var<R>>>,
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
            + DequantOps<R>
            + FlashAttentionOps<R>,
    {
        // Validated here so a bad input errors the same way on both paths.
        let batch = self.check_patch_input("z", z, None)?;
        self.check_patch_input("cond", cond, Some(batch))?;
        let mu_tokens = self.check_mu(mu, batch)?;
        if t_span.len() < 2 {
            return Err(Error::InvalidArgument {
                arg: "t_span",
                reason: format!("expected at least 2 entries, got {}", t_span.len()),
            });
        }

        let eager_only = trajectory.is_some()
            || z.requires_grad()
            || mu.requires_grad()
            || cond.requires_grad()
            || euler_steps(t_span, use_cfg_zero_star)
                .iter()
                .all(Option::is_none);

        #[cfg(feature = "cuda")]
        if !eager_only {
            let key = super::graph::EulerGraphKey {
                batch,
                patch_size: self.patch_size,
                feat_dim: self.feat_dim,
                mu_tokens,
                schedule: t_span.iter().map(|t| t.to_bits()).collect(),
                cfg_bits: cfg_value.to_bits(),
                use_cfg_zero_star,
                dtype: z.tensor().dtype(),
            };
            if let Some(out) = cuda::try_solve(self, client, &key, z, t_span, mu, cond) {
                return Ok(out);
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = (eager_only, mu_tokens);

        self.solve_euler(
            client,
            z,
            t_span,
            mu,
            cond,
            cfg_value,
            use_cfg_zero_star,
            trajectory,
        )
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    //! The `R == CudaRuntime` dispatch. `Runtime: 'static`, so `TypeId`
    //! settles the runtime exactly; the pointer casts below are then between
    //! one concrete type and itself. `C: 'static` likewise lets the client
    //! be downcast instead of cast, so a foreign client on a CUDA runtime
    //! falls back to eager instead of being reinterpreted.

    use std::any::{Any, TypeId};
    use std::sync::atomic::{AtomicBool, Ordering};

    use numr::autograd::Var;
    use numr::dtype::DType;
    use numr::runtime::Runtime;
    use numr::runtime::cuda::{CudaClient, CudaRuntime};
    use numr::tensor::Tensor;

    use crate::model::audio::voxcpm::local_dit::loader::LocalDit;
    use crate::model::audio::voxcpm::local_dit::sampler::graph::{EulerGraphKey, solve_euler_cuda};

    /// Set by the first capture/replay error; later errors stay silent so a
    /// long render does not log once per patch.
    static WARNED: AtomicBool = AtomicBool::new(false);

    /// `Some(x)` when the graph path ran; `None` means "use eager".
    pub(super) fn try_solve<R: Runtime<DType = DType>, C: 'static>(
        dit: &LocalDit<R>,
        client: &C,
        key: &EulerGraphKey,
        z: &Var<R>,
        t_span: &[f32],
        mu: &Var<R>,
        cond: &Var<R>,
    ) -> Option<Var<R>> {
        if TypeId::of::<R>() != TypeId::of::<CudaRuntime>() {
            return None;
        }
        let client = (client as &dyn Any).downcast_ref::<CudaClient>()?;

        // SAFETY: `R == CudaRuntime` was established above, so each cast
        // reinterprets a reference to `T<CudaRuntime>` as itself.
        let dit = unsafe { &*(dit as *const LocalDit<R>).cast::<LocalDit<CudaRuntime>>() };
        fn as_cuda<R: Runtime>(v: &Var<R>) -> &Tensor<CudaRuntime> {
            // SAFETY: only called after the `TypeId` check in `try_solve`.
            unsafe { &*(v.tensor() as *const Tensor<R>).cast::<Tensor<CudaRuntime>>() }
        }

        let out = solve_euler_cuda(
            dit,
            client,
            key,
            as_cuda(z),
            t_span,
            as_cuda(mu),
            as_cuda(cond),
        );
        match out {
            Ok(t) => {
                // SAFETY: same type identity as above; `read` + `forget`
                // moves the value across the identical generic instance.
                let t = std::mem::ManuallyDrop::new(t);
                let t: Tensor<R> =
                    unsafe { std::ptr::read((&*t as *const Tensor<CudaRuntime>).cast()) };
                Some(Var::new(t, false))
            }
            Err(e) => {
                if !WARNED.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        error = %e,
                        "VoxCPM2 Euler CUDA graph unavailable; using the eager loop"
                    );
                }
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! The non-CUDA contract: on a CPU runtime `solve_euler_graphed` IS the
    //! eager loop, trajectory included, and nothing is ever captured.

    use super::super::guidance::tests::values;
    use super::*;
    use crate::model::audio::voxcpm::local_dit::sampler::cfm_time_span;
    use crate::model::audio::voxcpm::local_dit::tests as fixture;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn var(shape: &[usize], seed: f32, device: &CpuDevice) -> Var<CpuRuntime> {
        Var::new(fixture::t(shape, seed, device), false)
    }

    #[test]
    fn cpu_runtime_returns_the_eager_result() {
        let (client, device) = cpu_setup();
        let m = fixture::model(1, &device);
        let span = cfm_time_span(4, 1.0).unwrap();
        let z = var(&[2, fixture::PATCH_SIZE, fixture::FEAT_DIM], 0.9, &device);
        let mu = var(&[2, fixture::MU_TOKENS * fixture::HIDDEN_DIM], 1.3, &device);
        let cond = var(&[2, fixture::PATCH_SIZE, fixture::FEAT_DIM], 1.7, &device);

        let eager = m
            .solve_euler(&client, &z, &span, &mu, &cond, 2.0, true, None)
            .unwrap();
        let graphed = m
            .solve_euler_graphed(&client, &z, &span, &mu, &cond, 2.0, true, None)
            .unwrap();

        let eager: Vec<u32> = values(&eager).iter().map(|x| x.to_bits()).collect();
        let graphed: Vec<u32> = values(&graphed).iter().map(|x| x.to_bits()).collect();
        assert_eq!(eager, graphed);
        assert_eq!(m.euler_graph_capture_count(), 0);
    }

    #[test]
    fn trajectory_request_records_every_step() {
        let (client, device) = cpu_setup();
        let m = fixture::model(1, &device);
        let span = cfm_time_span(4, 1.0).unwrap();
        let z = var(&[1, fixture::PATCH_SIZE, fixture::FEAT_DIM], 0.9, &device);
        let mu = var(&[1, fixture::MU_TOKENS * fixture::HIDDEN_DIM], 1.3, &device);
        let cond = var(&[1, fixture::PATCH_SIZE, fixture::FEAT_DIM], 1.7, &device);

        let mut trace = Vec::new();
        let out = m
            .solve_euler_graphed(&client, &z, &span, &mu, &cond, 2.0, true, Some(&mut trace))
            .unwrap();
        assert_eq!(trace.len(), 4);
        assert_eq!(values(&trace[3]), values(&out));
    }

    #[test]
    fn graphed_rejects_a_one_entry_schedule() {
        let (client, device) = cpu_setup();
        let m = fixture::model(1, &device);
        let z = var(&[1, fixture::PATCH_SIZE, fixture::FEAT_DIM], 0.9, &device);
        let mu = var(&[1, fixture::MU_TOKENS * fixture::HIDDEN_DIM], 1.3, &device);
        let cond = var(&[1, fixture::PATCH_SIZE, fixture::FEAT_DIM], 1.7, &device);
        assert!(
            m.solve_euler_graphed(&client, &z, &[1.0], &mu, &cond, 2.0, true, None)
                .is_err()
        );
    }
}
