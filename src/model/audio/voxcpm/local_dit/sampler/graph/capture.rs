//! Capture and replay of the post-warmup Euler loop on `CudaRuntime`.
//!
//! Capture region: from `mu_in`/`cond_in`/`mu_tok`/`cond_h` assembly through
//! every non-warmup step to one D2D copy of the final `x` into `x_out_buf`. The
//! warmup steps touch nothing on the device (`x == z`), so the graph starts
//! straight from `z_buf`. Every op inside the region is the SAME numr op the
//! eager loop issues, in the same order, so replay is bit-identical to
//! `LocalDit::solve_euler`.
//!
//! Replay: three stream-ordered D2D copies (`z`, `mu`, `cond` into the
//! entry's stable buffers), one graph launch, one D2D copy out of
//! `x_out_buf` into a fresh tensor. Nothing here synchronizes the host.

use numr::autograd::{Var, var_cat, var_mul_scalar, var_narrow, var_reshape, var_sub};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use super::cache::{CapturedEuler, EulerGraphKey};
use crate::error::{Error, Result};
use crate::model::audio::voxcpm::local_dit::loader::LocalDit;
use crate::model::audio::voxcpm::local_dit::sampler::guidance::{cfg_combine, optimized_scale};
use crate::model::audio::voxcpm::local_dit::sampler::schedule::{EulerStep, euler_steps};
use crate::nn::var_contiguous;

/// Integrate `z` through the cached graph for `key`, capturing it on a miss.
///
/// `z`, `mu` and `cond` are the caller's tensors; their shapes and dtype
/// already match `key` (the generic entry derived the key from them).
/// Returns a FRESH `[batch, patch_size, feat_dim]` tensor: the graph's own
/// output buffer is overwritten by the next launch, so it never escapes.
pub(in crate::model::audio::voxcpm::local_dit::sampler) fn solve_euler_cuda(
    dit: &LocalDit<CudaRuntime>,
    client: &CudaClient,
    key: &EulerGraphKey,
    z: &Tensor<CudaRuntime>,
    t_span: &[f32],
    mu: &Tensor<CudaRuntime>,
    cond: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    // One lock spans capture and replay: see `EulerGraphCache` for why.
    dit.euler_graphs.with_entry_or_capture(
        key,
        || capture(dit, client, key, t_span),
        |entry| replay(entry, client, z, mu, cond),
    )
}

/// Bytes of a contiguous tensor's elements, for the D2D copies.
fn byte_len(t: &Tensor<CudaRuntime>) -> usize {
    t.numel() * t.dtype().size_in_bytes()
}

/// Stream-ordered D2D copy of `src` (made contiguous) into `dst`.
///
/// Returns the contiguous source so the caller can keep it alive until the
/// launch that follows has been enqueued: a materialized temporary freed
/// earlier could be handed out again by the allocator before the copy runs.
fn copy_into(
    src: &Tensor<CudaRuntime>,
    dst: &Tensor<CudaRuntime>,
    what: &'static str,
) -> Result<Tensor<CudaRuntime>> {
    if src.shape() != dst.shape() || src.dtype() != dst.dtype() {
        return Err(Error::InvalidArgument {
            arg: what,
            reason: format!(
                "graph buffer is {:?} {}, got {:?} {}; the cache key must cover this input",
                dst.shape(),
                dst.dtype(),
                src.shape(),
                src.dtype()
            ),
        });
    }
    let src = src.contiguous().map_err(Error::Numr)?;
    CudaRuntime::copy_within_device(src.ptr(), dst.ptr(), byte_len(&src), dst.device())
        .map_err(Error::Numr)?;
    Ok(src)
}

fn replay(
    entry: &CapturedEuler,
    client: &CudaClient,
    z: &Tensor<CudaRuntime>,
    mu: &Tensor<CudaRuntime>,
    cond: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    // Held until after `launch` is enqueued; see `copy_into`.
    let _z = copy_into(z, entry.z_buf(), "z")?;
    let _mu = copy_into(mu, entry.mu_buf(), "mu")?;
    let _cond = copy_into(cond, entry.cond_buf(), "cond")?;

    // Stream order guarantees the three copies land before the graph reads.
    entry.launch().map_err(Error::Numr)?;

    // Copy out rather than alias: the caller may hold this result across the
    // next patch's launch, which rewrites `x_out_buf`.
    let x_out = entry.x_out_buf();
    let out = Tensor::<CudaRuntime>::empty(x_out.shape(), x_out.dtype(), client.device())
        .map_err(Error::Numr)?;
    CudaRuntime::copy_within_device(x_out.ptr(), out.ptr(), byte_len(x_out), client.device())
        .map_err(Error::Numr)?;
    Ok(out)
}

/// Lift a boostr error into the numr error the capture closure must return.
fn into_numr(e: Error) -> numr::error::Error {
    numr::error::Error::Backend(format!("VoxCPM2 Euler graph capture: {e:#}"))
}

fn capture(
    dit: &LocalDit<CudaRuntime>,
    client: &CudaClient,
    key: &EulerGraphKey,
    t_span: &[f32],
) -> Result<CapturedEuler> {
    let device = client.device();
    let dtype = key.dtype;
    let batch = key.batch;
    let patch_shape = [batch, dit.patch_size(), dit.feat_dim()];
    let mu_shape = [batch, key.mu_tokens * dit.hidden_dim()];

    let steps: Vec<EulerStep> = euler_steps(t_span, key.use_cfg_zero_star)
        .into_iter()
        .flatten()
        .collect();

    // Every buffer below is allocated BEFORE capture, for two reasons:
    //
    // - Addresses. The graph bakes device pointers in. A buffer allocated
    //   inside the region is a graph-owned allocation, freed on every launch,
    //   so replay would read and write freed memory.
    // - Host copies. `zeros`/`full_scalar` upload from a host temporary.
    //   Inside the region that records a memcpy node whose source is a
    //   stack address that no longer exists at replay.
    //
    // The three inputs are `empty`: replay fills them before each launch.
    let z_buf = Tensor::<CudaRuntime>::empty(&patch_shape, dtype, device).map_err(Error::Numr)?;
    let mu_buf = Tensor::<CudaRuntime>::empty(&mu_shape, dtype, device).map_err(Error::Numr)?;
    let cond_buf =
        Tensor::<CudaRuntime>::empty(&patch_shape, dtype, device).map_err(Error::Numr)?;
    let x_out_buf =
        Tensor::<CudaRuntime>::empty(&patch_shape, dtype, device).map_err(Error::Numr)?;
    // Constants: the unconditional half's zero `mu`, the estimator's zero
    // mean-velocity `dt`, and one `t` scalar per captured step (its value is
    // host-known from the schedule, so it is baked, never uploaded per patch).
    let mu_zero = Tensor::<CudaRuntime>::zeros(&mu_shape, dtype, device).map_err(Error::Numr)?;
    let dt_in = Tensor::<CudaRuntime>::zeros(&[2 * batch], dtype, device).map_err(Error::Numr)?;
    let t_ins = steps
        .iter()
        .map(|s| Tensor::<CudaRuntime>::full_scalar(&[2 * batch], dtype, s.t as f64, device))
        .collect::<numr::error::Result<Vec<_>>>()
        .map_err(Error::Numr)?;

    // Everything the graph reads goes in `inputs`: `CapturedGraph` clones
    // them and that is what keeps the allocations alive with the graph.
    let mut inputs: Vec<&Tensor<CudaRuntime>> = vec![&z_buf, &mu_buf, &cond_buf, &mu_zero, &dt_in];
    inputs.extend(t_ins.iter());

    let x_out_ptr = x_out_buf.ptr();
    let x_out_bytes = byte_len(&x_out_buf);
    let cfg_value = f32::from_bits(key.cfg_bits);

    let captured = CudaRuntime::capture_graph_into(client, &inputs, &[&x_out_buf], |cc| {
        let z = Var::new(z_buf.clone(), false);
        let mu = Var::new(mu_buf.clone(), false);
        let cond = Var::new(cond_buf.clone(), false);
        let mu_zero = Var::new(mu_zero.clone(), false);
        let dt_in = Var::new(dt_in.clone(), false);

        // Same assembly as the eager loop: conditional `mu` on top of zero
        // `mu`, `cond` duplicated, `mu` tokenized and `cond` projected once
        // for every step.
        let mu_in = var_cat(&[&mu, &mu_zero], 0, cc)?;
        let cond_in = var_cat(&[&cond, &cond], 0, cc)?;
        let mu_tok = var_reshape(
            &var_contiguous(&mu_in).map_err(into_numr)?,
            &[2 * batch, key.mu_tokens, dit.hidden_dim()],
        )?;
        let cond_h = dit.project_cond(cc, &cond_in).map_err(into_numr)?;

        let mut x = z;
        for (step, t_in) in steps.iter().zip(&t_ins) {
            let x_in = var_cat(&[&x, &x], 0, cc)?;
            let t_in = Var::new(t_in.clone(), false);
            let out = dit
                .forward_prepared(cc, &x_in, &mu_tok, &t_in, &cond_h, &dt_in)
                .map_err(into_numr)?;

            let v_cond = var_narrow(&out, 0, 0, batch)?;
            let v_uncond = var_narrow(&out, 0, batch, batch)?;
            let st_star = optimized_scale(cc, &v_cond, &v_uncond).map_err(into_numr)?;
            let velocity =
                cfg_combine(cc, &v_cond, &v_uncond, &st_star, cfg_value).map_err(into_numr)?;
            let move_by = var_mul_scalar(&velocity, step.dt as f64, cc)?;
            x = var_sub(&x, &move_by, cc)?;
        }

        // The final `x` is a graph-owned intermediate. Park it in the stable
        // output buffer so replay can read it after the launch.
        let x = x.tensor().contiguous()?;
        CudaRuntime::copy_within_device(x.ptr(), x_out_ptr, x_out_bytes, device)
    })
    .map_err(Error::Numr)?;

    Ok(CapturedEuler::new(captured))
}

#[cfg(test)]
mod tests {
    //! Graph vs eager on the tiny fixture, on a real device. Skips when the
    //! `cuda` feature is on but no device is present.

    use super::*;
    use crate::model::audio::voxcpm::local_dit::sampler::cfm_time_span;
    use crate::model::audio::voxcpm::local_dit::tests as fixture;
    use crate::test_utils::{CudaTest, cuda_setup};
    use numr::runtime::cuda::CudaDevice;

    /// Skips without a device. The returned guard serializes this test
    /// against every other in-process CUDA test; a capture on the shared
    /// stream would otherwise record their work too.
    fn cuda_client() -> Option<(CudaClient, CudaDevice, CudaTest)> {
        let cuda = cuda_setup()?;
        Some((cuda.client.clone(), cuda.device.clone(), cuda))
    }

    fn var(shape: &[usize], seed: f32, device: &CudaDevice) -> Var<CudaRuntime> {
        Var::new(fixture::t_on::<CudaRuntime>(shape, seed, device), false)
    }

    fn bits(v: &Var<CudaRuntime>) -> Vec<u32> {
        let host: Vec<f32> = v.tensor().contiguous().unwrap().to_vec();
        host.iter().map(|x| x.to_bits()).collect()
    }

    struct Inputs {
        z: Var<CudaRuntime>,
        mu: Var<CudaRuntime>,
        cond: Var<CudaRuntime>,
    }

    fn inputs(batch: usize, z_seed: f32, cond_seed: f32, device: &CudaDevice) -> Inputs {
        Inputs {
            z: var(
                &[batch, fixture::PATCH_SIZE, fixture::FEAT_DIM],
                z_seed,
                device,
            ),
            mu: var(
                &[batch, fixture::MU_TOKENS * fixture::HIDDEN_DIM],
                1.3,
                device,
            ),
            cond: var(
                &[batch, fixture::PATCH_SIZE, fixture::FEAT_DIM],
                cond_seed,
                device,
            ),
        }
    }

    /// Two calls with different `z`/`cond`: the first captures, the second
    /// replays the same graph with fresh buffer contents. Both must equal the
    /// eager loop bit for bit, and the second result must differ from the
    /// first (otherwise a stale buffer would pass).
    #[test]
    fn graph_matches_eager_bit_for_bit_and_replays_from_cache() {
        let Some((client, device, _serial)) = cuda_client() else {
            return;
        };
        let m = fixture::model_on::<CudaRuntime>(1, &device);
        let span = cfm_time_span(4, 1.0).unwrap();
        assert_eq!(m.euler_graph_capture_count(), 0);

        let mut results = Vec::new();
        for (call, (z_seed, cond_seed)) in [(0.9, 1.7), (2.1, 0.3)].into_iter().enumerate() {
            let i = inputs(2, z_seed, cond_seed, &device);
            let eager = m
                .solve_euler(&client, &i.z, &span, &i.mu, &i.cond, 2.0, true, None)
                .unwrap();
            let graphed = m
                .solve_euler_graphed(&client, &i.z, &span, &i.mu, &i.cond, 2.0, true, None)
                .unwrap();
            assert_eq!(graphed.shape(), i.z.shape());
            assert_eq!(
                bits(&eager),
                bits(&graphed),
                "call {call} diverged from eager"
            );
            assert_eq!(
                m.euler_graph_capture_count(),
                1,
                "call {call} must reuse the graph"
            );
            results.push(bits(&graphed));
        }
        assert_ne!(
            results[0], results[1],
            "second call must see its own inputs"
        );
    }

    /// A different schedule or guidance weight is a different graph, even at
    /// the same shapes: both are baked in.
    #[test]
    fn schedule_and_guidance_are_part_of_the_key() {
        let Some((client, device, _serial)) = cuda_client() else {
            return;
        };
        let m = fixture::model_on::<CudaRuntime>(1, &device);
        let i = inputs(1, 0.9, 1.7, &device);
        let span_a = cfm_time_span(3, 1.0).unwrap();
        let span_b = cfm_time_span(3, 0.5).unwrap();

        m.solve_euler_graphed(&client, &i.z, &span_a, &i.mu, &i.cond, 2.0, true, None)
            .unwrap();
        m.solve_euler_graphed(&client, &i.z, &span_b, &i.mu, &i.cond, 2.0, true, None)
            .unwrap();
        m.solve_euler_graphed(&client, &i.z, &span_b, &i.mu, &i.cond, 1.5, true, None)
            .unwrap();
        assert_eq!(m.euler_graph_capture_count(), 3);
    }

    /// `trajectory` and `requires_grad` both route to eager on a CUDA
    /// runtime: nothing is captured, and the trajectory is fully recorded.
    #[test]
    fn trajectory_and_grad_requests_stay_eager_on_cuda() {
        let Some((client, device, _serial)) = cuda_client() else {
            return;
        };
        let m = fixture::model_on::<CudaRuntime>(1, &device);
        let i = inputs(1, 0.9, 1.7, &device);
        let span = cfm_time_span(3, 1.0).unwrap();

        let mut trace = Vec::new();
        m.solve_euler_graphed(
            &client,
            &i.z,
            &span,
            &i.mu,
            &i.cond,
            2.0,
            true,
            Some(&mut trace),
        )
        .unwrap();
        assert_eq!(trace.len(), 3);
        assert_eq!(m.euler_graph_capture_count(), 0);

        let z_grad = Var::new(i.z.tensor().clone(), true);
        m.solve_euler_graphed(&client, &z_grad, &span, &i.mu, &i.cond, 2.0, true, None)
            .unwrap();
        assert_eq!(m.euler_graph_capture_count(), 0);
    }
}
