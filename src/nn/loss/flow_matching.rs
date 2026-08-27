//! Conditional flow-matching (CFM) training objective.
//!
//! Generic and model-agnostic: this file knows nothing about VoxCPM2, the
//! local DiT, or any other model. It only defines the probability path, the
//! velocity target, and the loss that trains a velocity estimator to match
//! that target.
//!
//! # The sign convention is PINNED, not derived here
//!
//! The path and target below are fixed by
//! `crate::model::audio::voxcpm::local_dit::sampler`, which already
//! implements CFM *inference* and is the source of truth:
//!
//! - `sampler::cfm_time_span` produces `t_span[0] == 1.0` descending to
//!   `t_span[n] == 0.0`.
//! - `sampler::solve_euler` starts `x` at NOISE at `t = 1` and updates with
//!   `x = x - velocity * dt` where `dt = t - t_next` is POSITIVE.
//!
//! So the probability path is `x_t = t * noise + (1 - t) * data`: at `t = 1`
//! it is pure noise, at `t = 0` it is pure data. Substituting
//! `v = noise - data` into the Euler update confirms it is self-consistent:
//! `x_t - v * dt = (t - dt) * noise + (1 - t + dt) * data`, which is exactly
//! `x` evaluated at `t - dt`.
//!
//! This is the OPPOSITE sign from the common rectified-flow convention where
//! `t = 0` is noise and `v = data - noise`. Do not "fix" it to match that
//! convention — training with the other sign integrates the ODE backwards
//! relative to `solve_euler`, and inference on the resulting checkpoint
//! produces garbage while the training loss curve still looks healthy.

use super::mse::mse_loss;
use crate::error::{Error, Result};
use numr::autograd::{Var, var_add, var_add_scalar, var_mul, var_mul_scalar, var_reshape, var_sub};
use numr::dtype::DType;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

/// Reshape `t` from `[batch]` to `[batch, 1, 1, ...]` matching `rank`.
fn reshape_time<R>(t: &Var<R>, batch: usize, rank: usize) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
{
    if t.shape() != [batch] {
        return Err(Error::InvalidArgument {
            arg: "t",
            reason: format!("expected shape [{batch}], got {:?}", t.shape()),
        });
    }
    let mut broadcast_shape = vec![1usize; rank];
    broadcast_shape[0] = batch;
    var_reshape(t, &broadcast_shape).map_err(Error::Numr)
}

/// Check that `noise` and `data` have identical, non-empty shapes.
fn check_matching_shapes<R: Runtime>(noise: &Var<R>, data: &Var<R>) -> Result<()> {
    if noise.shape().is_empty() {
        return Err(Error::InvalidArgument {
            arg: "noise",
            reason: "expected at least rank 1, got rank 0".to_string(),
        });
    }
    if noise.shape() != data.shape() {
        return Err(Error::InvalidArgument {
            arg: "data",
            reason: format!(
                "expected shape {:?} to match noise, got {:?}",
                noise.shape(),
                data.shape()
            ),
        });
    }
    Ok(())
}

/// The CFM probability path: `x_t = t * noise + (1 - t) * data`.
///
/// - `noise`, `data`: `[batch, ...]`, identical shapes.
/// - `t`: `[batch]`, broadcast over the remaining axes of `data`.
///
/// At `t = 1` this returns `noise` exactly; at `t = 0` it returns `data`
/// exactly. See the module doc for why `t = 1` is noise and not `t = 0`.
pub fn flow_matching_interpolate<R, C>(
    client: &C,
    noise: &Var<R>,
    data: &Var<R>,
    t: &Var<R>,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    check_matching_shapes(noise, data)?;
    let batch = data.shape()[0];
    let t_b = reshape_time::<R>(t, batch, data.shape().len())?;

    // 1 - t, built from scalar ops since there is no scalar-minus-var helper.
    let neg_t = var_mul_scalar(&t_b, -1.0, client).map_err(Error::Numr)?;
    let one_minus_t = var_add_scalar(&neg_t, 1.0, client).map_err(Error::Numr)?;

    let t_noise = var_mul(&t_b, noise, client).map_err(Error::Numr)?;
    let data_term = var_mul(&one_minus_t, data, client).map_err(Error::Numr)?;
    var_add(&t_noise, &data_term, client).map_err(Error::Numr)
}

/// The CFM velocity target: `v = noise - data`.
///
/// Trivial by itself — it exists so the sign lives in exactly one named,
/// documented place instead of being written inline at every call site. See
/// the module doc for why it is `noise - data` and not `data - noise`.
pub fn flow_matching_target<R, C>(client: &C, noise: &Var<R>, data: &Var<R>) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    check_matching_shapes(noise, data)?;
    var_sub(noise, data, client).map_err(Error::Numr)
}

/// Flow-matching training loss: MSE between the estimator's predicted
/// velocity and the target `noise - data`.
///
/// `predicted_velocity` carries the gradient; `noise` and `data` are targets
/// and need none.
pub fn flow_matching_loss<R, C>(
    client: &C,
    predicted_velocity: &Var<R>,
    noise: &Var<R>,
    data: &Var<R>,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    if predicted_velocity.shape() != noise.shape() {
        return Err(Error::InvalidArgument {
            arg: "predicted_velocity",
            reason: format!(
                "expected shape {:?} to match noise, got {:?}",
                noise.shape(),
                predicted_velocity.shape()
            ),
        });
    }
    let target = flow_matching_target(client, noise, data)?;
    mse_loss(client, predicted_velocity, &target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    fn var_from(data: &[f32], shape: &[usize]) -> Var<CpuRuntime> {
        let (_, device) = cpu_setup();
        Var::new(
            Tensor::<CpuRuntime>::from_slice(data, shape, &device).unwrap(),
            false,
        )
    }

    #[test]
    fn interpolate_at_t1_is_noise() {
        let (client, _) = cpu_setup();
        let noise = var_from(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let data = var_from(&[9.0, 9.0, 9.0, 9.0], &[2, 2]);
        let t = var_from(&[1.0, 1.0], &[2]);

        let x_t = flow_matching_interpolate(&client, &noise, &data, &t).unwrap();
        let got: Vec<f32> = x_t.tensor().to_vec();
        let want: Vec<f32> = noise.tensor().to_vec();
        assert_eq!(got, want);
    }

    #[test]
    fn interpolate_at_t0_is_data() {
        let (client, _) = cpu_setup();
        let noise = var_from(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let data = var_from(&[9.0, 9.0, 9.0, 9.0], &[2, 2]);
        let t = var_from(&[0.0, 0.0], &[2]);

        let x_t = flow_matching_interpolate(&client, &noise, &data, &t).unwrap();
        let got: Vec<f32> = x_t.tensor().to_vec();
        let want: Vec<f32> = data.tensor().to_vec();
        assert_eq!(got, want);
    }

    /// The load-bearing test: this is exactly the sampler's Euler update
    /// (`x = x - v * dt`, `sampler::solve_euler`), replayed on the
    /// interpolation formula. If the sign of `flow_matching_target` were
    /// flipped (`data - noise` instead of `noise - data`), `x_t - v * dt`
    /// would land on `x` evaluated at `t + dt`, not `t - dt`, and this
    /// assertion would fail.
    #[test]
    fn interpolate_round_trip_matches_euler_update() {
        let (client, _) = cpu_setup();
        let noise = var_from(&[0.3, -1.2, 2.5, 0.7, -0.4, 1.1], &[2, 3]);
        let data = var_from(&[1.5, 0.2, -0.8, 2.2, 1.0, -0.6], &[2, 3]);

        let t = 0.63f32;
        let dt = 0.11f32;
        let t_var = var_from(&[t, t], &[2]);
        let t_next_var = var_from(&[t - dt, t - dt], &[2]);

        let x_t = flow_matching_interpolate(&client, &noise, &data, &t_var).unwrap();
        let x_next = flow_matching_interpolate(&client, &noise, &data, &t_next_var).unwrap();
        let target = flow_matching_target(&client, &noise, &data).unwrap();

        let move_by = var_mul_scalar(&target, dt as f64, &client).unwrap();
        let predicted_next = var_sub(&x_t, &move_by, &client).unwrap();

        let got: Vec<f32> = predicted_next.tensor().to_vec();
        let want: Vec<f32> = x_next.tensor().to_vec();
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }
    }

    #[test]
    fn loss_is_zero_when_prediction_matches_target() {
        let (client, _) = cpu_setup();
        let noise = var_from(&[0.3, -1.2, 2.5, 0.7], &[2, 2]);
        let data = var_from(&[1.5, 0.2, -0.8, 2.2], &[2, 2]);
        let target = flow_matching_target(&client, &noise, &data).unwrap();

        let loss = flow_matching_loss(&client, &target, &noise, &data).unwrap();
        let val: Vec<f32> = loss.tensor().to_vec();
        assert!(val[0].abs() < 1e-6, "loss should be ~0, got {}", val[0]);
    }

    #[test]
    fn loss_is_positive_when_prediction_differs() {
        let (client, _) = cpu_setup();
        let noise = var_from(&[0.3, -1.2, 2.5, 0.7], &[2, 2]);
        let data = var_from(&[1.5, 0.2, -0.8, 2.2], &[2, 2]);
        let wrong_prediction = var_from(&[0.0, 0.0, 0.0, 0.0], &[2, 2]);

        let loss = flow_matching_loss(&client, &wrong_prediction, &noise, &data).unwrap();
        let val: Vec<f32> = loss.tensor().to_vec();
        assert!(val[0] > 0.0, "loss should be > 0, got {}", val[0]);
    }

    #[test]
    fn interpolate_shape_mismatch_is_err() {
        let (client, _) = cpu_setup();
        let noise = var_from(&[1.0, 2.0], &[2]);
        let data = var_from(&[1.0, 2.0, 3.0], &[3]);
        let t = var_from(&[0.5], &[1]);

        assert!(flow_matching_interpolate(&client, &noise, &data, &t).is_err());
    }

    #[test]
    fn target_shape_mismatch_is_err() {
        let (client, _) = cpu_setup();
        let noise = var_from(&[1.0, 2.0], &[2]);
        let data = var_from(&[1.0, 2.0, 3.0], &[3]);

        assert!(flow_matching_target(&client, &noise, &data).is_err());
    }

    #[test]
    fn loss_shape_mismatch_is_err() {
        let (client, _) = cpu_setup();
        let noise = var_from(&[1.0, 2.0], &[2]);
        let data = var_from(&[1.0, 2.0], &[2]);
        let predicted_velocity = var_from(&[1.0, 2.0, 3.0], &[3]);

        assert!(flow_matching_loss(&client, &predicted_velocity, &noise, &data).is_err());
    }
}
