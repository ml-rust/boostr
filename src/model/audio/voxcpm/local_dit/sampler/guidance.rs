//! Classifier-free guidance: the CFG-zero-star rescale and the combine.

use crate::error::{Error, Result};
use crate::model::traits::ModelClient;
use numr::autograd::{
    Var, var_add, var_add_scalar, var_div, var_mul, var_mul_scalar, var_reshape, var_square,
    var_sub, var_sum,
};
use numr::dtype::DType;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::Runtime;

/// CFG-zero-star rescale (`unified_cfm.py:79-82`).
///
/// ```text
/// st_star = sum(pos * neg) / (sum(neg^2) + 1e-8)
/// ```
///
/// `pos` is the CONDITIONAL velocity and `neg` the UNCONDITIONAL one, both
/// `[batch, feat_dim, patch_size]`. The reduction is per batch row over the
/// flattened `feat_dim * patch_size`, and `1e-8` is added INSIDE the
/// denominator sum, before the divide — not to the quotient. The result is
/// `[batch, 1, 1]`, shaped to broadcast back over the velocities.
pub(super) fn optimized_scale<R, C>(client: &C, pos: &Var<R>, neg: &Var<R>) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let shape = pos.shape().to_vec();
    if shape.len() != 3 || neg.shape() != shape.as_slice() {
        return Err(Error::InvalidArgument {
            arg: "pos",
            reason: format!(
                "expected two matching 3D velocities, got {shape:?} and {:?}",
                neg.shape()
            ),
        });
    }
    let batch = shape[0];
    let flat: usize = shape[1..].iter().product();
    let pos_flat = var_reshape(pos, &[batch, flat]).map_err(Error::Numr)?;
    let neg_flat = var_reshape(neg, &[batch, flat]).map_err(Error::Numr)?;

    let dot = var_sum(
        &var_mul(&pos_flat, &neg_flat, client).map_err(Error::Numr)?,
        &[1],
        true,
        client,
    )
    .map_err(Error::Numr)?;
    let sq = var_sum(
        &var_square(&neg_flat, client).map_err(Error::Numr)?,
        &[1],
        true,
        client,
    )
    .map_err(Error::Numr)?;
    let sq = var_add_scalar(&sq, 1e-8, client).map_err(Error::Numr)?;

    let scale = var_div(&dot, &sq, client).map_err(Error::Numr)?;
    var_reshape(&scale, &[batch, 1, 1]).map_err(Error::Numr)
}

/// Classifier-free guidance combine (`unified_cfm.py:128`).
///
/// ```text
/// v = v_uncond * st_star + cfg_value * (v_cond - v_uncond * st_star)
/// ```
///
/// `st_star` is `[batch, 1, 1]` and broadcasts over the velocities. At
/// `cfg_value == 1.0` the two terms telescope and the result is `v_cond`, so
/// guidance is off — `st_star` drops out entirely.
pub(super) fn cfg_combine<R, C>(
    client: &C,
    v_cond: &Var<R>,
    v_uncond: &Var<R>,
    st_star: &Var<R>,
    cfg_value: f32,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let scaled = var_mul(v_uncond, st_star, client).map_err(Error::Numr)?;
    let delta = var_sub(v_cond, &scaled, client).map_err(Error::Numr)?;
    let guided = var_mul_scalar(&delta, cfg_value as f64, client).map_err(Error::Numr)?;
    var_add(&scaled, &guided, client).map_err(Error::Numr)
}

#[cfg(test)]
pub(super) mod tests {
    //! The failure mode here is plausible-but-wrong output, so each test pins
    //! a value the reference fixes exactly: the `optimized_scale` quotient
    //! (epsilon placement included) and the `cfg_value = 1.0` collapse.

    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    pub(in super::super) fn var(
        data: &[f32],
        shape: &[usize],
        device: &CpuDevice,
    ) -> Var<CpuRuntime> {
        Var::new(
            Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap(),
            false,
        )
    }

    pub(in super::super) fn values(v: &Var<CpuRuntime>) -> Vec<f32> {
        v.tensor().contiguous().unwrap().to_vec()
    }

    /// Row 0: orthogonal velocities, so `dot = 0` and the scale is exactly 0.
    /// Row 1: `pos = 3 * neg`, so the scale is exactly 3 — `1e-8` is far below
    /// the `f32` ulp of `sum(neg^2) = 20`, and cannot perturb it.
    /// Two rows also pin the reduction as PER ROW, not over the whole batch.
    #[test]
    fn optimized_scale_matches_hand_computed_values() {
        let (client, device) = cpu_setup();
        let shape = [2, 2, 2];
        let pos = var(&[1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 12.0, 0.0], &shape, &device);
        let neg = var(&[0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 4.0, 0.0], &shape, &device);

        let scale = optimized_scale(&client, &pos, &neg).unwrap();
        assert_eq!(scale.shape(), &[2, 1, 1]);
        assert_eq!(values(&scale), vec![0.0, 3.0]);
    }

    /// The `1e-8` is INSIDE the denominator sum. With an all-zero `neg` the
    /// denominator is `1e-8`, so the scale is `0 / 1e-8 = 0`. Adding the epsilon
    /// after the divide instead gives `0 / 0 = NaN` and this assertion fires.
    #[test]
    fn optimized_scale_epsilon_guards_a_zero_denominator() {
        let (client, device) = cpu_setup();
        let shape = [1, 2, 2];
        let pos = var(&[1.0, 2.0, 3.0, 4.0], &shape, &device);
        let neg = var(&[0.0, 0.0, 0.0, 0.0], &shape, &device);

        let scale = values(&optimized_scale(&client, &pos, &neg).unwrap());
        assert!(scale[0].is_finite(), "scale must be finite, got {scale:?}");
        assert_eq!(scale[0], 0.0);
    }

    /// `v = v_uncond * st + cfg * (v_cond - v_uncond * st)`.
    ///
    /// At `cfg = 1.0` the two terms telescope and the result is `v_cond`. Every
    /// value here is dyadic, so the collapse is exact and the assertion compares
    /// bits — a swapped `v_cond`/`v_uncond`, or a missing `st`, breaks it.
    /// The `cfg = 2.0` case pins the formula itself, which `cfg = 1.0` alone
    /// cannot: it is the only weight the collapse does not hide.
    #[test]
    fn cfg_combine_collapses_to_the_conditional_velocity_at_one() {
        let (client, device) = cpu_setup();
        let shape = [1, 2, 2];
        let v_cond = var(&[1.0, 2.0, -3.0, 0.5], &shape, &device);
        let v_uncond = var(&[4.0, -1.0, 2.0, 8.0], &shape, &device);
        let st = var(&[0.5], &[1, 1, 1], &device);

        let at_one = values(&cfg_combine(&client, &v_cond, &v_uncond, &st, 1.0).unwrap());
        for (got, want) in at_one.iter().zip(values(&v_cond).iter()) {
            assert_eq!(got.to_bits(), want.to_bits(), "{at_one:?}");
        }

        // scaled = [2, -0.5, 1, 4]; delta = [-1, 2.5, -4, -3.5];
        // scaled + 2 * delta = [0, 4.5, -7, -3].
        let at_two = values(&cfg_combine(&client, &v_cond, &v_uncond, &st, 2.0).unwrap());
        assert_eq!(at_two, vec![0.0, 4.5, -7.0, -3.0]);
    }
}
