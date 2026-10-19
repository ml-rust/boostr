//! The device-side inverse-CDF draw of `speculative_verify`.
//!
//! Ids travel as exact F32 values, so every step runs on F32 masks.

use crate::error::Result;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, CumulativeOps, ReduceOps, ScalarOps, TensorOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A tensor of rank `like.ndim()` with every size 1, filled with `value`.
pub fn scalar_like<R: Runtime<DType = DType>>(like: &Tensor<R>, value: f64) -> Result<Tensor<R>> {
    let shape = vec![1; like.ndim()];
    Ok(Tensor::<R>::full_scalar(
        &shape,
        DType::F32,
        value,
        like.device(),
    )?)
}

/// `cond ? x : y` for a 0/1 F32 mask `cond`, with broadcasting.
///
/// The result is exact for finite `x` and `y`: one product is 0, the other is the pick.
pub fn select<R: Runtime<DType = DType>>(
    client: &R::Client,
    cond: &Tensor<R>,
    x: &Tensor<R>,
    y: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let not_cond = client.add_scalar(&client.mul_scalar(cond, -1.0)?, 1.0)?;
    let picked = client.mul(cond, x)?;
    let other = client.mul(&not_cond, y)?;
    Ok(client.add(&picked, &other)?)
}

/// Draws one id per row of `dist` by inverse CDF in id order.
///
/// - `dist`: `[B, 1, V]` F32, non-negative.
/// - `uniform`: `[B, 1, 1]` F32 in `[0, 1)`.
/// - `ids`: `[1, 1, V]` F32 holding `0 .. V`.
///
/// The draw is the first id with mass whose running sum exceeds `uniform * total`.
/// `total` is the last running sum, so the threshold and the scan share one rounding.
/// No crossing falls back to the last id with mass. An all-zero row returns id 0.
///
/// Returns `[B, 1]` F32 ids.
pub fn inverse_cdf_draw<R: Runtime<DType = DType>>(
    client: &R::Client,
    dist: &Tensor<R>,
    uniform: &Tensor<R>,
    ids: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R::Client: TensorOps<R> + CompareOps<R> + ScalarOps<R>,
{
    let vocab = dist.shape()[2];
    let running = client.cumsum(dist, 2)?;
    let total = running.narrow(2, vocab - 1, 1)?.contiguous()?;
    let threshold = client.mul(uniform, &total)?;
    let has_mass = client.gt(dist, &scalar_like(dist, 0.0)?)?;
    let crossed = client.mul(&client.gt(&running, &threshold)?, &has_mass)?;
    let no_hit = scalar_like(dist, vocab as f64)?;
    let candidates = select::<R>(client, &crossed, ids, &no_hit)?;
    let first = client.min(&candidates, &[2], false)?;
    let ids_plus_one = client.add_scalar(ids, 1.0)?;
    let last_plus_one = client.max(&client.mul(&has_mass, &ids_plus_one)?, &[2], false)?;
    let last = client.add_scalar(&last_plus_one, -1.0)?;
    let last = client.maximum(&last, &scalar_like(&last, 0.0)?)?;
    let found = client.lt(&first, &scalar_like(&first, vocab as f64)?)?;
    select::<R>(client, &found, &first, &last)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::ops::UtilityOps;
    use numr::runtime::cpu::CpuRuntime;

    fn draw(rows: &[&[f32]], us: &[f32]) -> Vec<f32> {
        let (client, device) = cpu_setup();
        let vocab = rows[0].len();
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let dist = Tensor::<CpuRuntime>::from_slice(&flat, &[rows.len(), 1, vocab], &device)
            .expect("dist");
        let u = Tensor::<CpuRuntime>::from_slice(us, &[us.len(), 1, 1], &device).expect("u");
        let ids = client
            .arange(0.0, vocab as f64, 1.0, DType::F32)
            .expect("ids")
            .reshape(&[1, 1, vocab])
            .expect("ids shape");
        inverse_cdf_draw::<CpuRuntime>(&client, &dist, &u, &ids)
            .expect("draw")
            .to_vec::<f32>()
    }

    #[test]
    fn the_draw_follows_the_cdf_in_id_order() {
        let row: &[f32] = &[0.25, 0.25, 0.5];
        assert_eq!(
            draw(&[row, row, row, row], &[0.0, 0.24, 0.25, 0.75]),
            [0.0, 0.0, 1.0, 2.0]
        );
    }

    #[test]
    fn the_draw_never_returns_a_zero_probability_id() {
        let row: &[f32] = &[0.0, 0.5, 0.0, 0.5, 0.0];
        let us = [0.0, 0.49, 0.5, 0.999_999_9, 1.0];
        assert_eq!(draw(&[row; 5], &us), [1.0, 1.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn an_unnormalized_row_scales_the_threshold() {
        let row: &[f32] = &[2.0, 0.0, 6.0];
        assert_eq!(draw(&[row, row], &[0.24, 0.26]), [0.0, 2.0]);
    }

    #[test]
    fn select_picks_per_element() {
        let (client, device) = cpu_setup();
        let c = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 1.0], &[3], &device).expect("c");
        let x = Tensor::<CpuRuntime>::from_slice(&[7.0f32, 8.0, 9.0], &[3], &device).expect("x");
        let y = scalar_like(&x, -2.5).expect("y");
        let out = select::<CpuRuntime>(&client, &c, &x, &y).expect("select");
        assert_eq!(out.to_vec::<f32>(), [7.0, -2.5, 9.0]);
    }
}
