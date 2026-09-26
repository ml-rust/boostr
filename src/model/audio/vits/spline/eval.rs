//! Forward and inverse evaluation of the rational-quadratic spline on the
//! per-element bin parameters [`super::knots`] and the gather in
//! `spline.rs` produce.

use crate::error::Result;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::RqSplineClient;

/// Per-element parameters of the bin each input falls in, shape `inputs.shape()`.
pub(super) struct Bin<R: Runtime> {
    pub(super) cumwidth: Tensor<R>,
    pub(super) width: Tensor<R>,
    pub(super) cumheight: Tensor<R>,
    pub(super) height: Tensor<R>,
    pub(super) delta: Tensor<R>,
    pub(super) deriv: Tensor<R>,
    pub(super) deriv_next: Tensor<R>,
}

/// `d_k + d_{k+1} - 2 * delta`.
fn intermediate1<R, C>(client: &C, bin: &Bin<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RqSplineClient<R>,
{
    let sum = client.add(&bin.deriv, &bin.deriv_next)?;
    Ok(client.sub(&sum, &client.mul_scalar(&bin.delta, 2.0)?)?)
}

/// `log(numerator') - 2 * log(denominator)` at bin position `t`.
///
/// `numerator' = delta^2 * (d_{k+1} * t^2 + 2 * delta * t(1-t) + d_k * (1-t)^2)`.
fn log_abs_det<R, C>(
    client: &C,
    bin: &Bin<R>,
    t: &Tensor<R>,
    one_minus_t: &Tensor<R>,
    t_one_minus_t: &Tensor<R>,
    denominator: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RqSplineClient<R>,
{
    let upper = client.mul(&bin.deriv_next, &client.square(t)?)?;
    let middle = client.mul(&client.mul_scalar(&bin.delta, 2.0)?, t_one_minus_t)?;
    let lower = client.mul(&bin.deriv, &client.square(one_minus_t)?)?;
    let inner = client.add(&client.add(&upper, &middle)?, &lower)?;
    let numerator = client.mul(&client.square(&bin.delta)?, &inner)?;
    let log_den = client.mul_scalar(&client.log(denominator)?, 2.0)?;
    Ok(client.sub(&client.log(&numerator)?, &log_den)?)
}

/// Forward map of in-range inputs: `(outputs, log_abs_det)`.
pub(super) fn forward<R, C>(
    client: &C,
    x: &Tensor<R>,
    bin: &Bin<R>,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: RqSplineClient<R>,
{
    let theta = client.div(&client.sub(x, &bin.cumwidth)?, &bin.width)?;
    let one_minus = client.rsub_scalar(&theta, 1.0)?;
    let t1mt = client.mul(&theta, &one_minus)?;

    let slope_term = client.mul(&bin.delta, &client.square(&theta)?)?;
    let deriv_term = client.mul(&bin.deriv, &t1mt)?;
    let numerator = client.mul(&bin.height, &client.add(&slope_term, &deriv_term)?)?;
    let inter1 = intermediate1(client, bin)?;
    let denominator = client.add(&bin.delta, &client.mul(&inter1, &t1mt)?)?;
    let outputs = client.add(&bin.cumheight, &client.div(&numerator, &denominator)?)?;

    let logdet = log_abs_det(client, bin, &theta, &one_minus, &t1mt, &denominator)?;
    Ok((outputs, logdet))
}

/// Inverse map of in-range inputs by the quadratic root: `(outputs, log_abs_det)`.
pub(super) fn inverse<R, C>(
    client: &C,
    y: &Tensor<R>,
    bin: &Bin<R>,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: RqSplineClient<R>,
{
    let inter1 = intermediate1(client, bin)?;
    let inter2 = client.sub(y, &bin.cumheight)?;
    let inter3 = client.mul(&inter2, &inter1)?;

    let a = client.add(
        &client.mul(&bin.height, &client.sub(&bin.delta, &bin.deriv)?)?,
        &inter3,
    )?;
    let b = client.sub(&client.mul(&bin.height, &bin.deriv)?, &inter3)?;
    let c = client.mul(&client.neg(&bin.delta)?, &inter2)?;

    let four_ac = client.mul(&client.mul_scalar(&a, 4.0)?, &c)?;
    let discriminant = client.sub(&client.square(&b)?, &four_ac)?;
    let root_den = client.sub(&client.neg(&b)?, &client.sqrt(&discriminant)?)?;
    let root = client.div(&client.mul_scalar(&c, 2.0)?, &root_den)?;
    let outputs = client.add(&client.mul(&root, &bin.width)?, &bin.cumwidth)?;

    let one_minus = client.rsub_scalar(&root, 1.0)?;
    let t1mt = client.mul(&root, &one_minus)?;
    let denominator = client.add(&bin.delta, &client.mul(&inter1, &t1mt)?)?;
    let logdet = log_abs_det(client, bin, &root, &one_minus, &t1mt, &denominator)?;
    Ok((outputs, client.neg(&logdet)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    /// A bin whose derivative matches its slope (`deriv == deriv_next ==
    /// delta == 1`) reduces the rational-quadratic map to the identity, with
    /// log-abs-det 0, for any `x` inside the bin.
    fn unit_slope_bin(device: &CpuDevice) -> Bin<CpuRuntime> {
        let one = |v: f32| Tensor::<CpuRuntime>::from_slice(&[v], &[1], device).unwrap();
        Bin {
            cumwidth: one(0.0),
            width: one(1.0),
            cumheight: one(0.0),
            height: one(1.0),
            delta: one(1.0),
            deriv: one(1.0),
            deriv_next: one(1.0),
        }
    }

    #[test]
    fn forward_is_identity_when_derivative_matches_slope() {
        let (client, device) = cpu_setup();
        let bin = unit_slope_bin(&device);
        let x = Tensor::<CpuRuntime>::from_slice(&[0.37f32], &[1], &device).unwrap();
        let (out, logdet) = forward(&client, &x, &bin).unwrap();
        assert!((out.to_vec::<f32>()[0] - 0.37).abs() < 1e-6);
        assert!(logdet.to_vec::<f32>()[0].abs() < 1e-6);
    }

    #[test]
    fn inverse_is_identity_when_derivative_matches_slope() {
        let (client, device) = cpu_setup();
        let bin = unit_slope_bin(&device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.62f32], &[1], &device).unwrap();
        let (out, logdet) = inverse(&client, &y, &bin).unwrap();
        assert!((out.to_vec::<f32>()[0] - 0.62).abs() < 1e-6);
        assert!(logdet.to_vec::<f32>()[0].abs() < 1e-6);
    }
}
