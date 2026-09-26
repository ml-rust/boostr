//! Monotonic rational-quadratic spline with identity tails, the coupling
//! transform of the VITS stochastic duration predictor.
//!
//! Port of HF `_unconstrained_rational_quadratic_spline` and
//! `_rational_quadratic_spline` in `transformers.models.vits.modeling_vits`.
//! Inside `[-tail_bound, tail_bound]` the transform is a `num_bins`-piece
//! rational-quadratic map. Outside it is the identity with log-abs-det 0.
//!
//! Every element runs the spline on its input clamped into the interval. A
//! final select keeps the spline result inside and the raw input outside, so
//! the transform stays on device with no masked compaction.

use super::eval::{Bin, forward, inverse};
use super::knots::knots;
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, CumulativeOps, IndexingOps, ReduceOps,
    ScalarOps, ShapeOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Lower bound on each bin width, as a fraction of the interval.
pub const MIN_BIN_WIDTH: f64 = 1e-3;
/// Lower bound on each bin height, as a fraction of the interval.
pub const MIN_BIN_HEIGHT: f64 = 1e-3;
/// Lower bound on each knot derivative.
pub const MIN_DERIVATIVE: f64 = 1e-3;

/// Margin on the last search knot, so an input equal to `tail_bound` lands in the last bin.
const LAST_KNOT_MARGIN: f64 = 1e-6;

/// Client ops the spline composes.
pub trait RqSplineClient<R: Runtime>:
    RuntimeClient<R>
    + ActivationOps<R>
    + BinaryOps<R>
    + CompareOps<R>
    + ConditionalOps<R>
    + CumulativeOps<R>
    + IndexingOps<R>
    + ReduceOps<R>
    + ScalarOps<R>
    + ShapeOps<R>
    + TypeConversionOps<R>
    + UnaryOps<R>
    + UtilityOps<R>
{
}

impl<R, C> RqSplineClient<R> for C
where
    R: Runtime,
    C: RuntimeClient<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CumulativeOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + UtilityOps<R>,
{
}

/// Applies the rational-quadratic spline to `inputs`, or its inverse when `reverse` is set.
///
/// Shapes: `inputs` is `[...]`, `unnormalized_widths` and `unnormalized_heights`
/// are `[..., num_bins]`, `unnormalized_derivatives` is `[..., num_bins - 1]`.
/// Returns `(outputs, log_abs_det)`, both shaped like `inputs`. In reverse,
/// `log_abs_det` is the log-abs-det of the inverse map.
///
/// HF raises on a negative discriminant in the inverse solve. Here that
/// element yields NaN, because the check needs a device-to-host read.
pub fn unconstrained_rational_quadratic_spline<R, C>(
    client: &C,
    inputs: &Tensor<R>,
    unnormalized_widths: &Tensor<R>,
    unnormalized_heights: &Tensor<R>,
    unnormalized_derivatives: &Tensor<R>,
    reverse: bool,
    tail_bound: f64,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: RqSplineClient<R>,
{
    let num_bins = check_shapes(
        inputs,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
    )?;
    if !(tail_bound.is_finite() && tail_bound > 0.0) {
        return Err(Error::InvalidArgument {
            arg: "tail_bound",
            reason: format!("expected a finite positive bound, got {tail_bound}"),
        });
    }
    let bin_axis = inputs.ndim();

    let clamped = client.clamp(inputs, -tail_bound, tail_bound)?;
    let inside = client.eq(&clamped, inputs)?;

    let widths = knots(client, unnormalized_widths, MIN_BIN_WIDTH, tail_bound)?;
    let heights = knots(client, unnormalized_heights, MIN_BIN_HEIGHT, tail_bound)?;
    let edge = ((1.0 - MIN_DERIVATIVE).exp() - 1.0).ln();
    let padded = client.pad(unnormalized_derivatives, &[1, 1], edge)?;
    let derivatives = client.add_scalar(&client.softplus(&padded)?, MIN_DERIVATIVE)?;

    // Bin index = (count of search knots <= x) - 1, in [0, num_bins - 1] for clamped x.
    let locations = if reverse { &heights.cum } else { &widths.cum };
    let search = client.pad(
        &locations.narrow(-1, 0, num_bins)?,
        &[0, 1],
        tail_bound + LAST_KNOT_MARGIN,
    )?;
    let hits = client.ge(&clamped.unsqueeze(-1)?, &search)?;
    let count = client.sum(&hits, &[bin_axis], true)?;
    let index = client.cast(&client.sub_scalar(&count, 1.0)?, DType::I32)?;

    let pick = |t: &Tensor<R>| -> Result<Tensor<R>> {
        Ok(client.gather(t, bin_axis, &index)?.squeeze(Some(-1)))
    };
    let delta = client.div(&heights.bins, &widths.bins)?;
    let bin = Bin {
        cumwidth: pick(&widths.cum)?,
        width: pick(&widths.bins)?,
        cumheight: pick(&heights.cum)?,
        height: pick(&heights.bins)?,
        delta: pick(&delta)?,
        deriv: pick(&derivatives)?,
        deriv_next: pick(&derivatives.narrow(-1, 1, num_bins)?)?,
    };

    let (spline_out, spline_logdet) = if reverse {
        inverse(client, &clamped, &bin)?
    } else {
        forward(client, &clamped, &bin)?
    };
    let outputs = client.where_cond(&inside, &spline_out, inputs)?;
    let zeros = Tensor::<R>::zeros(inputs.shape(), inputs.dtype(), inputs.device())?;
    let log_abs_det = client.where_cond(&inside, &spline_logdet, &zeros)?;
    Ok((outputs, log_abs_det))
}

/// Checks parameter shapes against `inputs` and returns `num_bins`.
fn check_shapes<R: Runtime>(
    inputs: &Tensor<R>,
    widths: &Tensor<R>,
    heights: &Tensor<R>,
    derivatives: &Tensor<R>,
) -> Result<usize> {
    let base = inputs.shape();
    let num_bins = match widths.shape().split_last() {
        Some((&k, lead)) if lead == base && k >= 1 => k,
        _ => {
            return Err(Error::InvalidArgument {
                arg: "unnormalized_widths",
                reason: format!(
                    "expected {base:?} plus a non-empty bin axis, got {:?}",
                    widths.shape()
                ),
            });
        }
    };
    if heights.shape() != widths.shape() {
        return Err(Error::InvalidArgument {
            arg: "unnormalized_heights",
            reason: format!("expected {:?}, got {:?}", widths.shape(), heights.shape()),
        });
    }
    let mut expected = base.to_vec();
    expected.push(num_bins - 1);
    if derivatives.shape() != expected.as_slice() {
        return Err(Error::InvalidArgument {
            arg: "unnormalized_derivatives",
            reason: format!("expected {expected:?}, got {:?}", derivatives.shape()),
        });
    }
    if MIN_BIN_WIDTH.max(MIN_BIN_HEIGHT) * num_bins as f64 > 1.0 {
        return Err(Error::InvalidArgument {
            arg: "unnormalized_widths",
            reason: format!("minimal bin size too large for {num_bins} bins"),
        });
    }
    Ok(num_bins)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    const NUM_BINS: usize = 10;
    const TAIL_BOUND: f64 = 5.0;
    const TOL: f32 = 1e-5;
    /// Row 12 in reverse: the log-abs-det has a kink at the knot, with slope about -255
    /// per unit input on the upper-bin side. The knot is `10 * cumsum - 5`, so one f32
    /// ulp of the cumsum near 0.45 moves it by 3e-7, and log-abs-det by up to 7.7e-5.
    /// HF's own formula at that knot offset, in the same bin, gives the same shift.
    const KNOT_ROW: usize = 12;
    const KNOT_LOGDET_TOL: f32 = 1e-4;

    // Values from `tests/tools/vits_rq_spline_oracle/rq_spline_oracle.py` (HF transformers).
    // The last two inputs sit on interior knots: width knot 4 of row 11, height knot 6 of row 12.
    const INPUTS: [f32; 13] = [
        -6.0, -5.0, -4.2, -2.5, -0.3, 0.0, 1.7, 3.3, 4.99, 5.0, 5.5, -1.0866232, -0.4575448,
    ];
    const FWD_OUT: [f32; 13] = [
        -6.0, -5.0, -4.534539, -1.958761, 0.8839252, 1.6868204, 3.255823, 4.0354095, 4.9899836,
        5.0, 5.5, -2.24038, -1.2379924,
    ];
    const FWD_LOGDET: [f32; 13] = [
        0.0,
        -5.9604645e-08,
        -0.9450022,
        2.3894436,
        -1.6463735,
        -0.46287966,
        -3.0933309,
        -1.3595376,
        0.0033640265,
        -1.1920929e-07,
        0.0,
        -1.3747706,
        -2.8113594,
    ];
    const REV_OUT: [f32; 13] = [
        -6.0,
        -5.0,
        -3.3577843,
        -2.560089,
        -2.6605542,
        -2.1329312,
        -0.16702181,
        0.25536245,
        4.990016,
        5.0,
        5.5,
        0.5909447,
        1.5598774,
    ];
    const REV_LOGDET: [f32; 13] = [
        0.0,
        5.9604645e-08,
        0.70236367,
        -1.8883882,
        -0.6668639,
        -0.4197416,
        -1.2634765,
        -1.3737197,
        -0.0033581555,
        1.1920929e-07,
        0.0,
        0.12790552,
        1.339207,
    ];
    /// Round-trip log_abs_det bound per element: 1e-5 plus |d log_abs_det / dy| of the
    /// reverse map times one f32 ulp of the forward output y. One ulp of y moves the
    /// reverse log_abs_det by that product, so the round trip carries no tighter bound.
    const ROUND_TRIP_LOGDET_TOL: [f32; 13] = [
        10e-6, 10e-6, 11e-6, 11e-6, 11e-6, 11e-6, 21e-6, 29e-6, 11e-6, 10e-6, 10e-6, 15e-6, 17e-6,
    ];

    /// Spline parameters for `n` rows, from the closed forms the oracle script uses.
    fn params(
        n: usize,
        device: &CpuDevice,
    ) -> (Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>) {
        let table = |cols: usize, f: &dyn Fn(f64, f64) -> f64| -> Tensor<CpuRuntime> {
            let data: Vec<f32> = (0..n)
                .flat_map(|i| (0..cols).map(move |j| (i as f64, j as f64)))
                .map(|(i, j)| f(i, j) as f32)
                .collect();
            Tensor::<CpuRuntime>::from_slice(&data, &[n, cols], device).unwrap()
        };
        let widths = table(NUM_BINS, &|i, j| (0.7 * i + 1.3 * j).sin());
        let heights = table(NUM_BINS, &|i, j| 1.5 * (0.5 * i - 0.9 * j).cos());
        let derivatives = table(NUM_BINS - 1, &|i, j| 2.0 * (0.3 * i * j + 0.2).sin());
        (widths, heights, derivatives)
    }

    fn run(x: &[f32], reverse: bool) -> (Vec<f32>, Vec<f32>) {
        let (client, device) = cpu_setup();
        let inputs = Tensor::<CpuRuntime>::from_slice(x, &[x.len()], &device).unwrap();
        let (w, h, d) = params(x.len(), &device);
        let (out, logdet) = unconstrained_rational_quadratic_spline(
            &client, &inputs, &w, &h, &d, reverse, TAIL_BOUND,
        )
        .unwrap();
        (out.to_vec(), logdet.to_vec())
    }

    fn assert_within(label: &str, got: &[f32], want: &[f32], tol: impl Fn(usize) -> f32) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let t = tol(i);
            assert!(
                (g - w).abs() <= t,
                "{label}[{i}]: got {g}, want {w}, tol {t}"
            );
        }
    }

    fn assert_close(label: &str, got: &[f32], want: &[f32], tol: f32) {
        assert_within(label, got, want, |_| tol);
    }

    #[test]
    fn forward_matches_hf_oracle() {
        let (out, logdet) = run(&INPUTS, false);
        assert_close("forward outputs", &out, &FWD_OUT, TOL);
        assert_close("forward log_abs_det", &logdet, &FWD_LOGDET, TOL);
    }

    #[test]
    fn reverse_matches_hf_oracle() {
        let (out, logdet) = run(&INPUTS, true);
        assert_close("reverse outputs", &out, &REV_OUT, TOL);
        let tol = |i| if i == KNOT_ROW { KNOT_LOGDET_TOL } else { TOL };
        assert_within("reverse log_abs_det", &logdet, &REV_LOGDET, tol);
    }

    #[test]
    fn tails_are_identity_with_zero_log_abs_det() {
        let inf = f32::INFINITY;
        let x = [-9.0f32, -5.001, 5.001, 9.0, inf, -inf];
        for reverse in [false, true] {
            let (out, logdet) = run(&x, reverse);
            assert_eq!(out, x.to_vec(), "reverse={reverse}");
            assert_eq!(logdet, vec![0.0; x.len()], "reverse={reverse}");
        }
    }

    #[test]
    fn inverse_undoes_forward() {
        let (client, device) = cpu_setup();
        let inputs = Tensor::<CpuRuntime>::from_slice(&INPUTS, &[INPUTS.len()], &device).unwrap();
        let (w, h, d) = params(INPUTS.len(), &device);
        let (y, fwd_logdet) = unconstrained_rational_quadratic_spline(
            &client, &inputs, &w, &h, &d, false, TAIL_BOUND,
        )
        .unwrap();
        let (x, rev_logdet) =
            unconstrained_rational_quadratic_spline(&client, &y, &w, &h, &d, true, TAIL_BOUND)
                .unwrap();
        assert_close("round trip", &x.to_vec::<f32>(), &INPUTS, TOL);
        let total: Vec<f32> = fwd_logdet
            .to_vec::<f32>()
            .iter()
            .zip(rev_logdet.to_vec::<f32>())
            .map(|(a, b)| a + b)
            .collect();
        let tol = |i: usize| ROUND_TRIP_LOGDET_TOL[i];
        assert_within("log_abs_det sum", &total, &[0.0; 13], tol);
    }

    #[test]
    fn rejects_mismatched_derivative_shape() {
        let (client, device) = cpu_setup();
        let inputs = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 2], &[2], &device).unwrap();
        let (w, h, _) = params(2, &device);
        let d = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 2 * NUM_BINS], &[2, NUM_BINS], &device)
            .unwrap();
        let err = unconstrained_rational_quadratic_spline(
            &client, &inputs, &w, &h, &d, false, TAIL_BOUND,
        );
        assert!(err.is_err());
    }
}
