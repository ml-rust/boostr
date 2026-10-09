//! Bicubic resampling along one axis with corners aligned: PyTorch's
//! `interpolate(mode="bicubic", align_corners=True)` restricted to an axis
//! whose length changes. Along an axis of unchanged length the bicubic
//! weights are `[0, 1, 0, 0]`, so a 2-D resize that changes one axis is
//! this 1-D resize.
//!
//! Output `j` reads source position `j · (n_in - 1) / (n_out - 1)`, in
//! `f32` as PyTorch computes it, and weighs the four inputs around it with
//! the cubic convolution kernel of `A = -0.75`. Indices past either end
//! clamp to the edge. The four taps gather on the device and sum there.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{BinaryOps, IndexingOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// PyTorch's bicubic `A`.
const CUBIC_A: f32 = -0.75;

/// Kernel weight at distance `x` in `[0, 1]`.
fn near(x: f32) -> f32 {
    ((CUBIC_A + 2.0) * x - (CUBIC_A + 3.0)) * x * x + 1.0
}

/// Kernel weight at distance `x` in `[1, 2]`.
fn far(x: f32) -> f32 {
    ((CUBIC_A * x - 5.0 * CUBIC_A) * x + 8.0 * CUBIC_A) * x - 4.0 * CUBIC_A
}

/// The four source indices and weights of every output position.
struct Taps {
    index: [Vec<i64>; 4],
    weight: [Vec<f32>; 4],
}

fn taps(n_in: usize, n_out: usize) -> Taps {
    let scale = if n_out > 1 {
        (n_in as f32 - 1.0) / (n_out as f32 - 1.0)
    } else {
        0.0
    };
    let last = n_in as i64 - 1;
    let mut index: [Vec<i64>; 4] = Default::default();
    let mut weight: [Vec<f32>; 4] = Default::default();
    for j in 0..n_out {
        let real = scale * j as f32;
        let base = real.floor();
        let t = real - base;
        let opposite = 1.0 - t;
        let w = [far(t + 1.0), near(t), near(opposite), far(opposite + 1.0)];
        for k in 0..4 {
            let at = (base as i64 - 1 + k as i64).clamp(0, last);
            index[k].push(at);
            weight[k].push(w[k]);
        }
    }
    Taps { index, weight }
}

/// Resample `x` along `dim` to `n_out` positions. `x` must be `f32`.
///
/// Returns [`Error::InvalidArgument`] for a `dim` past the rank, an empty
/// axis, or `n_out` of 0.
pub fn resize_bicubic<R, C>(
    client: &C,
    x: &Tensor<R>,
    dim: usize,
    n_out: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + IndexingOps<R> + BinaryOps<R>,
{
    let shape = x.shape().to_vec();
    let n_in = shape.get(dim).copied().unwrap_or(0);
    if n_in == 0 || n_out == 0 {
        return Err(Error::InvalidArgument {
            arg: "resize_bicubic",
            reason: format!("axis {dim} of {shape:?} to {n_out} positions"),
        });
    }
    if n_in == n_out {
        return Ok(x.clone());
    }
    let device = x.device();
    let taps = taps(n_in, n_out);
    let mut wshape = vec![1usize; shape.len()];
    wshape[dim] = n_out;
    let mut out: Option<Tensor<R>> = None;
    for k in 0..4 {
        let index = Tensor::<R>::from_slice(&taps.index[k], &[n_out], device)?;
        let weight = Tensor::<R>::from_slice(&taps.weight[k], &wshape, device)?;
        let term = client.mul(&client.index_select(x, dim, &index)?, &weight)?;
        out = Some(match out {
            None => term,
            Some(acc) => client.add(&acc, &term)?,
        });
    }
    out.ok_or_else(|| Error::InvalidArgument {
        arg: "resize_bicubic",
        reason: "no taps".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn weights_sum_to_one_and_end_points_land_on_inputs() {
        let t = taps(1001, 1024);
        for j in 0..1024 {
            let s: f32 = (0..4).map(|k| t.weight[k][j]).sum();
            assert!((s - 1.0).abs() < 1e-6, "output {j}: {s}");
        }
        assert_eq!((t.index[1][0], t.weight[1][0]), (0, 1.0));
        assert_eq!(t.index[1][1023], 1000);
    }

    #[test]
    fn a_ramp_matches_pytorch_with_clamped_edges() {
        // `F.interpolate(arange(11), size=21, mode="bicubic",
        // align_corners=True)`: the interior stays a ramp; the clamped edge
        // taps bend the second and second-to-last outputs.
        let (client, device) = cpu_setup();
        let ramp: Vec<f32> = (0..11).map(|i| i as f32).collect();
        let x = Tensor::<CpuRuntime>::from_slice(&ramp, &[1, 11, 1], &device).expect("x");
        let y: Vec<f32> = resize_bicubic(&client, &x, 1, 21).expect("resize").to_vec();
        let mut want: Vec<f32> = (0..21).map(|j| j as f32 * 0.5).collect();
        want[1] = 0.40625;
        want[19] = 9.59375;
        assert_eq!(y, want);
    }

    #[test]
    fn a_sine_matches_pytorch_at_the_clap_sizes() {
        let (client, device) = cpu_setup();
        let wave: Vec<f32> = (0..1001).map(|i| (i as f32).sin()).collect();
        let x = Tensor::<CpuRuntime>::from_slice(&wave, &[1001, 1], &device).expect("x");
        let y: Vec<f32> = resize_bicubic(&client, &x, 0, 1024)
            .expect("resize")
            .to_vec();
        for (j, want) in [
            (0, 0.0),
            (1, 0.825_875),
            (2, 0.931_757_75),
            (517, 0.402_959_64),
            (1023, 0.826_879_56),
        ] {
            assert!(
                (y[j] - want).abs() < 1e-6,
                "output {j}: {} against {want}",
                y[j]
            );
        }
    }
}
