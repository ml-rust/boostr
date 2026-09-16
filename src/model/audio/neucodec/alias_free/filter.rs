//! Kaiser-windowed sinc filter construction and replicate ("edge") padding —
//! the two building blocks [`super::resample::UpSample1d`] and
//! [`super::resample::DownSample1d`] compose into the resamplers.

use crate::error::{Error, Result};
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::nn::var_contiguous;
use numr::autograd::{Var, var_broadcast_to, var_cat, var_narrow};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// `sin(pi x) / (pi x)`, with the removable singularity at 0 filled in.
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let pix = std::f64::consts::PI * x;
        pix.sin() / pix
    }
}

/// Zeroth-order modified Bessel function of the first kind, used by the Kaiser
/// window. Series expansion; converges quickly for the `beta` values here.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half_x_sq = (x / 2.0) * (x / 2.0);
    for k in 1..64 {
        term *= half_x_sq / ((k * k) as f64);
        sum += term;
        if term < 1e-18 * sum {
            break;
        }
    }
    sum
}

/// Symmetric (`periodic = false`) Kaiser window, matching
/// `torch.kaiser_window(n, beta=beta, periodic=False)`.
fn kaiser_window(n: usize, beta: f64) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    let denom = bessel_i0(beta);
    let n_minus_1 = (n - 1) as f64;
    (0..n)
        .map(|i| {
            let r = 2.0 * (i as f64) / n_minus_1 - 1.0;
            bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / denom
        })
        .collect()
}

/// Kaiser-windowed sinc low-pass, normalized to unit sum.
///
/// Port of the reference NeuCodec implementation's `kaiser_sinc_filter1d`. `cutoff` and `half_width` are in
/// cycles/sample; `kernel_size` is even in every use here.
pub fn kaiser_sinc_filter1d(cutoff: f64, half_width: f64, kernel_size: usize) -> Vec<f32> {
    let half_size = kernel_size / 2;
    let delta_f = 4.0 * half_width;
    let a = 2.285 * ((half_size as f64) - 1.0) * std::f64::consts::PI * delta_f + 7.95;
    let beta = if a > 50.0 {
        0.1102 * (a - 8.7)
    } else if a >= 21.0 {
        0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0)
    } else {
        0.0
    };
    let window = kaiser_window(kernel_size, beta);

    if cutoff == 0.0 {
        return vec![0.0; kernel_size];
    }

    let even = kernel_size.is_multiple_of(2);
    let taps: Vec<f64> = (0..kernel_size)
        .map(|i| {
            let time = if even {
                (i as f64) - (half_size as f64) + 0.5
            } else {
                (i as f64) - (half_size as f64)
            };
            2.0 * cutoff * window[i] * sinc(2.0 * cutoff * time)
        })
        .collect();
    let sum: f64 = taps.iter().sum();
    taps.iter().map(|t| (t / sum) as f32).collect()
}

/// Replicate ("edge") padding along the last axis of a `[B, C, T]` tensor.
///
/// numr has no replicate `PaddingMode` (its `pad` fills a constant), and this
/// is a composition rather than a kernel, so it is built from `narrow` +
/// `broadcast_to` + `cat` — which keeps it tracked and backend-generic instead
/// of forcing a CPU round-trip.
///
/// The edge blocks are BROADCAST to their full width and concatenated as one
/// tensor each, rather than pushing `left` (then `right`) copies of a
/// single-frame slice into the `cat` list. Both produce identical values, but
/// CUDA's `cat` launches one kernel PER INPUT, so the naive form costs
/// `left + right + 1` launches. The alias-free activations call this twice per
/// `Activation1d` and the acoustic encoder holds ~36 of them, which turned edge
/// replication alone into ~800 launches per forward. This form is always 3.
pub fn replicate_pad_1d<R, C>(client: &C, x: &Var<R>, left: usize, right: usize) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: NeuCodecClient<R>,
    R::Client: NeuCodecClient<R>,
{
    if left == 0 && right == 0 {
        return Ok(x.alias());
    }
    let shape = x.shape().to_vec();
    if shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("expected [B, C, T], got {shape:?}"),
        });
    }
    let t = shape[2];
    if t == 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "T must be > 0 to replicate an edge".into(),
        });
    }

    // At most three: [left edge block] ++ [x] ++ [right edge block].
    let mut parts: Vec<Var<R>> = Vec::with_capacity(3);
    let edge_block = |offset: usize, width: usize| -> Result<Var<R>> {
        let edge = var_narrow(x, 2, offset, 1).map_err(Error::Numr)?;
        let block = var_broadcast_to(&edge, &[shape[0], shape[1], width]).map_err(Error::Numr)?;
        // `cat` reads its inputs; materialize the broadcast view so the stride-0
        // time axis never reaches a kernel that assumes contiguity.
        var_contiguous(&block)
    };
    if left > 0 {
        parts.push(edge_block(0, left)?);
    }
    parts.push(x.alias());
    if right > 0 {
        parts.push(edge_block(t - 1, right)?);
    }

    let refs: Vec<&Var<R>> = parts.iter().collect();
    let out = var_cat(&refs, 2, client).map_err(Error::Numr)?;
    var_contiguous(&out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    fn var(
        data: &[f32],
        shape: &[usize],
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Var<CpuRuntime> {
        Var::new(
            Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap(),
            false,
        )
    }

    /// The 12-tap ratio-2 filter is what both resamplers use; it must sum to 1
    /// and be symmetric, or the activation injects gain/phase error.
    #[test]
    fn kaiser_filter_is_normalized_and_symmetric() {
        let taps = kaiser_sinc_filter1d(0.25, 0.3, 12);
        assert_eq!(taps.len(), 12);
        let sum: f32 = taps.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "taps must sum to 1, got {sum}");
        for i in 0..6 {
            let (a, b) = (taps[i], taps[11 - i]);
            assert!((a - b).abs() < 1e-6, "tap {i} asymmetric: {a} vs {b}");
        }
    }

    #[test]
    fn replicate_pad_repeats_edges() {
        let (client, device) = cpu_setup();
        let x = var(&[1.0, 2.0, 3.0], &[1, 1, 3], &device);
        let out = replicate_pad_1d(&client, &x, 2, 1).unwrap();
        assert_eq!(out.shape(), &[1, 1, 6]);
        let got: Vec<f32> = out.tensor().contiguous().unwrap().to_vec();
        assert_eq!(got, vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0]);
    }

    /// The edge blocks are built by broadcasting a `[B, C, 1]` slice across
    /// TIME. With B = C = 1 that is indistinguishable from broadcasting across
    /// the wrong axis, so this checks a genuinely 3-D case: every padded column
    /// must equal its own row's edge, never another channel's or batch's.
    #[test]
    fn replicate_pad_replicates_per_batch_and_channel() {
        let (client, device) = cpu_setup();
        // [2, 3, 4], each (batch, channel) row a distinct decade.
        let data: Vec<f32> = (0..2 * 3 * 4).map(|i| i as f32).collect();
        let x = var(&data, &[2, 3, 4], &device);

        let (left, right) = (3, 2);
        let out = replicate_pad_1d(&client, &x, left, right).unwrap();
        assert_eq!(out.shape(), &[2, 3, 4 + left + right]);
        let got: Vec<f32> = out.tensor().contiguous().unwrap().to_vec();

        let width = 4 + left + right;
        for b in 0..2 {
            for c in 0..3 {
                let row = &data[(b * 3 + c) * 4..(b * 3 + c) * 4 + 4];
                let out_row = &got[(b * 3 + c) * width..(b * 3 + c) * width + width];
                for (i, v) in out_row.iter().enumerate() {
                    let want = if i < left {
                        row[0]
                    } else if i < left + 4 {
                        row[i - left]
                    } else {
                        row[3]
                    };
                    assert_eq!(
                        *v, want,
                        "batch {b} channel {c} position {i}: got {v}, want {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn replicate_pad_zero_is_identity_and_keeps_id() {
        let (client, device) = cpu_setup();
        let x = var(&[1.0, 2.0], &[1, 1, 2], &device);
        let out = replicate_pad_1d(&client, &x, 0, 0).unwrap();
        assert_eq!(out.id(), x.id(), "no-op pad must alias, not clone");
    }
}
