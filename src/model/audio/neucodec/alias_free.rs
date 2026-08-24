//! Alias-free activation primitives for NeuCodec's acoustic encoder
//! (BigCodec lineage, ultimately StyleGAN3's anti-aliased nonlinearity).
//!
//! A pointwise nonlinearity applied at the signal's own rate creates harmonics
//! above Nyquist that fold back as aliasing. `Activation1d` avoids that by
//! upsampling ×2, applying the nonlinearity at the higher rate, and filtering
//! back down:
//!
//! ```text
//! x -> UpSample1d(2) -> SnakeBeta -> DownSample1d(2) -> y   (same length as x)
//! ```
//!
//! Both resamplers use a 12-tap Kaiser-windowed sinc. **The filter taps are NOT
//! in the checkpoint** — upstream registers them as non-persistent buffers, so
//! they must be recomputed here exactly, or every activation in the encoder is
//! subtly wrong.
//!
//! Everything is composed from tracked `var_*` ops, so the encoder stays
//! differentiable (needed for any future codec finetune) and backend-generic.

use crate::error::{Error, Result};
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::nn::var_contiguous;
use numr::autograd::{
    Var, var_add, var_broadcast_to, var_cat, var_mul, var_mul_scalar, var_narrow, var_reshape,
};
use numr::dtype::DType;
use numr::ops::PaddingMode;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Resampling ratio used by every `Activation1d` in this encoder.
pub const RESAMPLE_RATIO: usize = 2;
/// Filter length used by both the up- and down-sampler.
pub const RESAMPLE_KERNEL_SIZE: usize = 12;
/// `1e-9` guard upstream adds to `beta` before dividing.
const SNAKE_EPS: f64 = 1e-9;

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
/// Port of upstream `kaiser_sinc_filter1d`. `cutoff` and `half_width` are in
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

/// SnakeBeta: `x + (1 / (exp(beta) + 1e-9)) * sin(x * exp(alpha))^2`.
///
/// `alpha`/`beta` are per-channel `[C]` and stored in LOG scale in this
/// checkpoint (`alpha_logscale=True`), so both are exponentiated first. Using
/// them directly would be a silent, plausible-looking error.
pub struct SnakeBeta<R: Runtime> {
    alpha: Var<R>,
    beta: Var<R>,
}

impl<R: Runtime<DType = DType>> SnakeBeta<R> {
    /// `alpha`/`beta`: `[channels]`, log-scale.
    pub fn new(alpha: Tensor<R>, beta: Tensor<R>, trainable: bool) -> Result<Self> {
        if alpha.shape().len() != 1 || alpha.shape() != beta.shape() {
            return Err(Error::InvalidArgument {
                arg: "alpha/beta",
                reason: format!(
                    "both must be 1-D and equal length, got {:?} and {:?}",
                    alpha.shape(),
                    beta.shape()
                ),
            });
        }
        Ok(Self {
            alpha: Var::new(alpha, trainable),
            beta: Var::new(beta, trainable),
        })
    }

    pub fn channels(&self) -> usize {
        self.alpha.shape()[0]
    }

    /// `x [B, C, T] -> [B, C, T]`.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let shape = x.shape().to_vec();
        if shape.len() != 3 || shape[1] != self.channels() {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!("expected [B, {}, T], got {shape:?}", self.channels()),
            });
        }

        // [C] -> [1, C, 1] so it broadcasts across batch and time.
        let a = var_reshape(&self.alpha, &[1, self.channels(), 1]).map_err(Error::Numr)?;
        let b = var_reshape(&self.beta, &[1, self.channels(), 1]).map_err(Error::Numr)?;
        let a = numr::autograd::var_exp(&a, client).map_err(Error::Numr)?;
        let b = numr::autograd::var_exp(&b, client).map_err(Error::Numr)?;

        let scaled = var_mul(x, &a, client).map_err(Error::Numr)?;
        let s = numr::autograd::var_sin(&scaled, client).map_err(Error::Numr)?;
        let s2 = var_mul(&s, &s, client).map_err(Error::Numr)?;

        let b_eps = numr::autograd::var_add_scalar(&b, SNAKE_EPS, client).map_err(Error::Numr)?;
        let recip = numr::autograd::var_div(&s2, &b_eps, client).map_err(Error::Numr)?;
        var_add(x, &recip, client).map_err(Error::Numr)
    }
}

/// ×2 upsampler: replicate-pad, grouped transposed convolution with the Kaiser
/// filter, scale by the ratio, then crop.
pub struct UpSample1d<R: Runtime> {
    filter: Tensor<R>,
    ratio: usize,
    pad: usize,
    pad_left: usize,
    pad_right: usize,
}

impl<R: Runtime<DType = DType>> UpSample1d<R> {
    pub fn new(ratio: usize, kernel_size: usize, device: &R::Device) -> Result<Self> {
        if ratio == 0 || kernel_size < ratio {
            return Err(Error::InvalidArgument {
                arg: "ratio/kernel_size",
                reason: format!("ratio must be > 0 and <= kernel_size, got {ratio}/{kernel_size}"),
            });
        }
        let taps = kaiser_sinc_filter1d(0.5 / ratio as f64, 0.6 / ratio as f64, kernel_size);
        let pad = kernel_size / ratio - 1;
        Ok(Self {
            filter: Tensor::from_slice(&taps, &[1, 1, kernel_size], device)?,
            ratio,
            pad,
            pad_left: pad * ratio + (kernel_size - ratio) / 2,
            pad_right: pad * ratio + (kernel_size - ratio).div_ceil(2),
        })
    }

    /// `x [B, C, T] -> [B, C, T * ratio]`.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let channels = x.shape()[1];
        let padded = replicate_pad_1d(client, x, self.pad, self.pad)?;

        // Depthwise: one filter per channel, groups = C.
        let weight = self
            .filter
            .broadcast_to(&[channels, 1, self.filter.shape()[2]])
            .map_err(Error::Numr)?
            .contiguous()?;
        let weight = Var::new(weight, false);

        let up = numr::autograd::var_conv_transpose1d(
            &padded,
            &weight,
            None,
            self.ratio,
            PaddingMode::Valid,
            0,
            1,
            channels,
            client,
        )
        .map_err(Error::Numr)?;
        let up = var_mul_scalar(&up, self.ratio as f64, client).map_err(Error::Numr)?;

        let total = up.shape()[2];
        if total <= self.pad_left + self.pad_right {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!(
                    "upsampled length {total} is too short to crop {}+{}",
                    self.pad_left, self.pad_right
                ),
            });
        }
        let keep = total - self.pad_left - self.pad_right;
        let out = var_narrow(&up, 2, self.pad_left, keep).map_err(Error::Numr)?;
        var_contiguous(&out)
    }
}

/// ×2 downsampler: replicate-pad, then a strided grouped convolution with the
/// same Kaiser filter.
pub struct DownSample1d<R: Runtime> {
    filter: Tensor<R>,
    ratio: usize,
    pad_left: usize,
    pad_right: usize,
}

impl<R: Runtime<DType = DType>> DownSample1d<R> {
    pub fn new(ratio: usize, kernel_size: usize, device: &R::Device) -> Result<Self> {
        if ratio == 0 {
            return Err(Error::InvalidArgument {
                arg: "ratio",
                reason: "must be > 0".into(),
            });
        }
        let taps = kaiser_sinc_filter1d(0.5 / ratio as f64, 0.6 / ratio as f64, kernel_size);
        let even = kernel_size.is_multiple_of(2);
        Ok(Self {
            filter: Tensor::from_slice(&taps, &[1, 1, kernel_size], device)?,
            ratio,
            pad_left: kernel_size / 2 - usize::from(even),
            pad_right: kernel_size / 2,
        })
    }

    /// `x [B, C, T] -> [B, C, ceil(T / ratio)]`.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let channels = x.shape()[1];
        let padded = replicate_pad_1d(client, x, self.pad_left, self.pad_right)?;

        let weight = self
            .filter
            .broadcast_to(&[channels, 1, self.filter.shape()[2]])
            .map_err(Error::Numr)?
            .contiguous()?;
        let weight = Var::new(weight, false);

        let out = numr::autograd::var_conv1d(
            &padded,
            &weight,
            None,
            self.ratio,
            PaddingMode::Valid,
            1,
            channels,
            client,
        )
        .map_err(Error::Numr)?;
        var_contiguous(&out)
    }
}

/// Anti-aliased nonlinearity: upsample -> SnakeBeta -> downsample.
pub struct Activation1d<R: Runtime> {
    up: UpSample1d<R>,
    act: SnakeBeta<R>,
    down: DownSample1d<R>,
}

impl<R: Runtime<DType = DType>> Activation1d<R> {
    pub fn new(act: SnakeBeta<R>, device: &R::Device) -> Result<Self> {
        Ok(Self {
            up: UpSample1d::new(RESAMPLE_RATIO, RESAMPLE_KERNEL_SIZE, device)?,
            act,
            down: DownSample1d::new(RESAMPLE_RATIO, RESAMPLE_KERNEL_SIZE, device)?,
        })
    }

    pub fn activation(&self) -> &SnakeBeta<R> {
        &self.act
    }

    /// The upsample stage alone.
    ///
    /// Exposed so parity tests can localize a mismatch to the resampler rather
    /// than the activation — the two compose, so a single end-to-end number
    /// cannot say which half is wrong.
    pub fn upsample_for_test<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        self.up.forward(client, x)
    }

    /// `x [B, C, T] -> [B, C, T]` (length preserved).
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let up = self.up.forward(client, x)?;
        let act = self.act.forward(client, &up)?;
        self.down.forward(client, &act)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

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

    /// alpha/beta are LOG-scale here. With both zero, `exp(0) = 1`, so
    /// SnakeBeta reduces to the textbook `x + sin^2(x)`.
    #[test]
    fn snake_beta_uses_log_scale_parameters() {
        let (client, device) = cpu_setup();
        let zeros = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2], &device).unwrap();
        let snake = SnakeBeta::new(zeros.clone(), zeros, false).unwrap();
        let xs = [0.5f32, -1.25, 2.0, 0.0];
        let x = var(&xs, &[1, 2, 2], &device);
        let out = snake.forward(&client, &x).unwrap();
        let got: Vec<f32> = out.tensor().contiguous().unwrap().to_vec();
        for (g, &v) in got.iter().zip(xs.iter()) {
            let want = v + v.sin().powi(2);
            assert!(
                (g - want).abs() < 1e-5,
                "expected x + sin^2(x) = {want}, got {g}"
            );
        }
    }

    /// A non-zero log-alpha must actually change the frequency — guards against
    /// silently dropping the `exp`.
    #[test]
    fn snake_beta_alpha_scales_frequency() {
        let (client, device) = cpu_setup();
        let ln2 = std::f32::consts::LN_2; // exp(ln2) = 2
        let alpha = Tensor::<CpuRuntime>::from_slice(&[ln2], &[1], &device).unwrap();
        let beta = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device).unwrap();
        let snake = SnakeBeta::new(alpha, beta, false).unwrap();
        let x = var(&[0.7f32], &[1, 1, 1], &device);
        let got: Vec<f32> = snake
            .forward(&client, &x)
            .unwrap()
            .tensor()
            .contiguous()
            .unwrap()
            .to_vec();
        let want = 0.7f32 + (0.7f32 * 2.0).sin().powi(2);
        assert!(
            (got[0] - want).abs() < 1e-5,
            "expected {want}, got {}",
            got[0]
        );
    }

    #[test]
    fn activation1d_preserves_length() {
        let (client, device) = cpu_setup();
        let c = 3;
        let t = 16;
        let alpha = Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c], &[c], &device).unwrap();
        let beta = Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; c], &[c], &device).unwrap();
        let act = Activation1d::new(SnakeBeta::new(alpha, beta, false).unwrap(), &device).unwrap();
        let data: Vec<f32> = (0..(c * t)).map(|i| (i as f32 * 0.3).sin()).collect();
        let x = var(&data, &[1, c, t], &device);
        let out = act.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[1, c, t]);
        for v in out.tensor().contiguous().unwrap().to_vec::<f32>() {
            assert!(v.is_finite());
        }
    }

    /// Up then down with the same filter should approximately reconstruct a
    /// smooth signal — a sanity check that the crops line up.
    #[test]
    fn upsample_then_downsample_round_trips_a_constant() {
        let (client, device) = cpu_setup();
        let t = 20;
        let up = UpSample1d::<CpuRuntime>::new(2, 12, &device).unwrap();
        let down = DownSample1d::<CpuRuntime>::new(2, 12, &device).unwrap();
        let x = var(&vec![2.5f32; t], &[1, 1, t], &device);
        let u = up.forward(&client, &x).unwrap();
        assert_eq!(u.shape(), &[1, 1, 2 * t]);
        let d = down.forward(&client, &u).unwrap();
        assert_eq!(d.shape(), &[1, 1, t]);
        for v in d.tensor().contiguous().unwrap().to_vec::<f32>() {
            assert!(
                (v - 2.5).abs() < 1e-3,
                "constant signal must survive resampling, got {v}"
            );
        }
    }
}
