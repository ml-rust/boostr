//! Tracked depthwise causal conv1d with optional SiLU, in `[batch, seq, channels]`.
//!
//! Training counterpart of the inference `causal_conv1d` for a fresh sequence
//! (zero window). The conv is a sum of `kernel` shifted copies of the left
//! padded input, each scaled per channel. Every step is a tracked op, so
//! gradient reaches `x`, `weight` and `bias`. This avoids the per-tap launch
//! cost of the generic depthwise conv backward.
//!
//! # Layout
//!
//! - `x`: `[batch, seq, channels]`
//! - `weight`: `[channels, 1, kernel]`, the layout of the inference conv
//! - `bias`: `[channels]` or `None`
//!
//! Tap `k` multiplies the input at position `t + k - (kernel - 1)`, the same
//! tap order as the inference conv over `[window | x]`.

use crate::error::{Error, Result};
use crate::nn::var_ops::var_contiguous;
use numr::autograd::{Var, var_add, var_cat, var_mul, var_narrow, var_reshape, var_silu};
use numr::dtype::DType;
use numr::ops::{ActivationOps, ScalarOps, ShapeOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Run the tracked causal conv over `x`, then SiLU when `apply_silu` is set.
///
/// Returns `[batch, seq, channels]`.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when a shape disagrees with the layout above
/// or `seq` is 0.
pub fn var_causal_conv1d_silu<R, C>(
    client: &C,
    x: &Var<R>,
    weight: &Var<R>,
    bias: Option<&Var<R>>,
    apply_silu: bool,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + ShapeOps<R> + ActivationOps<R> + ScalarOps<R>,
    R::Client: TensorOps<R> + ShapeOps<R> + ActivationOps<R> + ScalarOps<R>,
{
    let (batch, seq, channels, kernel) = check_shapes(x, weight, bias)?;

    let padded = if kernel == 1 {
        x.alias()
    } else {
        let zeros = Tensor::<R>::zeros(
            &[batch, kernel - 1, channels],
            x.tensor().dtype(),
            x.tensor().device(),
        )?;
        let zeros = Var::new(zeros, false);
        var_cat(&[&zeros, x], 1, client).map_err(Error::Numr)?
    };

    let mut acc: Option<Var<R>> = None;
    for k in 0..kernel {
        let shifted = var_narrow(&padded, 1, k, seq).map_err(Error::Numr)?;
        let tap = var_narrow(weight, 2, k, 1).map_err(Error::Numr)?;
        let tap = var_contiguous(&tap)?;
        let tap = var_reshape(&tap, &[1, 1, channels]).map_err(Error::Numr)?;
        let term = var_mul(&shifted, &tap, client).map_err(Error::Numr)?;
        acc = Some(match acc {
            Some(sum) => var_add(&sum, &term, client).map_err(Error::Numr)?,
            None => term,
        });
    }
    // `check_shapes` guarantees kernel >= 1, so the loop ran at least once.
    let mut out = acc.ok_or_else(|| Error::InvalidArgument {
        arg: "weight",
        reason: "kernel must be at least 1".into(),
    })?;

    if let Some(bias) = bias {
        let bias = var_reshape(bias, &[1, 1, channels]).map_err(Error::Numr)?;
        out = var_add(&out, &bias, client).map_err(Error::Numr)?;
    }
    if apply_silu {
        out = var_silu(&out, client).map_err(Error::Numr)?;
    }
    Ok(out)
}

/// Return `(batch, seq, channels, kernel)` after checking every operand.
fn check_shapes<R: Runtime>(
    x: &Var<R>,
    weight: &Var<R>,
    bias: Option<&Var<R>>,
) -> Result<(usize, usize, usize, usize)> {
    let xs = x.shape();
    if xs.len() != 3 || xs[1] == 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("expected [batch, seq >= 1, channels], got {xs:?}"),
        });
    }
    let (batch, seq, channels) = (xs[0], xs[1], xs[2]);
    let ws = weight.shape();
    if ws.len() != 3 || ws[0] != channels || ws[1] != 1 || ws[2] == 0 {
        return Err(Error::InvalidArgument {
            arg: "weight",
            reason: format!("expected [{channels}, 1, kernel >= 1], got {ws:?}"),
        });
    }
    if let Some(bias) = bias
        && bias.shape() != [channels]
    {
        return Err(Error::InvalidArgument {
            arg: "bias",
            reason: format!("expected [{channels}], got {:?}", bias.shape()),
        });
    }
    Ok((batch, seq, channels, ws[2]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::conv::causal_conv1d;
    use crate::test_utils::cpu_setup;
    use numr::autograd::{backward, var_sum};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn var32(data: &[f32], shape: &[usize], device: &CpuDevice, grad: bool) -> Var<CpuRuntime> {
        Var::new(
            Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap(),
            grad,
        )
    }

    fn var64(data: &[f64], shape: &[usize], device: &CpuDevice, grad: bool) -> Var<CpuRuntime> {
        Var::new(
            Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap(),
            grad,
        )
    }

    fn silu(v: f32) -> f32 {
        v / (1.0 + (-v).exp())
    }

    /// Matches the inference conv with a zero window, with and without SiLU.
    #[test]
    fn matches_inference_conv_with_zero_window() {
        let (client, device) = cpu_setup();
        let (batch, seq, channels, kernel) = (2usize, 6usize, 3usize, 4usize);

        let x: Vec<f32> = (0..batch * seq * channels)
            .map(|i| ((i * 7) % 11) as f32 * 0.25 - 1.0)
            .collect();
        let w: Vec<f32> = (0..channels * kernel)
            .map(|i| ((i * 5) % 9) as f32 * 0.125 - 0.5)
            .collect();
        let b: Vec<f32> = (0..channels).map(|c| c as f32 * 0.1 - 0.05).collect();

        // Inference layout [batch, channels, seq].
        let mut x_inf = vec![0.0f32; x.len()];
        for n in 0..batch {
            for t in 0..seq {
                for c in 0..channels {
                    x_inf[(n * channels + c) * seq + t] = x[(n * seq + t) * channels + c];
                }
            }
        }
        let x_t =
            Tensor::<CpuRuntime>::from_slice(&x_inf, &[batch, channels, seq], &device).unwrap();
        let w_t = Tensor::<CpuRuntime>::from_slice(&w, &[channels, 1, kernel], &device).unwrap();
        let b_t = Tensor::<CpuRuntime>::from_slice(&b, &[channels], &device).unwrap();
        let zero = Tensor::<CpuRuntime>::zeros(&[batch, channels, kernel - 1], DType::F32, &device)
            .unwrap();
        let (want, _) = causal_conv1d(&client, &x_t, &w_t, Some(&b_t), &zero).unwrap();
        let want: Vec<f32> = want.contiguous().unwrap().to_vec();

        let xv = var32(&x, &[batch, seq, channels], &device, false);
        let wv = var32(&w, &[channels, 1, kernel], &device, false);
        let bv = var32(&b, &[channels], &device, false);
        for apply_silu in [false, true] {
            let got = var_causal_conv1d_silu(&client, &xv, &wv, Some(&bv), apply_silu).unwrap();
            assert_eq!(got.shape(), &[batch, seq, channels]);
            let got: Vec<f32> = got.tensor().contiguous().unwrap().to_vec();
            for n in 0..batch {
                for t in 0..seq {
                    for c in 0..channels {
                        let mut expect = want[(n * channels + c) * seq + t];
                        if apply_silu {
                            expect = silu(expect);
                        }
                        let have = got[(n * seq + t) * channels + c];
                        assert!(
                            (have - expect).abs() < 1e-5,
                            "silu={apply_silu} n={n} t={t} c={c}: {have} vs {expect}"
                        );
                    }
                }
            }
        }
    }

    fn loss_value(
        client: &CpuClient,
        x: &[f64],
        w: &[f64],
        b: &[f64],
        dims: [usize; 3],
        device: &CpuDevice,
    ) -> f64 {
        let [batch, seq, channels] = dims;
        let kernel = w.len() / channels;
        let xv = var64(x, &[batch, seq, channels], device, false);
        let wv = var64(w, &[channels, 1, kernel], device, false);
        let bv = var64(b, &[channels], device, false);
        let y = var_causal_conv1d_silu(client, &xv, &wv, Some(&bv), true).unwrap();
        let sq = var_mul(&y, &y, client).unwrap();
        let loss = var_sum(&sq, &[0, 1, 2], false, client).unwrap();
        loss.tensor().to_vec::<f64>()[0]
    }

    /// Analytic gradients of x, w and bias match central finite differences.
    #[test]
    fn gradients_match_finite_differences() {
        let (client, device) = cpu_setup();
        let (batch, seq, channels, kernel) = (2usize, 5usize, 3usize, 3usize);
        let x: Vec<f64> = (0..batch * seq * channels)
            .map(|i| ((i * 7) % 11) as f64 * 0.2 - 1.0)
            .collect();
        let w: Vec<f64> = (0..channels * kernel)
            .map(|i| ((i * 5) % 9) as f64 * 0.15 - 0.6)
            .collect();
        let b: Vec<f64> = (0..channels).map(|c| c as f64 * 0.1 - 0.1).collect();
        let dims = [batch, seq, channels];

        let xv = var64(&x, &[batch, seq, channels], &device, true);
        let wv = var64(&w, &[channels, 1, kernel], &device, true);
        let bv = var64(&b, &[channels], &device, true);
        let y = var_causal_conv1d_silu(&client, &xv, &wv, Some(&bv), true).unwrap();
        let sq = var_mul(&y, &y, &client).unwrap();
        let loss = var_sum(&sq, &[0, 1, 2], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let eps = 1e-6;
        let check = |id, base: &[f64], which: usize| {
            let got: Vec<f64> = grads.get(id).unwrap().contiguous().unwrap().to_vec();
            assert_eq!(got.len(), base.len());
            for i in 0..base.len() {
                let (mut hi, mut lo) = (base.to_vec(), base.to_vec());
                hi[i] += eps;
                lo[i] -= eps;
                let eval = |v: &[f64]| match which {
                    0 => loss_value(&client, v, &w, &b, dims, &device),
                    1 => loss_value(&client, &x, v, &b, dims, &device),
                    _ => loss_value(&client, &x, &w, v, dims, &device),
                };
                let fd = (eval(&hi) - eval(&lo)) / (2.0 * eps);
                assert!(
                    (got[i] - fd).abs() < 1e-5 * (1.0 + fd.abs()),
                    "input {which} idx {i}: analytic {} vs fd {fd}",
                    got[i]
                );
            }
        };
        check(xv.id(), &x, 0);
        check(wv.id(), &w, 1);
        check(bv.id(), &b, 2);
    }

    /// A frozen weight receives no gradient while x still does.
    #[test]
    fn frozen_weight_gets_no_grad() {
        let (client, device) = cpu_setup();
        let x = var32(
            &[0.5, -1.0, 2.0, 0.25, 1.5, -0.5],
            &[1, 3, 2],
            &device,
            true,
        );
        let w = var32(
            &[1.0, 0.5, -0.25, 0.75, 0.1, 0.2],
            &[2, 1, 3],
            &device,
            false,
        );
        let y = var_causal_conv1d_silu(&client, &x, &w, None, false).unwrap();
        let loss = var_sum(&y, &[0, 1, 2], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();
        assert!(grads.get(x.id()).is_some());
        assert!(grads.get(w.id()).is_none());
    }

    /// `seq < kernel - 1` and `seq == 1` keep the output shape and match hand values.
    #[test]
    fn short_sequences() {
        let (client, device) = cpu_setup();
        // One channel, kernel 4, taps [1, 2, 3, 4]. out[t] = sum_k w[k] * xpad[t + k].
        let w = var32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4], &device, false);

        let x1 = var32(&[10.0], &[1, 1, 1], &device, false);
        let y1 = var_causal_conv1d_silu(&client, &x1, &w, None, false).unwrap();
        assert_eq!(y1.shape(), &[1, 1, 1]);
        assert_eq!(y1.tensor().to_vec::<f32>(), vec![40.0]);

        let x2 = var32(&[10.0, 20.0], &[1, 2, 1], &device, false);
        let y2 = var_causal_conv1d_silu(&client, &x2, &w, None, false).unwrap();
        assert_eq!(y2.shape(), &[1, 2, 1]);
        // t=0: 4*10. t=1: 3*10 + 4*20.
        assert_eq!(y2.tensor().to_vec::<f32>(), vec![40.0, 110.0]);
    }

    #[test]
    fn rejects_bad_shapes() {
        let (client, device) = cpu_setup();
        let x = var32(&[0.0; 6], &[1, 3, 2], &device, false);
        let w = var32(&[0.0; 6], &[3, 1, 2], &device, false);
        assert!(var_causal_conv1d_silu(&client, &x, &w, None, false).is_err());
        let w = var32(&[0.0; 4], &[2, 1, 2], &device, false);
        let b = var32(&[0.0; 3], &[3], &device, false);
        assert!(var_causal_conv1d_silu(&client, &x, &w, Some(&b), false).is_err());
    }
}
