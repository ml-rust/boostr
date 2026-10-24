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
//!
//! # Packed sequences
//!
//! With [`SegmentMasks`], the tap that reads `d` positions back is multiplied
//! by a constant 1/0 mask. A token then never reads, and never sends gradient
//! to, a token of an earlier document. `None` runs the plain conv.

use super::var_causal_conv1d_check::{check_segments, check_shapes};
use crate::error::{Error, Result};
use crate::nn::segment_masks::SegmentMasks;
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
/// [`Error::InvalidArgument`] when a shape disagrees with the layout above,
/// `seq` is 0, or `segments` disagrees with `x` in `[batch, seq]`, dtype or
/// kernel coverage.
pub fn var_causal_conv1d_silu<R, C>(
    client: &C,
    x: &Var<R>,
    weight: &Var<R>,
    bias: Option<&Var<R>>,
    apply_silu: bool,
    segments: Option<&SegmentMasks<R>>,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + ShapeOps<R> + ActivationOps<R> + ScalarOps<R>,
    R::Client: TensorOps<R> + ShapeOps<R> + ActivationOps<R> + ScalarOps<R>,
{
    let (batch, seq, channels, kernel) = check_shapes(x, weight, bias)?;
    if let Some(masks) = segments {
        check_segments(masks, batch, seq, kernel, x.tensor().dtype())?;
    }

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
        let mut shifted = var_narrow(&padded, 1, k, seq).map_err(Error::Numr)?;
        let back = kernel - 1 - k;
        if let (Some(masks), true) = (segments, back > 0) {
            let mask = Var::new(masks.tap(back)?.clone(), false);
            shifted = var_mul(&shifted, &mask, client).map_err(Error::Numr)?;
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::conv::causal_conv1d;
    use crate::test_utils::{Lcg, cpu_setup, max_abs_diff_f64, to_f64};
    use numr::autograd::{backward, var_sum};
    use numr::ops::BinaryOps;
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
            let got =
                var_causal_conv1d_silu(&client, &xv, &wv, Some(&bv), apply_silu, None).unwrap();
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
        let y = var_causal_conv1d_silu(client, &xv, &wv, Some(&bv), true, None).unwrap();
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
        let y = var_causal_conv1d_silu(&client, &xv, &wv, Some(&bv), true, None).unwrap();
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
        let y = var_causal_conv1d_silu(&client, &x, &w, None, false, None).unwrap();
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
        let y1 = var_causal_conv1d_silu(&client, &x1, &w, None, false, None).unwrap();
        assert_eq!(y1.shape(), &[1, 1, 1]);
        assert_eq!(y1.tensor().to_vec::<f32>(), vec![40.0]);

        let x2 = var32(&[10.0, 20.0], &[1, 2, 1], &device, false);
        let y2 = var_causal_conv1d_silu(&client, &x2, &w, None, false, None).unwrap();
        assert_eq!(y2.shape(), &[1, 2, 1]);
        // t=0: 4*10. t=1: 3*10 + 4*20.
        assert_eq!(y2.tensor().to_vec::<f32>(), vec![40.0, 110.0]);
    }

    #[test]
    fn rejects_bad_shapes() {
        let (client, device) = cpu_setup();
        let x = var32(&[0.0; 6], &[1, 3, 2], &device, false);
        let w = var32(&[0.0; 6], &[3, 1, 2], &device, false);
        assert!(var_causal_conv1d_silu(&client, &x, &w, None, false, None).is_err());
        let w = var32(&[0.0; 4], &[2, 1, 2], &device, false);
        let b = var32(&[0.0; 3], &[3], &device, false);
        assert!(var_causal_conv1d_silu(&client, &x, &w, Some(&b), false, None).is_err());
    }

    const CH: usize = 3;

    fn mk(data: &[f64], shape: &[usize], dtype: DType, device: &CpuDevice) -> Var<CpuRuntime> {
        let t = if dtype == DType::F64 {
            Tensor::<CpuRuntime>::from_slice(data, shape, device)
        } else {
            let d: Vec<f32> = data.iter().map(|v| *v as f32).collect();
            Tensor::<CpuRuntime>::from_slice(&d, shape, device)
        };
        Var::new(t.unwrap(), true)
    }

    fn rand_vec(n: usize, seed: u64) -> Vec<f64> {
        let mut g = Lcg(seed);
        (0..n)
            .map(|_| f64::from(g.uniform_f32(-1.0, 1.0)))
            .collect()
    }

    fn ids_for(lens: &[usize]) -> Vec<i32> {
        let mut ids = Vec::new();
        for (i, &n) in lens.iter().enumerate() {
            ids.extend(std::iter::repeat_n(i as i32 * 3 + 1, n));
        }
        ids
    }

    struct Out {
        y: Vec<f64>,
        gx: Vec<f64>,
        gw: Vec<f64>,
        gb: Vec<f64>,
    }

    /// One row of `x.len() / CH` tokens. Loss is the sum of squared outputs.
    fn run(dtype: DType, x: &[f64], w: &[f64], b: &[f64], ids: Option<&[i32]>) -> Out {
        let (client, device) = cpu_setup();
        let (seq, kernel) = (x.len() / CH, w.len() / CH);
        let masks = ids.map(|ids| {
            let t = Tensor::<CpuRuntime>::from_slice(ids, &[1, seq], &device).unwrap();
            SegmentMasks::from_ids(&client, &t, kernel, dtype).unwrap()
        });
        let xv = mk(x, &[1, seq, CH], dtype, &device);
        let wv = mk(w, &[CH, 1, kernel], dtype, &device);
        let bv = mk(b, &[CH], dtype, &device);
        let y = var_causal_conv1d_silu(&client, &xv, &wv, Some(&bv), true, masks.as_ref()).unwrap();
        let sq = var_mul(&y, &y, &client).unwrap();
        let loss = var_sum(&sq, &[0, 1, 2], false, &client).unwrap();
        let g = backward(&loss, &client).unwrap();
        let get = |v: &Var<CpuRuntime>| to_f64(g.get(v.id()).unwrap());
        Out {
            y: to_f64(y.tensor()),
            gx: get(&xv),
            gw: get(&wv),
            gb: get(&bv),
        }
    }

    /// Packed documents match separate runs in output, and in x, weight and
    /// bias gradients (summed over documents for the parameters).
    fn check_packed_vs_separate(dtype: DType, kernel: usize, lens: &[usize]) {
        let seq: usize = lens.iter().sum();
        let x = rand_vec(seq * CH, 1);
        let w = rand_vec(CH * kernel, 2);
        let b = rand_vec(CH, 3);
        let packed = run(dtype, &x, &w, &b, Some(&ids_for(lens)));
        let (mut y, mut gx) = (Vec::new(), Vec::new());
        let (mut gw, mut gb) = (vec![0.0; w.len()], vec![0.0; CH]);
        let mut at = 0;
        for &n in lens {
            let o = run(dtype, &x[at * CH..(at + n) * CH], &w, &b, None);
            y.extend(o.y);
            gx.extend(o.gx);
            gw.iter_mut().zip(&o.gw).for_each(|(a, v)| *a += v);
            gb.iter_mut().zip(&o.gb).for_each(|(a, v)| *a += v);
            at += n;
        }
        let tol = if dtype == DType::F64 { 1e-12 } else { 1e-5 };
        let tag = format!("{dtype:?} kernel={kernel} lens={lens:?}");
        assert!(max_abs_diff_f64(&packed.y, &y) < tol, "y {tag}");
        assert!(max_abs_diff_f64(&packed.gx, &gx) < tol, "gx {tag}");
        assert!(max_abs_diff_f64(&packed.gw, &gw) < tol, "gw {tag}");
        assert!(max_abs_diff_f64(&packed.gb, &gb) < tol, "gb {tag}");
    }

    #[test]
    fn packed_equals_separate_runs() {
        for dtype in [DType::F32, DType::F64] {
            for kernel in 2..=4 {
                check_packed_vs_separate(dtype, kernel, &[3, 4]);
            }
        }
    }

    /// Single-token documents and documents shorter than `kernel - 1`.
    #[test]
    fn short_and_single_token_documents() {
        for dtype in [DType::F32, DType::F64] {
            for kernel in 2..=4 {
                check_packed_vs_separate(dtype, kernel, &[1, 2, 1, 3, 1]);
                check_packed_vs_separate(dtype, kernel, &[1]);
            }
        }
    }

    /// Changing document A leaves the outputs of document B bit-identical.
    #[test]
    fn earlier_document_does_not_leak() {
        for dtype in [DType::F32, DType::F64] {
            for kernel in 2..=4 {
                let ids = ids_for(&[3, 4]);
                let x = rand_vec(7 * CH, 1);
                let mut x2 = x.clone();
                x2[..3 * CH].copy_from_slice(&rand_vec(3 * CH, 99));
                let w = rand_vec(CH * kernel, 2);
                let b = rand_vec(CH, 3);
                let o1 = run(dtype, &x, &w, &b, Some(&ids));
                let o2 = run(dtype, &x2, &w, &b, Some(&ids));
                assert_eq!(o1.y[3 * CH..], o2.y[3 * CH..], "{dtype:?} kernel={kernel}");
                assert_ne!(o1.y[..3 * CH], o2.y[..3 * CH]);
            }
        }
    }

    /// `segments: None` equals a tensor-level reference in the same op order.
    #[test]
    fn none_path_is_bit_identical_to_reference() {
        let (client, device) = cpu_setup();
        let (batch, seq, kernel) = (2usize, 6usize, 4usize);
        for dtype in [DType::F32, DType::F64] {
            let x = mk(
                &rand_vec(batch * seq * CH, 5),
                &[batch, seq, CH],
                dtype,
                &device,
            );
            let w = mk(&rand_vec(CH * kernel, 6), &[CH, 1, kernel], dtype, &device);
            let b = mk(&rand_vec(CH, 7), &[CH], dtype, &device);
            let got = var_causal_conv1d_silu(&client, &x, &w, Some(&b), true, None).unwrap();

            let zeros = Tensor::<CpuRuntime>::zeros(&[batch, kernel - 1, CH], dtype, &device);
            let padded = client.cat(&[&zeros.unwrap(), x.tensor()], 1).unwrap();
            let mut acc: Option<Tensor<CpuRuntime>> = None;
            for k in 0..kernel {
                let sh = padded.narrow(1, k, seq).unwrap().contiguous().unwrap();
                let wk = w.tensor().narrow(2, k, 1).unwrap().contiguous().unwrap();
                let term = client.mul(&sh, &wk.reshape(&[1, 1, CH]).unwrap()).unwrap();
                acc = Some(match acc {
                    Some(s) => client.add(&s, &term).unwrap(),
                    None => term,
                });
            }
            let bias = b.tensor().reshape(&[1, 1, CH]).unwrap();
            let want = client.add(&acc.unwrap(), &bias).unwrap();
            let want = client.silu(&want).unwrap();
            assert_eq!(to_f64(got.tensor()), to_f64(&want), "{dtype:?}");
        }
    }

    #[test]
    fn rejects_mismatched_segments() {
        let (client, device) = cpu_setup();
        let x = mk(&[0.5; 6], &[1, 3, 2], DType::F32, &device);
        let w = mk(&[0.5; 6], &[2, 1, 3], DType::F32, &device);
        let build = |ids: &[i32], seq: usize, kernel: usize, dtype: DType| {
            let t = Tensor::<CpuRuntime>::from_slice(ids, &[1, seq], &device).unwrap();
            SegmentMasks::from_ids(&client, &t, kernel, dtype).unwrap()
        };
        let run = |m: &SegmentMasks<CpuRuntime>| {
            var_causal_conv1d_silu(&client, &x, &w, None, false, Some(m))
        };
        assert!(run(&build(&[0, 0, 1], 3, 3, DType::F32)).is_ok());
        assert!(run(&build(&[0, 0], 2, 3, DType::F32)).is_err());
        assert!(run(&build(&[0, 0, 1], 3, 2, DType::F32)).is_err());
        assert!(run(&build(&[0, 0, 1], 3, 3, DType::F64)).is_err());
    }
}
