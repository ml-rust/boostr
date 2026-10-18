//! One LSTM time step from tracked `Var` ops.
//!
//! `gates = xw_t + h @ W_hh^T + b_hh`, split at `H` into `[i, f, g, o]`.
//! Then `c' = f * c + i * g` and `h' = o * tanh(c')`, with
//! `i, f, o = sigmoid(.)` and `g = tanh(.)`.

use crate::error::{Error, Result};
use numr::autograd::{Var, var_add, var_matmul, var_mul, var_narrow, var_sigmoid, var_tanh};
use numr::dtype::DType;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

/// Recurrent weights of one direction, prepared once per sequence.
pub(super) struct Recurrence<'a, R: Runtime> {
    /// `W_hh^T`, `[H, 4H]`.
    pub(super) w_hh_t: &'a Var<R>,
    /// `b_hh`, `[4H]`, if the layer has bias.
    pub(super) b_hh: Option<&'a Var<R>>,
    pub(super) hidden: usize,
}

/// Advance `(h, c)` by one step.
///
/// * `xw_t` — `[B, 4H]`, the input projection `x_t @ W_ih^T + b_ih`.
/// * `h`, `c` — `[B, H]` state entering the step.
///
/// Returns `(h', c')`, both `[B, H]`.
pub(super) fn lstm_cell_step<R, C>(
    client: &C,
    rec: &Recurrence<'_, R>,
    xw_t: &Var<R>,
    h: &Var<R>,
    c: &Var<R>,
) -> Result<(Var<R>, Var<R>)>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let hh = var_matmul(h, rec.w_hh_t, client).map_err(Error::Numr)?;
    let hh = match rec.b_hh {
        Some(b) => var_add(&hh, b, client).map_err(Error::Numr)?,
        None => hh,
    };
    let gates = var_add(xw_t, &hh, client).map_err(Error::Numr)?;

    let hs = rec.hidden;
    let gate = |k: usize| var_narrow(&gates, 1, k * hs, hs).map_err(Error::Numr);
    let i = var_sigmoid(&gate(0)?, client).map_err(Error::Numr)?;
    let f = var_sigmoid(&gate(1)?, client).map_err(Error::Numr)?;
    let g = var_tanh(&gate(2)?, client).map_err(Error::Numr)?;
    let o = var_sigmoid(&gate(3)?, client).map_err(Error::Numr)?;

    let fc = var_mul(&f, c, client).map_err(Error::Numr)?;
    let ig = var_mul(&i, &g, client).map_err(Error::Numr)?;
    let c_next = var_add(&fc, &ig, client).map_err(Error::Numr)?;
    let c_tanh = var_tanh(&c_next, client).map_err(Error::Numr)?;
    let h_next = var_mul(&o, &c_tanh, client).map_err(Error::Numr)?;
    Ok((h_next, c_next))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::lstm::Lstm;
    use crate::test_utils::cpu_setup;
    use numr::autograd::var_transpose;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    fn tensor(n: usize, salt: usize, shape: &[usize], d: &CpuDevice) -> Tensor<CpuRuntime> {
        let v: Vec<f32> = (0..n)
            .map(|k| (((k * 37 + salt * 11) % 23) as f32 / 23.0 - 0.5) * 0.8)
            .collect();
        Tensor::<CpuRuntime>::from_slice(&v, shape, d).unwrap()
    }

    /// One composed step equals the inference `Lstm::step`.
    #[test]
    fn step_matches_inference_lstm() {
        let (client, device) = cpu_setup();
        let (b, i, h) = (2, 3, 4);
        let w_ih = tensor(4 * h * i, 1, &[4 * h, i], &device);
        let w_hh = tensor(4 * h * h, 2, &[4 * h, h], &device);
        let b_ih = tensor(4 * h, 3, &[4 * h], &device);
        let b_hh = tensor(4 * h, 4, &[4 * h], &device);
        let x = tensor(b * i, 5, &[b, i], &device);
        let h0 = tensor(b * h, 6, &[b, h], &device);
        let c0 = tensor(b * h, 7, &[b, h], &device);

        let oracle = Lstm::new(w_ih.clone(), w_hh.clone(), b_ih.clone(), b_hh.clone()).unwrap();
        let (want_h, want_c) = oracle.step(&client, &x, &h0, &c0).unwrap();

        let leaf = |t: &Tensor<CpuRuntime>| Var::new(t.clone(), false);
        let w_ih_t = var_transpose(&leaf(&w_ih)).unwrap();
        let xw = var_matmul(&leaf(&x), &w_ih_t, &client).unwrap();
        let xw = var_add(&xw, &leaf(&b_ih), &client).unwrap();
        let w_hh_t = var_transpose(&leaf(&w_hh)).unwrap();
        let b_hh_v = leaf(&b_hh);
        let rec = Recurrence {
            w_hh_t: &w_hh_t,
            b_hh: Some(&b_hh_v),
            hidden: h,
        };
        let (got_h, got_c) = lstm_cell_step(&client, &rec, &xw, &leaf(&h0), &leaf(&c0)).unwrap();

        for (got, want) in [(got_h.tensor(), &want_h), (got_c.tensor(), &want_c)] {
            let got: Vec<f32> = got.contiguous().unwrap().to_vec();
            let want: Vec<f32> = want.contiguous().unwrap().to_vec();
            for (a, w) in got.iter().zip(&want) {
                assert!((a - w).abs() < 1e-5, "{a} vs {w}");
            }
        }
    }
}
