//! One GRU time step from tracked `Var` ops, as `torch.nn.GRU` computes it.
//!
//! `hh = h @ W_hh^T + b_hh` stays apart from `xw_t`, so `b_hn` sits inside
//! the reset product. Both split at `H` into `[r, z, n]`:
//!
//! * `r = sigmoid(xw_r + hh_r)`, `z = sigmoid(xw_z + hh_z)`
//! * `n = tanh(xw_n + r * hh_n)`
//! * `h' = (1 - z) * n + z * h`, computed as `n + z * (h - n)`

use super::cell::{CellKind, GruKind, Recurrence, state_count_error};
use crate::error::{Error, Result};
use numr::autograd::{
    Var, var_add, var_matmul, var_mul, var_narrow, var_sigmoid, var_sub, var_tanh,
};
use numr::dtype::DType;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

impl CellKind for GruKind {
    const GATES: usize = 3;
    const STATES: &'static [&'static str] = &["h0"];

    fn step<R, C>(
        client: &C,
        rec: &Recurrence<'_, R>,
        xw_t: &Var<R>,
        state: &[Var<R>],
    ) -> Result<Vec<Var<R>>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let [h] = state else {
            return Err(state_count_error(1, state.len()));
        };
        Ok(vec![gru_cell_step(client, rec, xw_t, h)?])
    }
}

/// Advance `h` by one step.
///
/// * `xw_t` — `[B, 3H]`, the input projection `x_t @ W_ih^T + b_ih`.
/// * `h` — `[B, H]` state entering the step.
///
/// Returns `h'`, `[B, H]`.
pub(super) fn gru_cell_step<R, C>(
    client: &C,
    rec: &Recurrence<'_, R>,
    xw_t: &Var<R>,
    h: &Var<R>,
) -> Result<Var<R>>
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

    let hs = rec.hidden;
    let part = |v: &Var<R>, k: usize| var_narrow(v, 1, k * hs, hs).map_err(Error::Numr);
    let r_in = var_add(&part(xw_t, 0)?, &part(&hh, 0)?, client).map_err(Error::Numr)?;
    let z_in = var_add(&part(xw_t, 1)?, &part(&hh, 1)?, client).map_err(Error::Numr)?;
    let r = var_sigmoid(&r_in, client).map_err(Error::Numr)?;
    let z = var_sigmoid(&z_in, client).map_err(Error::Numr)?;

    let r_hn = var_mul(&r, &part(&hh, 2)?, client).map_err(Error::Numr)?;
    let n_in = var_add(&part(xw_t, 2)?, &r_hn, client).map_err(Error::Numr)?;
    let n = var_tanh(&n_in, client).map_err(Error::Numr)?;

    let h_minus_n = var_sub(h, &n, client).map_err(Error::Numr)?;
    let z_term = var_mul(&z, &h_minus_n, client).map_err(Error::Numr)?;
    var_add(&n, &z_term, client).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::autograd::var_transpose;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    fn vals(n: usize, salt: usize) -> Vec<f64> {
        (0..n)
            .map(|k| (((k * 37 + salt * 11) % 23) as f64 / 23.0 - 0.5) * 1.2)
            .collect()
    }

    fn t64(v: &[f64], shape: &[usize], d: &CpuDevice) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(v, shape, d).unwrap()
    }

    fn sigmoid(v: f64) -> f64 {
        1.0 / (1.0 + (-v).exp())
    }

    /// Plain-f64 GRU step: row-major `w_ih [3H, I]`, `w_hh [3H, H]`.
    fn reference(
        w_ih: &[f64],
        w_hh: &[f64],
        b_ih: &[f64],
        b_hh: &[f64],
        x: &[f64],
        h: &[f64],
        (b, i, hs): (usize, usize, usize),
    ) -> Vec<f64> {
        let mut out = vec![0.0; b * hs];
        for row in 0..b {
            let dot = |w: &[f64], v: &[f64], width: usize, gate: usize, j: usize| -> f64 {
                (0..width)
                    .map(|k| w[(gate * hs + j) * width + k] * v[row * width + k])
                    .sum()
            };
            for j in 0..hs {
                let xg = |g: usize| dot(w_ih, x, i, g, j) + b_ih[g * hs + j];
                let hg = |g: usize| dot(w_hh, h, hs, g, j) + b_hh[g * hs + j];
                let r = sigmoid(xg(0) + hg(0));
                let z = sigmoid(xg(1) + hg(1));
                let n = (xg(2) + r * hg(2)).tanh();
                out[row * hs + j] = (1.0 - z) * n + z * h[row * hs + j];
            }
        }
        out
    }

    /// One composed step equals the formulas, evaluated by a plain loop.
    #[test]
    fn step_matches_plain_formulas() {
        let (client, device) = cpu_setup();
        let (b, i, h) = (2, 3, 2);
        let w_ih = vals(3 * h * i, 1);
        let w_hh = vals(3 * h * h, 2);
        let b_ih = vals(3 * h, 3);
        let b_hh = vals(3 * h, 4);
        let x = vals(b * i, 5);
        let h0 = vals(b * h, 6);
        let want = reference(&w_ih, &w_hh, &b_ih, &b_hh, &x, &h0, (b, i, h));

        let leaf = |v: &[f64], s: &[usize]| Var::new(t64(v, s, &device), false);
        let w_ih_t = var_transpose(&leaf(&w_ih, &[3 * h, i])).unwrap();
        let xw = var_matmul(&leaf(&x, &[b, i]), &w_ih_t, &client).unwrap();
        let xw = var_add(&xw, &leaf(&b_ih, &[3 * h]), &client).unwrap();
        let w_hh_t = var_transpose(&leaf(&w_hh, &[3 * h, h])).unwrap();
        let b_hh_v = leaf(&b_hh, &[3 * h]);
        let rec = Recurrence {
            w_hh_t: &w_hh_t,
            b_hh: Some(&b_hh_v),
            hidden: h,
        };
        let got = gru_cell_step(&client, &rec, &xw, &leaf(&h0, &[b, h])).unwrap();
        assert_eq!(got.shape(), &[b, h]);
        let got: Vec<f64> = got.tensor().contiguous().unwrap().to_vec();
        for (k, (a, w)) in got.iter().zip(&want).enumerate() {
            assert!((a - w).abs() < 1e-12, "h'[{k}]: {a} vs {w}");
        }

        let next = GruKind::step(&client, &rec, &xw, &[leaf(&h0, &[b, h])]).unwrap();
        assert_eq!(next.len(), 1);
        let via_trait: Vec<f64> = next[0].tensor().contiguous().unwrap().to_vec();
        assert_eq!(via_trait, got);
        assert!(GruKind::step(&client, &rec, &xw, &[]).is_err());
    }

    /// Without bias, `b_hn` is absent and `n = tanh(xw_n + r * (h @ W_hn^T))`.
    #[test]
    fn step_without_bias_matches_plain_formulas() {
        let (client, device) = cpu_setup();
        let (b, i, h) = (1, 2, 3);
        let w_ih = vals(3 * h * i, 7);
        let w_hh = vals(3 * h * h, 8);
        let zero = vec![0.0; 3 * h];
        let x = vals(b * i, 9);
        let h0 = vals(b * h, 10);
        let want = reference(&w_ih, &w_hh, &zero, &zero, &x, &h0, (b, i, h));

        let leaf = |v: &[f64], s: &[usize]| Var::new(t64(v, s, &device), false);
        let w_ih_t = var_transpose(&leaf(&w_ih, &[3 * h, i])).unwrap();
        let xw = var_matmul(&leaf(&x, &[b, i]), &w_ih_t, &client).unwrap();
        let w_hh_t = var_transpose(&leaf(&w_hh, &[3 * h, h])).unwrap();
        let rec = Recurrence {
            w_hh_t: &w_hh_t,
            b_hh: None,
            hidden: h,
        };
        let got = gru_cell_step(&client, &rec, &xw, &leaf(&h0, &[b, h])).unwrap();
        let got: Vec<f64> = got.tensor().contiguous().unwrap().to_vec();
        for (k, (a, w)) in got.iter().zip(&want).enumerate() {
            assert!((a - w).abs() < 1e-12, "h'[{k}]: {a} vs {w}");
        }
    }
}
