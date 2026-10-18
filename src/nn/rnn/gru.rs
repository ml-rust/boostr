//! [`TrainableGru`]: a multi-layer, optionally bidirectional GRU in autograd.
//!
//! The math and weight layout equal `torch.nn.GRU`. The shared
//! [`RnnStack`] driver runs layers, directions and dropout, and each
//! time step runs the GRU cell: gates `[r, z, n]`, state `h`.

use super::cell::{GruKind, state_count_error};
use super::stack::RnnStack;
use crate::error::Result;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{RandomOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

/// Trainable GRU stack with `torch.nn.GRU` semantics and parameter names.
///
/// Build it with `TrainableGru::load`, `TrainableGru::init` or
/// `TrainableGru::from_parts`. Each weight has `3H` rows, `[r, z, n]`.
/// It starts in training mode, as PyTorch does.
pub type TrainableGru<R> = RnnStack<R, GruKind>;

impl<R: Runtime<DType = DType>> RnnStack<R, GruKind> {
    /// Run the stack over a whole sequence.
    ///
    /// * `x` — `[B, T, input_size]` with `batch_first`, else `[T, B, input_size]`.
    /// * `h0` — optional `[num_layers * dirs, B, H]`. `None` starts from zeros.
    ///
    /// Returns `(output, h_n)`. `output` is `[B, T, dirs * H]` with
    /// `batch_first`, else `[T, B, dirs * H]`. `h_n` is always
    /// `[num_layers * dirs, B, H]`. The reverse entry holds its state after t = 0.
    pub fn forward<C>(
        &self,
        client: &C,
        x: &Var<R>,
        h0: Option<&Var<R>>,
    ) -> Result<(Var<R>, Var<R>)>
    where
        C: RuntimeClient<R> + TensorOps<R> + RandomOps<R> + ScalarOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let given = h0.map(|h| [h]);
        let (output, finals) = self.run(client, x, given.as_ref().map(|s| s.as_slice()))?;
        let [h_n] = <[Var<R>; 1]>::try_from(finals).map_err(|v| state_count_error(1, v.len()))?;
        Ok((output, h_n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::module::{Module, TrainMode};
    use crate::nn::rnn::config::RnnConfig;
    use crate::nn::rnn::params::{GruCellParams, param_suffix};
    use crate::nn::var_builder::VarBuilder;
    use crate::nn::varmap::VarMap;
    use crate::test_utils::cpu_setup;
    use numr::autograd::{backward, var_add, var_mul, var_sum};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::{Tensor, TensorId};

    type Named = Vec<(String, Vec<usize>, Vec<f64>)>;

    /// Deterministic values in about `[-0.45, 0.45]`, never zero.
    fn vals(n: usize, salt: usize) -> Vec<f64> {
        (0..n)
            .map(|k| {
                let u = ((k * 29 + salt * 17 + 3) % 31) as f64 / 31.0;
                (u - 0.5) * 0.9 + 0.01
            })
            .collect()
    }

    /// Every parameter of `cfg` under its PyTorch name, with fixed values.
    fn weights(cfg: &RnnConfig) -> Named {
        let gate = 3 * cfg.hidden_size;
        let mut out = Vec::new();
        let mut salt = 0;
        for layer in 0..cfg.num_layers {
            for dir in 0..cfg.num_directions() {
                let s = param_suffix(layer, dir == 1);
                let mut shapes = vec![
                    (
                        format!("weight_ih_{s}"),
                        vec![gate, cfg.layer_input_size(layer)],
                    ),
                    (format!("weight_hh_{s}"), vec![gate, cfg.hidden_size]),
                ];
                if cfg.bias {
                    shapes.push((format!("bias_ih_{s}"), vec![gate]));
                    shapes.push((format!("bias_hh_{s}"), vec![gate]));
                }
                for (name, shape) in shapes {
                    salt += 1;
                    let n = shape.iter().product();
                    out.push((name, shape, vals(n, salt)));
                }
            }
        }
        out
    }

    fn t64(v: &[f64], shape: &[usize], d: &CpuDevice) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(v, shape, d).unwrap()
    }

    fn f64s(t: &Tensor<CpuRuntime>) -> Vec<f64> {
        t.contiguous().unwrap().to_vec()
    }

    fn build(
        cfg: &RnnConfig,
        w: &Named,
        trainable: bool,
        d: &CpuDevice,
    ) -> TrainableGru<CpuRuntime> {
        let mut map = VarMap::<CpuRuntime>::new();
        for (name, shape, v) in w {
            map.insert(name.clone(), t64(v, shape, d));
        }
        let vb = VarBuilder::new(&mut map, d);
        TrainableGru::load(&vb, cfg.clone(), trainable).unwrap()
    }

    fn x_shape(cfg: &RnnConfig, batch: usize, steps: usize) -> [usize; 3] {
        if cfg.batch_first {
            [batch, steps, cfg.input_size]
        } else {
            [steps, batch, cfg.input_size]
        }
    }

    fn state_shape(cfg: &RnnConfig, batch: usize) -> [usize; 3] {
        [
            cfg.num_layers * cfg.num_directions(),
            batch,
            cfg.hidden_size,
        ]
    }

    fn sigmoid(v: f64) -> f64 {
        1.0 / (1.0 + (-v).exp())
    }

    /// Plain-f64 GRU stack over time-major `x`, from the textbook formulas.
    ///
    /// Returns `(output [T, B, dirs * H], h_n [L * dirs, B, H])`, flat.
    fn reference(
        cfg: &RnnConfig,
        w: &Named,
        x_tm: &[f64],
        (steps, batch): (usize, usize),
        h0: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let hs = cfg.hidden_size;
        let dirs = cfg.num_directions();
        let get = |name: String| -> Vec<f64> {
            match w.iter().find(|(n, _, _)| *n == name) {
                Some((_, _, v)) => v.clone(),
                None => vec![0.0; 3 * hs],
            }
        };
        let mut cur = x_tm.to_vec();
        let mut h_n = Vec::new();
        for layer in 0..cfg.num_layers {
            let width = cfg.layer_input_size(layer);
            let mut out = vec![0.0; steps * batch * dirs * hs];
            for dir in 0..dirs {
                let s = param_suffix(layer, dir == 1);
                let (w_ih, w_hh) = (get(format!("weight_ih_{s}")), get(format!("weight_hh_{s}")));
                let (b_ih, b_hh) = (get(format!("bias_ih_{s}")), get(format!("bias_hh_{s}")));
                let slot = layer * dirs + dir;
                let mut h = h0[slot * batch * hs..(slot + 1) * batch * hs].to_vec();
                for step in 0..steps {
                    let t = if dir == 1 { steps - 1 - step } else { step };
                    let mut next = vec![0.0; batch * hs];
                    for b in 0..batch {
                        let xr = &cur[(t * batch + b) * width..(t * batch + b + 1) * width];
                        let hr = &h[b * hs..(b + 1) * hs];
                        for j in 0..hs {
                            let xg = |g: usize| -> f64 {
                                let row = &w_ih[(g * hs + j) * width..(g * hs + j + 1) * width];
                                row.iter().zip(xr).map(|(a, v)| a * v).sum::<f64>()
                                    + b_ih[g * hs + j]
                            };
                            let hg = |g: usize| -> f64 {
                                let row = &w_hh[(g * hs + j) * hs..(g * hs + j + 1) * hs];
                                row.iter().zip(hr).map(|(a, v)| a * v).sum::<f64>()
                                    + b_hh[g * hs + j]
                            };
                            let r = sigmoid(xg(0) + hg(0));
                            let z = sigmoid(xg(1) + hg(1));
                            let n = (xg(2) + r * hg(2)).tanh();
                            next[b * hs + j] = (1.0 - z) * n + z * hr[j];
                        }
                    }
                    h = next;
                    for b in 0..batch {
                        let at = (t * batch + b) * dirs * hs + dir * hs;
                        out[at..at + hs].copy_from_slice(&h[b * hs..(b + 1) * hs]);
                    }
                }
                h_n.extend(h);
            }
            cur = out;
        }
        (cur, h_n)
    }

    /// Swap the first two axes of a flat `[a, b, f]` buffer.
    fn swap01(v: &[f64], a: usize, b: usize, f: usize) -> Vec<f64> {
        let mut out = vec![0.0; v.len()];
        for i in 0..a {
            for j in 0..b {
                let src = (i * b + j) * f;
                let dst = (j * a + i) * f;
                out[dst..dst + f].copy_from_slice(&v[src..src + f]);
            }
        }
        out
    }

    fn assert_close(label: &str, got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (k, (a, b)) in got.iter().zip(want).enumerate() {
            assert!((a - b).abs() < 1e-12, "{label}[{k}]: {a} vs {b}");
        }
    }

    /// Output and `h_n` equal the plain-f64 reference stack.
    fn check_against_reference(cfg: RnnConfig, batch: usize, steps: usize) {
        let (client, device) = cpu_setup();
        let w = weights(&cfg);
        let gru = build(&cfg, &w, false, &device);
        let xs = x_shape(&cfg, batch, steps);
        let ss = state_shape(&cfg, batch);
        let x = vals(xs.iter().product(), 40);
        let h0 = vals(ss.iter().product(), 41);
        let x_tm = if cfg.batch_first {
            swap01(&x, batch, steps, cfg.input_size)
        } else {
            x.clone()
        };
        let (want_out, want_h) = reference(&cfg, &w, &x_tm, (steps, batch), &h0);
        let want_out = if cfg.batch_first {
            swap01(&want_out, steps, batch, cfg.output_size())
        } else {
            want_out
        };

        let xv = Var::new(t64(&x, &xs, &device), false);
        let hv = Var::new(t64(&h0, &ss, &device), false);
        let (out, h_n) = gru.forward(&client, &xv, Some(&hv)).unwrap();
        let mut out_shape = xs;
        out_shape[2] = cfg.output_size();
        assert_eq!(out.shape(), &out_shape);
        assert_eq!(h_n.shape(), &ss);
        assert_close("output", &f64s(out.tensor()), &want_out);
        assert_close("h_n", &f64s(h_n.tensor()), &want_h);
    }

    #[test]
    fn matches_reference_one_layer_unidirectional() {
        check_against_reference(RnnConfig::new(3, 4), 2, 5);
    }

    #[test]
    fn matches_reference_two_layer_bidirectional_batch_first() {
        let cfg = RnnConfig {
            num_layers: 2,
            bidirectional: true,
            batch_first: true,
            ..RnnConfig::new(3, 2)
        };
        check_against_reference(cfg, 2, 4);
    }

    #[test]
    fn matches_reference_without_bias() {
        let cfg = RnnConfig {
            num_layers: 2,
            bidirectional: true,
            bias: false,
            ..RnnConfig::new(2, 3)
        };
        check_against_reference(cfg, 1, 3);
    }

    /// `sum(out * Wo) + sum(h_n * Wh)` with fixed probes.
    fn loss(
        c: &CpuClient,
        gru: &TrainableGru<CpuRuntime>,
        x: &Var<CpuRuntime>,
        h0: &Var<CpuRuntime>,
    ) -> Var<CpuRuntime> {
        let (out, hn) = gru.forward(c, x, Some(h0)).unwrap();
        let mut total: Option<Var<CpuRuntime>> = None;
        for (k, v) in [out, hn].iter().enumerate() {
            let probe = Var::new(
                t64(
                    &vals(v.tensor().numel(), 100 + k),
                    v.shape(),
                    v.tensor().device(),
                ),
                false,
            );
            let dims: Vec<usize> = (0..v.shape().len()).collect();
            let s = var_sum(&var_mul(v, &probe, c).unwrap(), &dims, false, c).unwrap();
            total = Some(match total {
                None => s,
                Some(t) => var_add(&t, &s, c).unwrap(),
            });
        }
        total.unwrap()
    }

    /// A deterministic spread of up to 6 entries: first, thirds, middle, last.
    fn probes(n: usize) -> Vec<usize> {
        let mut p = vec![0, n / 3, n / 2, (2 * n) / 3, (5 * n) / 7, n - 1];
        p.sort_unstable();
        p.dedup();
        p
    }

    /// Compare `analytic` against central differences of `f` at `probes`.
    fn compare(label: &str, analytic: &[f64], base: &[f64], f: impl Fn(&[f64]) -> f64) {
        const EPS: f64 = 1e-6;
        assert_eq!(analytic.len(), base.len(), "{label}: gradient length");
        for i in probes(base.len()) {
            let (mut hi, mut lo) = (base.to_vec(), base.to_vec());
            hi[i] += EPS;
            lo[i] -= EPS;
            let numeric = (f(&hi) - f(&lo)) / (2.0 * EPS);
            let a = analytic[i];
            let scale = a.abs().max(numeric.abs()).max(1.0);
            assert!(
                (a - numeric).abs() <= 1e-6 * scale,
                "{label}[{i}]: analytic {a} vs numeric {numeric}"
            );
        }
    }

    /// Finite-difference check of every parameter, `x` and `h0`.
    fn check_gradients(cfg: RnnConfig, batch: usize, steps: usize) {
        let (client, device) = cpu_setup();
        let w = weights(&cfg);
        let xs = x_shape(&cfg, batch, steps);
        let ss = state_shape(&cfg, batch);
        let x = vals(xs.iter().product(), 50);
        let h0 = vals(ss.iter().product(), 60);

        let gru = build(&cfg, &w, true, &device);
        let xv = Var::new(t64(&x, &xs, &device), true);
        let hv = Var::new(t64(&h0, &ss, &device), true);
        let grads = backward(&loss(&client, &gru, &xv, &hv), &client).unwrap();
        let grad = |id| f64s(grads.get(id).unwrap());

        let eval = |w: &Named, x: &[f64], h0: &[f64]| {
            let gru = build(&cfg, w, false, &device);
            let leaf = |v: &[f64], s: &[usize]| Var::new(t64(v, s, &device), false);
            let l = loss(&client, &gru, &leaf(x, &xs), &leaf(h0, &ss));
            f64s(l.tensor())[0]
        };

        let named = gru.named_parameters();
        assert_eq!(named.len(), w.len());
        for (p, (name, var)) in named.iter().enumerate() {
            assert_eq!(name, &w[p].0);
            compare(name, &grad(var.id()), &w[p].2, |v| {
                let mut wp = w.clone();
                wp[p].2 = v.to_vec();
                eval(&wp, &x, &h0)
            });
        }
        compare("x", &grad(xv.id()), &x, |v| eval(&w, v, &h0));
        compare("h0", &grad(hv.id()), &h0, |v| eval(&w, &x, v));
    }

    #[test]
    fn gradients_one_layer_unidirectional() {
        check_gradients(RnnConfig::new(3, 3), 2, 4);
    }

    #[test]
    fn gradients_two_layer_bidirectional_batch_first() {
        let cfg = RnnConfig {
            num_layers: 2,
            bidirectional: true,
            batch_first: true,
            ..RnnConfig::new(3, 2)
        };
        check_gradients(cfg, 2, 3);
    }

    fn two_layer_bi(bias: bool, dropout: f64) -> RnnConfig {
        RnnConfig {
            num_layers: 2,
            bidirectional: true,
            bias,
            dropout,
            ..RnnConfig::new(3, 4)
        }
    }

    fn names(gru: &TrainableGru<CpuRuntime>) -> Vec<String> {
        gru.named_parameters().into_iter().map(|(n, _)| n).collect()
    }

    /// The key order equals `torch.nn.GRU(3, 4, 2, bidirectional=True)`.
    #[test]
    fn names_match_pytorch_two_layer_bidirectional() {
        let (_, device) = cpu_setup();
        let cfg = two_layer_bi(true, 0.0);
        let gru = build(&cfg, &weights(&cfg), true, &device);
        let want = [
            "weight_ih_l0",
            "weight_hh_l0",
            "bias_ih_l0",
            "bias_hh_l0",
            "weight_ih_l0_reverse",
            "weight_hh_l0_reverse",
            "bias_ih_l0_reverse",
            "bias_hh_l0_reverse",
            "weight_ih_l1",
            "weight_hh_l1",
            "bias_ih_l1",
            "bias_hh_l1",
            "weight_ih_l1_reverse",
            "weight_hh_l1_reverse",
            "bias_ih_l1_reverse",
            "bias_hh_l1_reverse",
        ];
        assert_eq!(names(&gru), want);
        let named = gru.named_parameters();
        assert_eq!(named[0].1.shape(), &[12, 3]);
        assert_eq!(named[1].1.shape(), &[12, 4]);
        assert_eq!(named[2].1.shape(), &[12]);
        assert_eq!(
            named[8].1.shape(),
            &[12, 8],
            "layer 1 reads both directions"
        );

        let ids: Vec<TensorId> = gru.parameters().into_iter().map(|(id, _)| id).collect();
        let named_ids: Vec<TensorId> = named.iter().map(|(_, v)| v.id()).collect();
        assert_eq!(ids, named_ids);
        assert_eq!(gru.trainable_parameters().len(), 16);
        assert_eq!(Module::parameters(&gru).len(), 16);
        assert_eq!(Module::named_parameters(&gru).len(), 16);
    }

    #[test]
    fn names_without_bias_match_pytorch() {
        let (_, device) = cpu_setup();
        let cfg = two_layer_bi(false, 0.0);
        let gru = build(&cfg, &weights(&cfg), false, &device);
        let want = [
            "weight_ih_l0",
            "weight_hh_l0",
            "weight_ih_l0_reverse",
            "weight_hh_l0_reverse",
            "weight_ih_l1",
            "weight_hh_l1",
            "weight_ih_l1_reverse",
            "weight_hh_l1_reverse",
        ];
        assert_eq!(names(&gru), want);
        assert!(gru.trainable_parameters().is_empty());
    }

    fn run(client: &CpuClient, gru: &TrainableGru<CpuRuntime>, x: &Tensor<CpuRuntime>) -> Vec<f64> {
        let out = gru
            .forward(client, &Var::new(x.clone(), false), None)
            .unwrap();
        f64s(out.0.tensor())
    }

    #[test]
    fn eval_mode_ignores_dropout() {
        let (client, device) = cpu_setup();
        let w = weights(&two_layer_bi(true, 0.0));
        let x = t64(&vals(5 * 2 * 3, 4), &[5, 2, 3], &device);
        let plain_gru = build(&two_layer_bi(true, 0.0), &w, true, &device);
        assert!(plain_gru.is_training());
        let plain = run(&client, &plain_gru, &x);
        assert_eq!(
            plain,
            run(&client, &plain_gru, &x),
            "zero dropout is deterministic"
        );

        let mut dropped = build(&two_layer_bi(true, 0.5), &w, true, &device);
        assert_ne!(
            run(&client, &dropped, &x),
            plain,
            "training mode must apply dropout"
        );
        dropped.eval();
        assert!(!dropped.is_training());
        assert_eq!(run(&client, &dropped, &x), plain);
        dropped.train();
        assert!(dropped.is_training());
    }

    #[test]
    fn no_dropout_after_last_layer() {
        let (client, device) = cpu_setup();
        let cfg = RnnConfig {
            dropout: 0.5,
            ..RnnConfig::new(3, 4)
        };
        let x = t64(&vals(5 * 2 * 3, 4), &[5, 2, 3], &device);
        let mut gru = build(&cfg, &weights(&cfg), true, &device);
        let trained = run(&client, &gru, &x);
        gru.eval();
        assert_eq!(trained, run(&client, &gru, &x));
    }

    #[test]
    fn missing_h0_equals_zero_h0() {
        let (client, device) = cpu_setup();
        let cfg = two_layer_bi(true, 0.0);
        let gru = build(&cfg, &weights(&cfg), false, &device);
        let x = Var::new(t64(&vals(5 * 2 * 3, 4), &[5, 2, 3], &device), false);
        let zeros = Var::new(t64(&[0.0; 4 * 2 * 4], &[4, 2, 4], &device), false);
        let (a_out, a_h) = gru.forward(&client, &x, None).unwrap();
        let (b_out, b_h) = gru.forward(&client, &x, Some(&zeros)).unwrap();
        assert_eq!(f64s(a_out.tensor()), f64s(b_out.tensor()));
        assert_eq!(f64s(a_h.tensor()), f64s(b_h.tensor()));
    }

    #[test]
    fn rejects_bad_shapes() {
        let (client, device) = cpu_setup();
        let cfg = two_layer_bi(true, 0.0);
        let gru = build(&cfg, &weights(&cfg), false, &device);
        let var = |shape: &[usize]| {
            Var::new(t64(&vals(shape.iter().product(), 1), shape, &device), false)
        };
        let fwd = |x: &Var<CpuRuntime>, h0: Option<&Var<CpuRuntime>>| {
            gru.forward(&client, x, h0).err().map(|e| e.to_string())
        };
        assert!(fwd(&var(&[5, 3]), None).unwrap().contains("'x'"));
        assert!(fwd(&var(&[5, 2, 4]), None).unwrap().contains("'x'"));
        assert!(fwd(&var(&[0, 2, 3]), None).unwrap().contains("'x'"));
        let x = var(&[5, 2, 3]);
        assert!(fwd(&x, Some(&var(&[4, 2, 4]))).is_none());
        assert!(fwd(&x, Some(&var(&[2, 2, 4]))).unwrap().contains("'h0'"));
        assert!(fwd(&x, Some(&var(&[4, 2, 3]))).unwrap().contains("'h0'"));
    }

    #[test]
    fn from_parts_rejects_lstm_shaped_cells() {
        let (_, device) = cpu_setup();
        let cfg = RnnConfig::new(3, 2);
        let cell = |rows: usize| {
            let leaf = |shape: &[usize]| {
                Var::new(t64(&vals(shape.iter().product(), 2), shape, &device), true)
            };
            GruCellParams::new(
                leaf(&[rows, 3]),
                leaf(&[rows, 2]),
                Some((leaf(&[rows]), leaf(&[rows]))),
            )
        };
        assert!(TrainableGru::from_parts(cfg.clone(), vec![cell(6)]).is_ok());
        let err = TrainableGru::from_parts(cfg, vec![cell(8)]).err().unwrap();
        assert!(err.to_string().contains("weight_ih_l0"), "{err}");
    }

    #[test]
    fn init_draws_within_pytorch_bound() {
        let (client, device) = cpu_setup();
        let cfg = RnnConfig {
            num_layers: 2,
            bidirectional: true,
            ..RnnConfig::new(5, 16)
        };
        let mut map = VarMap::<CpuRuntime>::new();
        let mut vb = VarBuilder::new(&mut map, &device).with_seed(7);
        let gru = TrainableGru::init(&mut vb, cfg, DType::F64, &client, true).unwrap();
        assert_eq!(gru.named_parameters()[0].1.shape(), &[48, 5]);
        assert_eq!(gru.named_parameters()[8].1.shape(), &[48, 32]);
        let bound = 1.0 / 4.0;
        for (name, var) in gru.named_parameters() {
            let data = f64s(var.tensor());
            assert!(
                data.iter().all(|v| v.abs() <= bound + 1e-6),
                "{name} exceeds {bound}"
            );
            assert!(data.iter().any(|v| *v != 0.0), "{name} is all zero");
        }
        assert_eq!(gru.trainable_parameters().len(), 16);
    }
}
