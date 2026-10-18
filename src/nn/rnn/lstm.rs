//! [`TrainableLstm`]: a multi-layer, optionally bidirectional LSTM in autograd.
//!
//! The math and weight layout equal `torch.nn.LSTM`. Each direction projects
//! the whole input in one matmul, `xw = x @ W_ih^T + b_ih`, then runs
//! [`lstm_cell_step`] once per time step. The reverse direction walks time
//! backward and stores each output at its original index. Directions join on
//! the feature axis. Dropout runs between layers in training mode only, never
//! after the last layer.
//!
//! The inference [`Lstm`](crate::nn::Lstm) holds raw tensors with private
//! fields, so no bridge from it exists. Load both from the same checkpoint
//! names instead.

use super::config::RnnConfig;
use super::lstm_cell::{Recurrence, lstm_cell_step};
use super::params::LstmCellParams;
use crate::error::{Error, Result};
use crate::nn::module::TrainMode;
use crate::nn::var_ops::var_contiguous;
use numr::autograd::{
    Var, var_add, var_cat, var_dropout, var_matmul, var_narrow, var_permute, var_reshape,
    var_transpose,
};
use numr::dtype::DType;
use numr::ops::{RandomOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Final `(h_n, c_n)`, each `[num_layers * num_directions, B, H]`.
pub type LstmState<R> = (Var<R>, Var<R>);

/// Trainable LSTM stack with `torch.nn.LSTM` semantics and parameter names.
///
/// Build it with [`TrainableLstm::load`], [`TrainableLstm::init`] or
/// [`TrainableLstm::from_parts`]. It starts in training mode, as PyTorch does.
pub struct TrainableLstm<R: Runtime> {
    pub(super) config: RnnConfig,
    /// One entry per layer and direction, at index `layer * dirs + dir`.
    pub(super) cells: Vec<LstmCellParams<R>>,
    training: bool,
}

impl<R: Runtime> TrainableLstm<R> {
    /// Build from one [`LstmCellParams`] per layer and direction.
    ///
    /// `cells[layer * dirs + dir]` holds layer `layer`, with `dir == 1` the
    /// reverse direction. A wrong count, a wrong shape or a bias that
    /// disagrees with `config.bias` is an error.
    pub fn from_parts(config: RnnConfig, cells: Vec<LstmCellParams<R>>) -> Result<Self> {
        config.validate()?;
        let dirs = config.num_directions();
        let want = config.num_layers * dirs;
        if cells.len() != want {
            return Err(Error::InvalidArgument {
                arg: "cells",
                reason: format!(
                    "expected {want} cells ({} layers x {dirs} directions), got {}",
                    config.num_layers,
                    cells.len()
                ),
            });
        }
        for (slot, cell) in cells.iter().enumerate() {
            cell.check(&config, slot / dirs, slot % dirs == 1)?;
        }
        Ok(Self {
            config,
            cells,
            training: true,
        })
    }

    pub fn config(&self) -> &RnnConfig {
        &self.config
    }

    /// Weights of `layer` in the given direction, if that cell exists.
    pub fn cell(&self, layer: usize, reverse: bool) -> Option<&LstmCellParams<R>> {
        if reverse && !self.config.bidirectional {
            return None;
        }
        let slot = layer * self.config.num_directions() + usize::from(reverse);
        self.cells.get(slot)
    }
}

impl<R: Runtime<DType = DType>> TrainableLstm<R> {
    /// Run the stack over a whole sequence.
    ///
    /// * `x` — `[B, T, input_size]` with `batch_first`, else `[T, B, input_size]`.
    /// * `state` — optional `(h0, c0)`, each `[num_layers * dirs, B, H]`.
    ///   `None` starts every layer and direction from zeros.
    ///
    /// Returns `(output, (h_n, c_n))`. `output` is `[B, T, dirs * H]` with
    /// `batch_first`, else `[T, B, dirs * H]`. `h_n` and `c_n` are always
    /// `[num_layers * dirs, B, H]`. The reverse entry holds its state after t = 0.
    pub fn forward<C>(
        &self,
        client: &C,
        x: &Var<R>,
        state: Option<(&Var<R>, &Var<R>)>,
    ) -> Result<(Var<R>, LstmState<R>)>
    where
        C: RuntimeClient<R> + TensorOps<R> + RandomOps<R> + ScalarOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let cfg = &self.config;
        let layout = if cfg.batch_first {
            "[B, T, I]"
        } else {
            "[T, B, I]"
        };
        if x.shape().len() != 3 || x.shape()[2] != cfg.input_size {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!(
                    "expected {layout} with I = {}, got {:?}",
                    cfg.input_size,
                    x.shape()
                ),
            });
        }
        let x_tm = if cfg.batch_first {
            var_permute(x, &[1, 0, 2]).map_err(Error::Numr)?
        } else {
            x.alias()
        };
        let x_tm = var_contiguous(&x_tm)?;
        let (steps, batch) = (x_tm.shape()[0], x_tm.shape()[1]);
        if steps == 0 || batch == 0 {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!(
                    "sequence length and batch must be at least 1, got {:?} for {layout}",
                    x.shape()
                ),
            });
        }

        let dirs = cfg.num_directions();
        let slots = cfg.num_layers * dirs;
        let state_shape = [slots, batch, cfg.hidden_size];
        let initial = match state {
            Some((h0, c0)) => {
                for (arg, v) in [("h0", h0), ("c0", c0)] {
                    if v.shape() != state_shape {
                        return Err(Error::InvalidArgument {
                            arg,
                            reason: format!(
                                "expected {state_shape:?} [layers * dirs, B, H], got {:?}",
                                v.shape()
                            ),
                        });
                    }
                }
                Some((var_contiguous(h0)?, var_contiguous(c0)?))
            }
            None => None,
        };
        let start = |slot: usize| -> Result<(Var<R>, Var<R>)> {
            let shape = [batch, cfg.hidden_size];
            match &initial {
                Some((h0, c0)) => {
                    Ok((state_slot(h0, slot, &shape)?, state_slot(c0, slot, &shape)?))
                }
                None => {
                    let zeros = || {
                        Tensor::<R>::zeros(&shape, x.tensor().dtype(), x.tensor().device())
                            .map(|t| Var::new(t, false))
                            .map_err(Error::Numr)
                    };
                    Ok((zeros()?, zeros()?))
                }
            }
        };

        let mut layer_in = x_tm;
        let mut h_n = Vec::with_capacity(slots);
        let mut c_n = Vec::with_capacity(slots);
        for (layer, cells) in self.cells.chunks(dirs).enumerate() {
            let mut outs = Vec::with_capacity(dirs);
            for (dir, cell) in cells.iter().enumerate() {
                let (h, c) = start(layer * dirs + dir)?;
                let (out, h_last, c_last) =
                    self.run_direction(client, cell, &layer_in, h, c, dir == 1)?;
                outs.push(out);
                h_n.push(var_reshape(&h_last, &[1, batch, cfg.hidden_size]).map_err(Error::Numr)?);
                c_n.push(var_reshape(&c_last, &[1, batch, cfg.hidden_size]).map_err(Error::Numr)?);
            }
            let out = if outs.len() == 1 {
                outs.swap_remove(0)
            } else {
                let refs: Vec<&Var<R>> = outs.iter().collect();
                var_cat(&refs, 2, client).map_err(Error::Numr)?
            };
            let drop = self.training && cfg.dropout > 0.0 && layer + 1 < cfg.num_layers;
            layer_in = if drop {
                var_dropout(&out, cfg.dropout, client)
                    .map_err(Error::Numr)?
                    .0
            } else {
                out
            };
        }

        let output = if cfg.batch_first {
            var_contiguous(&var_permute(&layer_in, &[1, 0, 2]).map_err(Error::Numr)?)?
        } else {
            layer_in
        };
        let h_refs: Vec<&Var<R>> = h_n.iter().collect();
        let c_refs: Vec<&Var<R>> = c_n.iter().collect();
        let h_n = var_cat(&h_refs, 0, client).map_err(Error::Numr)?;
        let c_n = var_cat(&c_refs, 0, client).map_err(Error::Numr)?;
        Ok((output, (h_n, c_n)))
    }

    /// One direction of one layer over time-major `input` `[T, B, F]`.
    ///
    /// Returns `(output [T, B, H], h_last [B, H], c_last [B, H])`. `output`
    /// is in original time order for both directions.
    fn run_direction<C>(
        &self,
        client: &C,
        cell: &LstmCellParams<R>,
        input: &Var<R>,
        mut h: Var<R>,
        mut c: Var<R>,
        reverse: bool,
    ) -> Result<(Var<R>, Var<R>, Var<R>)>
    where
        C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let (steps, batch, features) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let hidden = self.config.hidden_size;
        let gate = self.config.gate_size();

        let flat = var_reshape(input, &[steps * batch, features]).map_err(Error::Numr)?;
        let w_ih_t = var_transpose(&cell.weight_ih).map_err(Error::Numr)?;
        let xw = var_matmul(&flat, &w_ih_t, client).map_err(Error::Numr)?;
        let xw = match &cell.bias_ih {
            Some(b) => var_add(&xw, b, client).map_err(Error::Numr)?,
            None => xw,
        };
        let xw = var_reshape(&xw, &[steps, batch, gate]).map_err(Error::Numr)?;

        let w_hh_t = var_transpose(&cell.weight_hh).map_err(Error::Numr)?;
        let rec = Recurrence {
            w_hh_t: &w_hh_t,
            b_hh: cell.bias_hh.as_ref(),
            hidden,
        };
        let mut outs = Vec::with_capacity(steps);
        for step in 0..steps {
            let t = if reverse { steps - 1 - step } else { step };
            let xw_t = var_narrow(&xw, 0, t, 1).map_err(Error::Numr)?;
            let xw_t = var_reshape(&var_contiguous(&xw_t)?, &[batch, gate]).map_err(Error::Numr)?;
            let (h_next, c_next) = lstm_cell_step(client, &rec, &xw_t, &h, &c)?;
            outs.push(var_reshape(&h_next, &[1, batch, hidden]).map_err(Error::Numr)?);
            h = h_next;
            c = c_next;
        }
        if reverse {
            outs.reverse();
        }
        let refs: Vec<&Var<R>> = outs.iter().collect();
        let output = var_cat(&refs, 0, client).map_err(Error::Numr)?;
        Ok((output, h, c))
    }
}

/// Entry `slot` of a `[slots, B, H]` state, as `[B, H]`.
fn state_slot<R: Runtime<DType = DType>>(
    state: &Var<R>,
    slot: usize,
    shape: &[usize],
) -> Result<Var<R>>
where
    R::Client: TensorOps<R>,
{
    let row = var_narrow(state, 0, slot, 1).map_err(Error::Numr)?;
    var_reshape(&var_contiguous(&row)?, shape).map_err(Error::Numr)
}

impl<R: Runtime> TrainMode for TrainableLstm<R> {
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::lstm::{BiLstm, Lstm};
    use crate::nn::rnn::params::param_suffix;
    use crate::nn::var_builder::VarBuilder;
    use crate::nn::varmap::VarMap;
    use crate::test_utils::cpu_setup;
    use numr::autograd::{backward, var_mul, var_sum};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

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
        let gate = cfg.gate_size();
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

    fn t32(v: &[f64], shape: &[usize], d: &CpuDevice) -> Tensor<CpuRuntime> {
        let v: Vec<f32> = v.iter().map(|x| *x as f32).collect();
        Tensor::<CpuRuntime>::from_slice(&v, shape, d).unwrap()
    }

    fn build(
        cfg: &RnnConfig,
        w: &Named,
        f64_dtype: bool,
        trainable: bool,
        d: &CpuDevice,
    ) -> TrainableLstm<CpuRuntime> {
        let mut map = VarMap::<CpuRuntime>::new();
        for (name, shape, v) in w {
            let t = if f64_dtype {
                t64(v, shape, d)
            } else {
                t32(v, shape, d)
            };
            map.insert(name.clone(), t);
        }
        let vb = VarBuilder::new(&mut map, d);
        TrainableLstm::load(&vb, cfg.clone(), trainable).unwrap()
    }

    fn f32s(v: &Var<CpuRuntime>) -> Vec<f32> {
        v.tensor().contiguous().unwrap().to_vec()
    }

    fn f64s(t: &Tensor<CpuRuntime>) -> Vec<f64> {
        t.contiguous().unwrap().to_vec()
    }

    /// `sum(out * Wo) + sum(h_n * Wh) + sum(c_n * Wc)` with fixed probes.
    fn loss(
        c: &CpuClient,
        lstm: &TrainableLstm<CpuRuntime>,
        x: &Var<CpuRuntime>,
        h0: &Var<CpuRuntime>,
        c0: &Var<CpuRuntime>,
    ) -> Var<CpuRuntime> {
        let (out, (hn, cn)) = lstm.forward(c, x, Some((h0, c0))).unwrap();
        let mut total: Option<Var<CpuRuntime>> = None;
        for (k, v) in [out, hn, cn].iter().enumerate() {
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

    fn scalar(v: &Var<CpuRuntime>) -> f64 {
        f64s(v.tensor())[0]
    }

    /// A deterministic spread of entries: first, thirds, middle, last.
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

    /// Finite-difference check of every parameter, `x`, `h0` and `c0`.
    fn check_gradients(cfg: RnnConfig, batch: usize, steps: usize) {
        let (client, device) = cpu_setup();
        let w = weights(&cfg);
        let x_shape = if cfg.batch_first {
            [batch, steps, cfg.input_size]
        } else {
            [steps, batch, cfg.input_size]
        };
        let s_shape = [
            cfg.num_layers * cfg.num_directions(),
            batch,
            cfg.hidden_size,
        ];
        let x = vals(x_shape.iter().product(), 50);
        let h0 = vals(s_shape.iter().product(), 60);
        let c0 = vals(s_shape.iter().product(), 70);

        let lstm = build(&cfg, &w, true, true, &device);
        let xv = Var::new(t64(&x, &x_shape, &device), true);
        let hv = Var::new(t64(&h0, &s_shape, &device), true);
        let cv = Var::new(t64(&c0, &s_shape, &device), true);
        let grads = backward(&loss(&client, &lstm, &xv, &hv, &cv), &client).unwrap();
        let grad = |id| f64s(grads.get(id).unwrap());

        let eval = |w: &Named, x: &[f64], h0: &[f64], c0: &[f64]| {
            let lstm = build(&cfg, w, true, false, &device);
            let leaf = |v: &[f64], s: &[usize]| Var::new(t64(v, s, &device), false);
            let l = loss(
                &client,
                &lstm,
                &leaf(x, &x_shape),
                &leaf(h0, &s_shape),
                &leaf(c0, &s_shape),
            );
            scalar(&l)
        };

        let named = lstm.named_parameters();
        assert_eq!(named.len(), w.len());
        for (p, (name, var)) in named.iter().enumerate() {
            assert_eq!(name, &w[p].0);
            compare(name, &grad(var.id()), &w[p].2, |v| {
                let mut wp = w.clone();
                wp[p].2 = v.to_vec();
                eval(&wp, &x, &h0, &c0)
            });
        }
        compare("x", &grad(xv.id()), &x, |v| eval(&w, v, &h0, &c0));
        compare("h0", &grad(hv.id()), &h0, |v| eval(&w, &x, v, &c0));
        compare("c0", &grad(cv.id()), &c0, |v| eval(&w, &x, &h0, v));
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

    /// Inference `Lstm` for the named cell, from the same F32 values.
    fn oracle(w: &Named, layer: usize, reverse: bool, d: &CpuDevice) -> Lstm<CpuRuntime> {
        let s = param_suffix(layer, reverse);
        let get = |kind: &str| {
            let (_, shape, v) = w
                .iter()
                .find(|(n, _, _)| *n == format!("{kind}_{s}"))
                .unwrap();
            t32(v, shape, d)
        };
        Lstm::new(
            get("weight_ih"),
            get("weight_hh"),
            get("bias_ih"),
            get("bias_hh"),
        )
        .unwrap()
    }

    fn assert_close(label: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (k, (a, b)) in got.iter().zip(want).enumerate() {
            assert!((a - b).abs() < 1e-5, "{label}[{k}]: {a} vs {b}");
        }
    }

    /// Output and final state equal the inference modules, layer by layer.
    fn check_against_inference(cfg: RnnConfig, batch: usize, steps: usize) {
        let (client, device) = cpu_setup();
        let w = weights(&cfg);
        let lstm = build(&cfg, &w, false, false, &device);
        let x_bf = t32(
            &vals(batch * steps * cfg.input_size, 9),
            &[batch, steps, cfg.input_size],
            &device,
        );
        let (out, (hn, cn)) = lstm
            .forward(&client, &Var::new(x_bf.clone(), false), None)
            .unwrap();

        let mut layer_in = x_bf;
        let (mut want_h, mut want_c) = (Vec::new(), Vec::new());
        for layer in 0..cfg.num_layers {
            let fwd = oracle(&w, layer, false, &device);
            let (f_out, f_h, f_c) = fwd.forward(&client, &layer_in, false).unwrap();
            want_h.extend(f_h.to_vec::<f32>());
            want_c.extend(f_c.to_vec::<f32>());
            layer_in = if cfg.bidirectional {
                let bwd = oracle(&w, layer, true, &device);
                let (_, b_h, b_c) = bwd.forward(&client, &layer_in, true).unwrap();
                want_h.extend(b_h.to_vec::<f32>());
                want_c.extend(b_c.to_vec::<f32>());
                BiLstm::new(fwd, bwd)
                    .unwrap()
                    .forward(&client, &layer_in)
                    .unwrap()
            } else {
                f_out
            };
        }
        assert_eq!(out.shape(), &[batch, steps, cfg.output_size()]);
        assert_close("output", &f32s(&out), &tensor_f32(&layer_in));
        assert_close("h_n", &f32s(&hn), &want_h);
        assert_close("c_n", &f32s(&cn), &want_c);
    }

    fn tensor_f32(t: &Tensor<CpuRuntime>) -> Vec<f32> {
        t.contiguous().unwrap().to_vec()
    }

    #[test]
    fn matches_inference_lstm_unidirectional() {
        let cfg = RnnConfig {
            batch_first: true,
            ..RnnConfig::new(3, 4)
        };
        check_against_inference(cfg, 2, 5);
    }

    #[test]
    fn matches_inference_bilstm_two_layers() {
        let cfg = RnnConfig {
            num_layers: 2,
            bidirectional: true,
            batch_first: true,
            ..RnnConfig::new(3, 4)
        };
        check_against_inference(cfg, 2, 5);
    }

    fn two_layer(dropout: f64) -> RnnConfig {
        RnnConfig {
            num_layers: 2,
            bidirectional: true,
            dropout,
            ..RnnConfig::new(3, 4)
        }
    }

    fn run(
        client: &CpuClient,
        lstm: &TrainableLstm<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
    ) -> Vec<f32> {
        f32s(
            &lstm
                .forward(client, &Var::new(x.clone(), false), None)
                .unwrap()
                .0,
        )
    }

    #[test]
    fn zero_dropout_is_deterministic_in_training() {
        let (client, device) = cpu_setup();
        let cfg = two_layer(0.0);
        let lstm = build(&cfg, &weights(&cfg), false, true, &device);
        assert!(lstm.is_training());
        let x = t32(&vals(5 * 2 * 3, 4), &[5, 2, 3], &device);
        assert_eq!(run(&client, &lstm, &x), run(&client, &lstm, &x));
    }

    #[test]
    fn eval_mode_ignores_dropout() {
        let (client, device) = cpu_setup();
        let w = weights(&two_layer(0.0));
        let x = t32(&vals(5 * 2 * 3, 4), &[5, 2, 3], &device);
        let plain = run(
            &client,
            &build(&two_layer(0.0), &w, false, true, &device),
            &x,
        );

        let mut dropped = build(&two_layer(0.5), &w, false, true, &device);
        let trained = run(&client, &dropped, &x);
        assert_ne!(trained, plain, "training mode must apply dropout");
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
        let w = weights(&cfg);
        let x = t32(&vals(5 * 2 * 3, 4), &[5, 2, 3], &device);
        let mut lstm = build(&cfg, &w, false, true, &device);
        let trained = run(&client, &lstm, &x);
        lstm.eval();
        assert_eq!(trained, run(&client, &lstm, &x));
    }

    #[test]
    fn batch_first_round_trip() {
        let (client, device) = cpu_setup();
        let tm_cfg = two_layer(0.0);
        let bf_cfg = RnnConfig {
            batch_first: true,
            ..tm_cfg.clone()
        };
        let w = weights(&tm_cfg);
        let x_bf = t32(&vals(2 * 5 * 3, 8), &[2, 5, 3], &device);
        let x_tm = x_bf.permute(&[1, 0, 2]).unwrap().contiguous().unwrap();

        let bf = build(&bf_cfg, &w, false, false, &device);
        let tm = build(&tm_cfg, &w, false, false, &device);
        let (bf_out, (bf_h, bf_c)) = bf.forward(&client, &Var::new(x_bf, false), None).unwrap();
        let (tm_out, (tm_h, tm_c)) = tm.forward(&client, &Var::new(x_tm, false), None).unwrap();
        assert_eq!(bf_out.shape(), &[2, 5, 8]);
        assert_eq!(tm_out.shape(), &[5, 2, 8]);
        let tm_as_bf = tm_out.tensor().permute(&[1, 0, 2]).unwrap();
        assert_eq!(f32s(&bf_out), tensor_f32(&tm_as_bf));
        assert_eq!(bf_h.shape(), &[4, 2, 4]);
        assert_eq!(f32s(&bf_h), f32s(&tm_h));
        assert_eq!(f32s(&bf_c), f32s(&tm_c));
    }

    #[test]
    fn rejects_bad_shapes() {
        let (client, device) = cpu_setup();
        let cfg = two_layer(0.0);
        let lstm = build(&cfg, &weights(&cfg), false, false, &device);
        let var = |shape: &[usize]| {
            Var::new(t32(&vals(shape.iter().product(), 1), shape, &device), false)
        };
        let fwd = |x: &Var<CpuRuntime>, s: Option<(&Var<CpuRuntime>, &Var<CpuRuntime>)>| {
            lstm.forward(&client, x, s).err().map(|e| e.to_string())
        };
        assert!(fwd(&var(&[5, 3]), None).unwrap().contains("'x'"));
        assert!(fwd(&var(&[5, 2, 4]), None).unwrap().contains("'x'"));
        assert!(fwd(&var(&[0, 2, 3]), None).is_some());
        let x = var(&[5, 2, 3]);
        let good = var(&[4, 2, 4]);
        let bad = var(&[2, 2, 4]);
        assert!(fwd(&x, Some((&good, &good))).is_none());
        assert!(fwd(&x, Some((&bad, &good))).unwrap().contains("'h0'"));
        assert!(fwd(&x, Some((&good, &bad))).unwrap().contains("'c0'"));
    }

    #[test]
    fn cell_lookup_follows_directions() {
        let (_, device) = cpu_setup();
        let cfg = two_layer(0.0);
        let lstm = build(&cfg, &weights(&cfg), false, false, &device);
        assert_eq!(lstm.cell(1, true).unwrap().weight_ih().shape(), &[16, 8]);
        assert!(lstm.cell(2, false).is_none());
        let uni = RnnConfig::new(3, 4);
        let lstm = build(&uni, &weights(&uni), false, false, &device);
        assert!(lstm.cell(0, true).is_none());
        assert!(lstm.cell(0, false).unwrap().bias_hh().is_some());
    }
}
