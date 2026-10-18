//! The sequence driver shared by every [`CellKind`].
//!
//! Each direction projects the whole input in one matmul,
//! `xw = x @ W_ih^T + b_ih`, then runs [`CellKind::step`] once per time step.
//! The reverse direction walks time backward and stores each output at its
//! original index. Directions join on the feature axis. Dropout runs between
//! layers in training mode only, never after the last layer.

use super::cell::{CellKind, Recurrence, state_count_error};
use super::params::RnnCellParams;
use super::stack::RnnStack;
use crate::error::{Error, Result};
use crate::nn::var_ops::var_contiguous;
use numr::autograd::{
    Var, var_add, var_cat, var_dropout, var_matmul, var_narrow, var_permute, var_reshape,
    var_transpose,
};
use numr::dtype::DType;
use numr::ops::{RandomOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>, K: CellKind> RnnStack<R, K> {
    /// Run the stack over a whole sequence.
    ///
    /// * `x` — `[B, T, input_size]` with `batch_first`, else `[T, B, input_size]`.
    /// * `state` — optional initial state, one `[num_layers * dirs, B, H]`
    ///   tensor per entry of `K::STATES`. `None` starts from zeros.
    ///
    /// Returns `(output, finals)`. `finals` holds one `[num_layers * dirs, B, H]`
    /// tensor per entry of `K::STATES`. The reverse entry holds its state after t = 0.
    pub(super) fn run<C>(
        &self,
        client: &C,
        x: &Var<R>,
        state: Option<&[&Var<R>]>,
    ) -> Result<(Var<R>, Vec<Var<R>>)>
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
            Some(given) => {
                if given.len() != K::STATES.len() {
                    return Err(state_count_error(K::STATES.len(), given.len()));
                }
                let mut checked = Vec::with_capacity(given.len());
                for (&arg, v) in K::STATES.iter().zip(given) {
                    if v.shape() != state_shape {
                        return Err(Error::InvalidArgument {
                            arg,
                            reason: format!(
                                "expected {state_shape:?} [layers * dirs, B, H], got {:?}",
                                v.shape()
                            ),
                        });
                    }
                    checked.push(var_contiguous(v)?);
                }
                Some(checked)
            }
            None => None,
        };
        let start = |slot: usize| -> Result<Vec<Var<R>>> {
            let shape = [batch, cfg.hidden_size];
            match &initial {
                Some(given) => given.iter().map(|s| state_slot(s, slot, &shape)).collect(),
                None => (0..K::STATES.len())
                    .map(|_| {
                        Tensor::<R>::zeros(&shape, x.tensor().dtype(), x.tensor().device())
                            .map(|t| Var::new(t, false))
                            .map_err(Error::Numr)
                    })
                    .collect(),
            }
        };

        let mut layer_in = x_tm;
        let mut finals: Vec<Vec<Var<R>>> = (0..K::STATES.len())
            .map(|_| Vec::with_capacity(slots))
            .collect();
        for (layer, cells) in self.cells.chunks(dirs).enumerate() {
            let mut outs = Vec::with_capacity(dirs);
            for (dir, cell) in cells.iter().enumerate() {
                let init = start(layer * dirs + dir)?;
                let (out, last) = self.run_direction(client, cell, &layer_in, init, dir == 1)?;
                outs.push(out);
                for (parts, v) in finals.iter_mut().zip(&last) {
                    parts.push(var_reshape(v, &[1, batch, cfg.hidden_size]).map_err(Error::Numr)?);
                }
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
        let finals = finals
            .iter()
            .map(|parts| {
                let refs: Vec<&Var<R>> = parts.iter().collect();
                var_cat(&refs, 0, client).map_err(Error::Numr)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((output, finals))
    }

    /// One direction of one layer over time-major `input` `[T, B, F]`.
    ///
    /// Returns `(output [T, B, H], last state)`, each state entry `[B, H]`.
    /// `output` is in original time order for both directions.
    fn run_direction<C>(
        &self,
        client: &C,
        cell: &RnnCellParams<R>,
        input: &Var<R>,
        mut state: Vec<Var<R>>,
        reverse: bool,
    ) -> Result<(Var<R>, Vec<Var<R>>)>
    where
        C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let (steps, batch, features) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let hidden = self.config.hidden_size;
        let gate = Self::gate_rows(&self.config);

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
            state = K::step(client, &rec, &xw_t, &state)?;
            let h = state
                .first()
                .ok_or_else(|| state_count_error(K::STATES.len(), 0))?;
            outs.push(var_reshape(h, &[1, batch, hidden]).map_err(Error::Numr)?);
        }
        if reverse {
            outs.reverse();
        }
        let refs: Vec<&Var<R>> = outs.iter().collect();
        let output = var_cat(&refs, 0, client).map_err(Error::Numr)?;
        Ok((output, state))
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
