//! The cell abstraction shared by the LSTM and GRU stacks.
//!
//! A [`CellKind`] fixes the gate count of each weight and the step math.
//! [`RnnStack`](super::stack::RnnStack) runs layers, directions, layout and
//! dropout the same way for every kind.

use crate::error::{Error, Result};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

/// Recurrent weights of one direction, prepared once per sequence.
pub struct Recurrence<'a, R: Runtime> {
    /// `W_hh^T`, `[H, gates * H]`.
    pub(super) w_hh_t: &'a Var<R>,
    /// `b_hh`, `[gates * H]`, if the layer has bias.
    pub(super) b_hh: Option<&'a Var<R>>,
    pub(super) hidden: usize,
}

/// One recurrent cell type: its gate count, state names and step.
///
/// The step signature names a crate-internal type, so only
/// [`LstmKind`] and [`GruKind`] implement it.
pub trait CellKind {
    /// Gate blocks stacked in each weight and bias.
    const GATES: usize;
    /// Names of the initial-state inputs, `h0` first.
    const STATES: &'static [&'static str];

    /// Advance `state` by one step.
    ///
    /// * `xw_t` — `[B, GATES * H]`, the input projection `x_t @ W_ih^T + b_ih`.
    /// * `state` — one `[B, H]` tensor per entry of [`Self::STATES`].
    ///
    /// Returns the next state in the same order, `h'` first.
    fn step<R, C>(
        client: &C,
        rec: &Recurrence<'_, R>,
        xw_t: &Var<R>,
        state: &[Var<R>],
    ) -> Result<Vec<Var<R>>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>;
}

/// LSTM cell: gates `[i, f, g, o]`, state `(h, c)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LstmKind;

/// GRU cell: gates `[r, z, n]`, state `h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GruKind;

/// Error for a state list whose length differs from the cell's state count.
pub(super) fn state_count_error(want: usize, got: usize) -> Error {
    Error::InvalidArgument {
        arg: "state",
        reason: format!("expected {want} state tensors, got {got}"),
    }
}
