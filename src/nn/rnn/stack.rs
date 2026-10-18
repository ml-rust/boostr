//! [`RnnStack`]: the layer stack behind [`TrainableLstm`] and [`TrainableGru`].
//!
//! The stack owns the config, one [`RnnCellParams`] per layer and direction,
//! and the training flag. The cell kind `K` fixes the gate count and the step.
//!
//! [`TrainableLstm`]: super::lstm::TrainableLstm
//! [`TrainableGru`]: super::gru::TrainableGru

use super::cell::CellKind;
use super::config::RnnConfig;
use super::params::RnnCellParams;
use crate::error::{Error, Result};
use crate::nn::module::TrainMode;
use numr::runtime::Runtime;
use std::marker::PhantomData;

/// Trainable recurrent stack with PyTorch semantics and parameter names.
///
/// Use it through [`TrainableLstm`](super::lstm::TrainableLstm) or
/// [`TrainableGru`](super::gru::TrainableGru). Build it with `load`, `init`
/// or [`RnnStack::from_parts`]. It starts in training mode, as PyTorch does.
pub struct RnnStack<R: Runtime, K> {
    pub(super) config: RnnConfig,
    /// One entry per layer and direction, at index `layer * dirs + dir`.
    pub(super) cells: Vec<RnnCellParams<R>>,
    pub(super) training: bool,
    kind: PhantomData<K>,
}

impl<R: Runtime, K: CellKind> RnnStack<R, K> {
    /// Build from one [`RnnCellParams`] per layer and direction.
    ///
    /// `cells[layer * dirs + dir]` holds layer `layer`, with `dir == 1` the
    /// reverse direction. A wrong count, a wrong shape or a bias that
    /// disagrees with `config.bias` is an error.
    pub fn from_parts(config: RnnConfig, cells: Vec<RnnCellParams<R>>) -> Result<Self> {
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
        let rows = Self::gate_rows(&config);
        for (slot, cell) in cells.iter().enumerate() {
            cell.check(&config, rows, slot / dirs, slot % dirs == 1)?;
        }
        Ok(Self {
            config,
            cells,
            training: true,
            kind: PhantomData,
        })
    }

    /// Rows of each weight and bias: `K::GATES * hidden_size`.
    pub(super) fn gate_rows(config: &RnnConfig) -> usize {
        K::GATES * config.hidden_size
    }
}

impl<R: Runtime, K> RnnStack<R, K> {
    pub fn config(&self) -> &RnnConfig {
        &self.config
    }

    /// Weights of `layer` in the given direction, if that cell exists.
    pub fn cell(&self, layer: usize, reverse: bool) -> Option<&RnnCellParams<R>> {
        if reverse && !self.config.bidirectional {
            return None;
        }
        let slot = layer * self.config.num_directions() + usize::from(reverse);
        self.cells.get(slot)
    }
}

impl<R: Runtime, K> TrainMode for RnnStack<R, K> {
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}
