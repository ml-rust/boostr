//! Trainable recurrent layers composed from tracked `Var` ops.

mod cell;
pub mod config;
mod driver;
pub mod gru;
mod gru_cell;
mod init;
pub mod lstm;
mod lstm_cell;
pub mod params;
pub mod stack;

pub use cell::{CellKind, GruKind, LstmKind};
pub use config::RnnConfig;
pub use gru::TrainableGru;
pub use lstm::{LstmState, TrainableLstm};
pub use params::{GruCellParams, LstmCellParams, RnnCellParams};
pub use stack::RnnStack;
