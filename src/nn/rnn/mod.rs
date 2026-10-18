//! Trainable recurrent layers composed from tracked `Var` ops.

pub mod config;
mod init;
pub mod lstm;
mod lstm_cell;
pub mod params;

pub use config::RnnConfig;
pub use lstm::{LstmState, TrainableLstm};
pub use params::LstmCellParams;
