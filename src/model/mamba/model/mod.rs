//! Full Mamba models — embedding + N layers + norm + LM head, one file per
//! generation: `mamba1`, `mamba2`, `mamba3`.

mod mamba1;
mod mamba2;
mod mamba3;

pub use mamba1::Mamba1Model;
pub use mamba2::Mamba2Model;
pub use mamba3::Mamba3Model;

#[cfg(test)]
pub(crate) use mamba2::tests::tiny_model as tiny_mamba2;
