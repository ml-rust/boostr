//! Hybrid model mixing attention (dense or MLA) and Mamba2 SSM blocks.
//!
//! - `build`: the [`HybridModel`] type and [`HybridModel::from_varbuilder`]
//! - `layers`: the block one layer builds from its role and the config
//! - `forward`: the KV-cached forwards and the hidden-state forward
//! - `accessors`: the config, the layer split and the cache shapes

mod accessors;
mod build;
mod forward;
mod layers;

pub use build::HybridModel;

#[cfg(test)]
pub(crate) use build::tests::tiny_model;
#[cfg(test)]
pub(crate) use forward::tests::{
    checkpoint as mla_checkpoint, config as mla_config, map as mla_map,
};
