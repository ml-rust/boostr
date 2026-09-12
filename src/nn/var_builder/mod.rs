//! VarBuilder: scoped access to weights in a VarMap.
//!
//! Provides prefix-based navigation for hierarchical weight names
//! (e.g., "model.layers.0.self_attn.q_proj.weight").
//!
//! - `builder`: the [`VarBuilder`] type, prefix navigation, borrowing getters
//! - `take`: owning `take_*` accessors that remove entries from the map
//! - `init`: `take_or_init_tensor` and the seeded-init derivation

mod builder;
mod init;
mod take;

pub use builder::VarBuilder;
