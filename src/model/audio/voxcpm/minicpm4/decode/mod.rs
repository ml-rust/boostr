//! Incremental (KV-cached) decode for VoxCPM2's MiniCPM4 `base_lm`.
//!
//! The sibling [`MiniCpm4Model::forward`] recomputes every position on every
//! call. This module adds the generation path: prime a cache from a prefix
//! once, then advance one position at a time.
//!
//! ```text
//! let mut cache = model.new_kv_cache(batch, max_length)?;   // preallocated
//! let prefix_out = model.prefill(&client, &prefix_embeds, &mut cache)?;
//! let step_out   = model.decode_step(&client, &embed, &mut cache, &step)?;
//! ```
//!
//! `cache` documents cache-shape equivalence with the reference. `step`
//! documents the position bounds. `stack` documents the numerical agreement.
//!
//! - `cache`: `new_kv_cache` sizing and the shared cache validation
//! - `stack`: the cached layer stack both entry points share
//! - `step`: `prefill` and `decode_step`
//!
//! [`MiniCpm4Model::forward`]: crate::model::audio::voxcpm::minicpm4::MiniCpm4Model::forward

mod cache;
mod stack;
mod step;
