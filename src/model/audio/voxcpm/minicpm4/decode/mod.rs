//! Incremental (KV-cached) decode for VoxCPM2's MiniCPM4 `base_lm`.
//!
//! The sibling [`MiniCpm4Model::forward`] recomputes every position on every
//! call. This module adds the generation path: prime a cache from a prefix
//! once, then advance one position at a time.
//!
//! ```text
//! let mut cache = model.new_kv_cache(batch, max_length)?;   // preallocated
//! let prefix_out = model.prefill(&client, &prefix_embeds, &mut cache)?;
//! let step_out   = model.decode_step(&client, &embed, &mut cache, position)?;
//! ```
//!
//! # Cache-shape equivalence with the reference
//!
//! The reference (`voxcpm/modules/minicpm4/cache.py`) preallocates the cache to
//! `max_length` and masks with `arange(max_length) <= position_id`;
//! [`KvCache`](crate::inference::KvCache) exposes only the slots actually
//! written, and the shared causal-mask builder admits `0..=position`. Both
//! select the SAME keys, and only while positions are written in order from 0.
//! [`MiniCpm4Attention::forward_cached`] carries the full argument at the
//! masking site; [`MiniCpm4Model::decode_step`] enforces the ordering premise.
//!
//! # Bounds
//!
//! The reference's `step()` raises when `current_length >= max_length`, and
//! nothing deeper in `forward_step` re-checks — an out-of-range position
//! silently corrupts the cache write. Here [`MiniCpm4Model::decode_step`]
//! rejects `position >= max_length` with an
//! [`Error::InvalidArgument`](crate::error::Error::InvalidArgument) before
//! any tensor is touched; there is no panicking index on this path.
//!
//! # Numerical agreement
//!
//! The step-wise and full-sequence paths run the same weights, the same
//! precomputed RoPE tables at the same absolute positions, and select the same
//! keys — but reduce over the key axis in a different order (one row at a time
//! versus a full matrix). They agree to roughly `1e-4` absolute, not bitwise.
//! The reference's own two paths differ by `9.9e-5` for the same reason.
//!
//! - `cache`: `new_kv_cache` sizing and the shared cache validation
//! - `step`: `prefill`, `decode_step`, and the cached layer stack
//!
//! [`MiniCpm4Model::forward`]: crate::model::audio::voxcpm::minicpm4::MiniCpm4Model::forward
//! [`MiniCpm4Model::decode_step`]: crate::model::audio::voxcpm::minicpm4::MiniCpm4Model::decode_step
//! [`MiniCpm4Attention::forward_cached`]: crate::model::audio::voxcpm::minicpm4::MiniCpm4Attention::forward_cached

mod cache;
mod step;
