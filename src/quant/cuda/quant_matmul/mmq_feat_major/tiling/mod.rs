//! Tile geometry for the feature-major MMQ launch: the (feature tile, token
//! tile) pair that serves a shape, the shared memory it needs, how many
//! ranges the K walk is cut into, and whether the split-K pair or the
//! tile-parallel grid runs them.
//!
//! Split into `geometry` (constants, [`FeatTile`], [`Cadence`], [`Role`],
//! shared-memory math), `tile` ([`Tiling`] and its derived values),
//! `variant` (the token tile at one feature tile), `select` (the tiling
//! rule) and `split_k` (the split count and the launch choice) to stay under
//! this repo's 500-line file limit.

mod geometry;
mod select;
mod split_k;
mod tile;
mod variant;

pub use geometry::FeatTile;
pub(super) use geometry::{Cadence, FEAT_TILE_DEFAULT, Role, smem_opt_in_limit};
// Only test code outside this module (`formats/*`'s own unit tests) reaches
// `VARIANTS` through this re-export.
#[allow(unused_imports)]
pub(super) use geometry::{SMALL_X, VARIANTS};
// Only test code outside this module (`formats/*`'s own unit tests) reaches
// `smem_bytes` through this re-export; a plain `cargo check` of the lib
// alone never touches it.
#[allow(unused_imports)]
pub(super) use geometry::smem_bytes;
pub(super) use select::select_tiling;
pub(super) use split_k::{split_count, use_split_launch};
pub(super) use tile::Tiling;
pub(super) use variant::{record_token_slots, select_variant};
