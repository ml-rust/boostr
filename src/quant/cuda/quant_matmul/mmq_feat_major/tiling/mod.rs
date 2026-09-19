//! Tile geometry for the feature-major MMQ launch: the (feature tile, token
//! tile) pair that serves a shape, the shared memory it needs, how many
//! ranges the K walk is cut into, and whether the split-K pair or the
//! tile-parallel grid runs them.
//!
//! `geometry` holds constants, [`FeatTile`], [`Schedule`], [`Cadence`],
//! [`Role`], and shared-memory math. `tile` holds [`Tiling`] and its
//! derived values. `variant` holds the token tile at one feature tile.
//! `select` holds the forced-tiling paths and the entry point. `auto` holds
//! the automatic tiling policy `select` applies for [`FeatTile::Auto`].
//! `split_k` holds the split count and the launch choice. `tile_parallel_tune`
//! holds the per-device measurement the launch choice reads.

mod auto;
mod geometry;
mod select;
mod split_k;
mod tile;
mod tile_parallel_tune;
mod variant;

pub(super) use geometry::{Cadence, FEAT_TILE_DEFAULT, Role, smem_opt_in_limit};
pub use geometry::{FeatTile, Schedule};
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
pub(in crate::quant::cuda::quant_matmul) use tile_parallel_tune::{
    prefers_tile_parallel, tile_parallel_probe_shape,
};
pub(super) use variant::{record_token_slots, select_variant};
