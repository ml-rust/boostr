//! Tile geometry for the feature-major MMQ launch: the (feature tile, token
//! tile) pair that serves a shape, the shared memory it needs, and whether
//! the stream-k pair or the tile-parallel grid runs it.
//!
//! Split into `geometry` (constants, [`FeatTile`], [`Cadence`], [`Role`],
//! shared-memory math), `tile` ([`Tiling`] and its derived values),
//! `select` (the tiling rule, and its tests) and `stream_k` (the stream-k
//! gate) to stay under this repo's 500-line file limit.

mod geometry;
mod select;
mod stream_k;
mod tile;

pub use geometry::FeatTile;
pub(super) use geometry::{Cadence, FEAT_TILE_DEFAULT, Role, VARIANTS, smem_opt_in_limit};
// Only test code outside this module (`formats/*`'s own unit tests) reaches
// `smem_bytes` through this re-export; a plain `cargo check` of the lib
// alone never touches it.
#[allow(unused_imports)]
pub(super) use geometry::smem_bytes;
pub(super) use select::{select_tiling, select_variant};
pub(super) use stream_k::use_stream_k;
pub(super) use tile::Tiling;
