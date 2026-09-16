//! [`Tiling`]: one launch's tile geometry, and the values derived from it
//! (thread count, shared memory, tile counts, kernel symbol name).

use super::super::formats::FeatMajorFormat;
use super::Role;
use super::geometry::{Cadence, FEAT_TILE_DEFAULT, FEATURES_PER_WARP, WARP_SIZE, smem_bytes};

/// One launch's tile geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::quant::cuda::quant_matmul::mmq_feat_major) struct Tiling {
    pub feat_tile: u32,
    pub mmq_x: u32,
    pub cadence: Cadence,
}

impl Tiling {
    /// Threads per block: one warp per 16-feature minitile row of the tile.
    pub const fn threads(self) -> u32 {
        self.feat_tile / FEATURES_PER_WARP * WARP_SIZE
    }

    pub const fn smem_bytes(self, format: &FeatMajorFormat) -> u32 {
        smem_bytes(format, self.feat_tile, self.mmq_x, self.cadence)
    }

    pub const fn feat_tiles(self, n: u32) -> u32 {
        n.div_ceil(self.feat_tile)
    }

    pub const fn token_tiles(self, m: u32) -> u32 {
        m.div_ceil(self.mmq_x)
    }

    /// Output tiles the tile-parallel grid launches for an `m` x `n` product.
    pub const fn tiles(self, m: u32, n: u32) -> u32 {
        self.token_tiles(m) * self.feat_tiles(n)
    }

    /// Kernel symbol for `role`, as the `MMQ_FM_KERNEL_AT` macro spells it.
    /// The tiling infix is placed just before `_x<X>`. It is empty at the
    /// default tile. At the narrow tile it is `_y<Y>` on the two-half
    /// cadence and `_y<Y>g` on the full-group cadence.
    pub fn kernel_name(self, format: &FeatMajorFormat, role: Role) -> String {
        let role = match role {
            Role::TileParallel => "",
            Role::StreamK => "_sk",
            Role::Fixup => "_fixup",
        };
        let tag = match (self.feat_tile == FEAT_TILE_DEFAULT, self.cadence) {
            (true, _) => String::new(),
            (false, Cadence::Halves) => format!("_y{}", self.feat_tile),
            (false, Cadence::Group) => format!("_y{}g", self.feat_tile),
        };
        format!(
            "quant_mmq_{}_q8_1_mma{role}{tag}_x{}",
            format.kernel_infix, self.mmq_x
        )
    }
}
