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
    /// default tile. At the narrow and single-warp tiles it is `_y<Y>` on the
    /// two-half cadence, and at the narrow tile `_y<Y>g` on the full-group
    /// cadence.
    pub fn kernel_name(self, format: &FeatMajorFormat, role: Role) -> String {
        let role = match role {
            Role::TileParallel => "",
            Role::Fused => "_ms",
            Role::SplitK => "_sk",
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

#[cfg(test)]
mod tests {
    use super::super::super::formats::{Q4_K, Q6_K};
    use super::super::geometry::{FEAT_TILE_DEFAULT, FEAT_TILE_NARROW, FEAT_TILE_SMALL};
    use super::*;

    fn wide(mmq_x: u32) -> Tiling {
        Tiling {
            feat_tile: FEAT_TILE_DEFAULT,
            mmq_x,
            cadence: Cadence::Halves,
        }
    }

    fn narrow(mmq_x: u32) -> Tiling {
        Tiling {
            feat_tile: FEAT_TILE_NARROW,
            mmq_x,
            cadence: Cadence::Halves,
        }
    }

    fn group(mmq_x: u32) -> Tiling {
        Tiling {
            feat_tile: FEAT_TILE_NARROW,
            mmq_x,
            cadence: Cadence::Group,
        }
    }

    fn small(mmq_x: u32) -> Tiling {
        Tiling {
            feat_tile: FEAT_TILE_SMALL,
            mmq_x,
            cadence: Cadence::Halves,
        }
    }

    #[test]
    fn threads_follow_the_feature_tile() {
        let wide = wide(24);
        let narrow = narrow(24);
        assert_eq!(wide.threads(), 256);
        assert_eq!(narrow.threads(), 128);
        assert_eq!(small(8).threads(), 32);
    }

    #[test]
    fn kernel_names_follow_the_macro_spelling() {
        let wide = wide(24);
        let narrow = narrow(24);
        assert_eq!(
            wide.kernel_name(&Q4_K, Role::TileParallel),
            "quant_mmq_q4_k_q8_1_mma_x24"
        );
        assert_eq!(
            wide.kernel_name(&Q4_K, Role::SplitK),
            "quant_mmq_q4_k_q8_1_mma_sk_x24"
        );
        assert_eq!(
            narrow.kernel_name(&Q6_K, Role::TileParallel),
            "quant_mmq_q6_k_q8_1_mma_y64_x24"
        );
        assert_eq!(
            narrow.kernel_name(&Q6_K, Role::SplitK),
            "quant_mmq_q6_k_q8_1_mma_sk_y64_x24"
        );
        assert_eq!(
            narrow.kernel_name(&Q6_K, Role::Fixup),
            "quant_mmq_q6_k_q8_1_mma_fixup_y64_x24"
        );
        assert_eq!(
            small(8).kernel_name(&Q4_K, Role::SplitK),
            "quant_mmq_q4_k_q8_1_mma_sk_y16_x8"
        );
        let group = group(48);
        assert_eq!(
            group.kernel_name(&Q6_K, Role::TileParallel),
            "quant_mmq_q6_k_q8_1_mma_y64g_x48"
        );
        assert_eq!(
            group.kernel_name(&Q6_K, Role::SplitK),
            "quant_mmq_q6_k_q8_1_mma_sk_y64g_x48"
        );
        assert_eq!(
            group.kernel_name(&Q6_K, Role::Fixup),
            "quant_mmq_q6_k_q8_1_mma_fixup_y64g_x48"
        );
    }
}
