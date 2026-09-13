//! Tile geometry for the feature-major MMQ launch: the (feature tile, token
//! tile) pair that serves a shape, the shared memory it needs, and whether
//! the stream-k pair or the tile-parallel grid runs it.

use crate::error::{Error, Result};

use super::formats::FeatMajorFormat;

/// Feature tile every format compiles, at every token tile in [`VARIANTS`].
pub(super) const FEAT_TILE_DEFAULT: u32 = 128;

/// Narrow feature tile for the CTA-starved regime. Compiled only for the
/// formats whose `narrow_tile` flag is set, and only at [`NARROW_VARIANTS`].
pub(super) const FEAT_TILE_NARROW: u32 = 64;

/// Compiled token-tile variants at the default feature tile, ascending. Below
/// 48 the tile steps by 8, at and above it by 16; the kernel's warp blocking
/// rejects every other value.
pub(super) const VARIANTS: &[u32] = &[8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128];

/// Token tiles compiled at the narrow feature tile. Stops at 64: a batch that
/// wants a wider token tile already launches enough tiles at the default
/// feature tile. Must match the `MMQ_FM_KERNEL_Y64` list in
/// `src/quant/cuda/kernels/quant_mmq_mma.cu`.
pub(super) const NARROW_VARIANTS: &[u32] = &[8, 16, 24, 32, 40, 48, 64];

/// Widest narrow-tile token tile whose activation staging chain is short
/// enough to win without the doubled grid co-residing — see [`select_tiling`].
/// Measured across the K-quant formats at the small-N shapes.
const NARROW_SHORT_CHAIN_X: u32 = 32;

/// Activation row stride in the shared tile, in ints: 4 half2 scale pairs plus
/// 32 quant words. The same for every weight format.
const ACT_STRIDE: u32 = 36;

/// Output features one warp owns per MMA minitile row: the `m` of
/// `mma.m16n8k32`. The kernel derives its warp count from this
/// (`MMQF_WARPS_OF`), so the launch's block size must too.
const FEATURES_PER_WARP: u32 = 16;

const WARP_SIZE: u32 = 32;

/// The caller's say over the feature tile.
///
/// `Auto` is the production rule in [`select_tiling`]. `Force` exists for the
/// kernel A/B in `examples/quant_shape_bench.rs`, which needs both tiles at
/// one shape; no production caller passes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatTile {
    Auto,
    Force(u32),
}

/// One launch's tile geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Tiling {
    pub feat_tile: u32,
    pub mmq_x: u32,
}

impl Tiling {
    /// Threads per block: one warp per 16-feature minitile row of the tile.
    pub const fn threads(self) -> u32 {
        self.feat_tile / FEATURES_PER_WARP * WARP_SIZE
    }

    pub const fn smem_bytes(self, format: &FeatMajorFormat) -> u32 {
        smem_bytes(format, self.feat_tile, self.mmq_x)
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

    /// Kernel symbol for `role`, as the `MMQ_FM_KERNEL_AT` macro spells it:
    /// the feature-tile infix is empty at the default tile and `_y<Y>`
    /// otherwise, placed just before `_x<X>`.
    pub fn kernel_name(self, format: &FeatMajorFormat, role: Role) -> String {
        let role = match role {
            Role::TileParallel => "",
            Role::StreamK => "_sk",
            Role::Fixup => "_fixup",
        };
        let tag = if self.feat_tile == FEAT_TILE_DEFAULT {
            String::new()
        } else {
            format!("_y{}", self.feat_tile)
        };
        format!(
            "quant_mmq_{}_q8_1_mma{role}{tag}_x{}",
            format.kernel_infix, self.mmq_x
        )
    }
}

/// The three entry points each (format, tiling) compiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Role {
    /// One block per output tile, whole K.
    TileParallel,
    /// The stream-k walk over (feature tile, token tile, k-block).
    StreamK,
    /// Folds stream-k partials into the output.
    Fixup,
}

/// Dynamic shared memory one tiling needs: a `feat_tile`-row weight tile at
/// the format's stride, an `mmq_x`-row activation tile at `ACT_STRIDE`, and the
/// format's per-token activation scratch, which is zero for every format whose
/// minimum term is no finer than the record's 32-value sub-block.
///
/// Takes the whole descriptor rather than a stride so a format cannot be
/// launched with less shared memory than its kernel indexes.
pub(super) const fn smem_bytes(format: &FeatMajorFormat, feat_tile: u32, mmq_x: u32) -> u32 {
    4 * (feat_tile * format.x_stride + mmq_x * (ACT_STRIDE + format.act_scratch_ints_per_token))
}

/// Per-block dynamic shared-memory ceiling this device grants on opt-in.
///
/// The per-block attribute reports only the static default, which every variant
/// above the smallest exceeds. The per-SM figure less the driver's reservation
/// is the bound that actually applies once a function opts in.
pub(super) fn smem_opt_in_limit(shared_mem_per_unit: u32) -> u32 {
    shared_mem_per_unit.saturating_sub(1024)
}

/// Token tiles compiled at `feat_tile`.
const fn variants_at(feat_tile: u32) -> &'static [u32] {
    if feat_tile == FEAT_TILE_NARROW {
        NARROW_VARIANTS
    } else {
        VARIANTS
    }
}

/// Picks the token tile at `feat_tile` that launches the fewest token tiles
/// for `m`, breaking ties toward the smaller tile because it costs fewer
/// registers and less shared memory. `None` means no variant fits the device.
pub(super) fn select_variant(
    m: u32,
    smem_limit: u32,
    format: &FeatMajorFormat,
    feat_tile: u32,
) -> Option<u32> {
    let mut best: Option<(u32, u32)> = None;
    for &mmq_x in variants_at(feat_tile) {
        if smem_bytes(format, feat_tile, mmq_x) > smem_limit {
            continue;
        }
        let tiles = m.div_ceil(mmq_x);
        if best.is_none_or(|(_, best_tiles)| tiles < best_tiles) {
            best = Some((mmq_x, tiles));
        }
        if tiles == 1 {
            break;
        }
    }
    best.map(|(mmq_x, _)| mmq_x)
}

/// The tiling rule. `Ok(None)` means no compiled variant fits the device.
///
/// The default feature tile is tried first. The narrow tile replaces it only
/// when the format compiles it AND the default tiling is starved: fewer
/// output tiles than two waves of SMs, the same bound `use_stream_k` draws.
/// Below it the tile-parallel grid leaves SMs idle for the whole K walk, and
/// stream-k can only split what few tiles there are.
///
/// Starvation alone is not enough. Every block stages the whole `mmq_x` x
/// 256-k activation slice, whatever its feature tile, so at the narrow tile
/// half the threads copy the same slice and each block's per-group latency
/// chain grows with `mmq_x`. That is repaid only when the doubled block
/// count actually runs alongside: the narrow tile's smaller shared-memory
/// request lets more blocks co-reside per SM, so once the narrow grid
/// reaches one block per SM the extra blocks overlap each other's staging
/// stalls. So the narrow tile is taken when its token tile is small enough
/// for the longer staging chain not to matter, or when its own grid covers
/// the SMs. A wide token tile on a grid that still leaves SMs empty pays the
/// longer chain and gets nothing back.
///
/// The starvation test runs on the default tiling's own token tile; the
/// narrow tiling re-selects its token tile for `m`. Both run before the
/// stream-k decision, which then applies to whichever tiling was chosen.
///
/// A forced tile bypasses the rule; a forced tile the format does not compile
/// is an error, not a silent fallback, because a caller forcing a tile is
/// measuring that tile.
pub(super) fn select_tiling(
    m: u32,
    n: u32,
    smem_limit: u32,
    sms: u32,
    format: &FeatMajorFormat,
    feat_tile: FeatTile,
) -> Result<Option<Tiling>> {
    let at = |tile: u32| {
        select_variant(m, smem_limit, format, tile).map(|mmq_x| Tiling {
            feat_tile: tile,
            mmq_x,
        })
    };
    match feat_tile {
        FeatTile::Force(tile) if tile != FEAT_TILE_DEFAULT && tile != FEAT_TILE_NARROW => {
            Err(Error::QuantError {
                reason: format!(
                    "MMQ feature tile {tile} is not compiled for {}; force {FEAT_TILE_DEFAULT} or {FEAT_TILE_NARROW}",
                    format.kernel_infix
                ),
            })
        }
        FeatTile::Force(tile) if tile == FEAT_TILE_NARROW && !format.narrow_tile => {
            Err(Error::QuantError {
                reason: format!(
                    "MMQ feature tile {tile} is not compiled for {}; force {FEAT_TILE_DEFAULT} or use the automatic tile",
                    format.kernel_infix
                ),
            })
        }
        FeatTile::Force(tile) => Ok(at(tile)),
        FeatTile::Auto => {
            let Some(wide) = at(FEAT_TILE_DEFAULT) else {
                return Ok(None);
            };
            let starved = wide.tiles(m, n) < 2 * sms;
            if format.narrow_tile
                && starved
                && let Some(narrow) = at(FEAT_TILE_NARROW)
                && (narrow.mmq_x <= NARROW_SHORT_CHAIN_X || narrow.tiles(m, n) >= sms)
            {
                return Ok(Some(narrow));
            }
            Ok(Some(wide))
        }
    }
}

/// Whether to launch the stream-k pair rather than the tile-parallel grid.
///
/// Stream-k splits every tile's K dimension across blocks and pays a fixup
/// pass to rejoin the partials, trading that pass for the wave a ragged tile
/// count leaves half empty. Once the tiles fill the device, tile-parallel
/// wins and needs no workspace.
///
/// A format vetoes this call through `prefers_tile_parallel`, but only once
/// the tile count passes about four thirds of the SM count: past that point
/// the split saves too little to cover the fixup pass. Below that threshold
/// the tile-parallel grid cannot fill the device, and stream-k wins for every
/// format, veto or not.
///
/// K gates it too. The partial stores and the fixup pass are a fixed cost per
/// split, paid once however short the K walk is, so a short K cannot amortise
/// them and the tile-parallel grid wins even with most SMs idle. Measured on
/// every K-quant and IQ4 format at the DiT projection shapes: below
/// `STREAM_K_MIN_K` stream-k loses for every format at every tile count
/// tried; at and above it stream-k wins.
pub(super) const fn use_stream_k(tiles: u32, sms: u32, k: u32, format: &FeatMajorFormat) -> bool {
    sms > 0
        && k >= STREAM_K_MIN_K
        && tiles < 2 * sms
        && !(format.prefers_tile_parallel && 3 * tiles >= 4 * sms)
}

/// Shortest K the stream-k split is worth. See [`use_stream_k`].
const STREAM_K_MIN_K: u32 = 2048;

#[cfg(test)]
#[path = "dispatch_tests.rs"]
mod tests;
