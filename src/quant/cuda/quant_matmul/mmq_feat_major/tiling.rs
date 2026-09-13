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

/// Narrowest token tile the narrow tile compiles both cadences at. Below it
/// only [`Cadence::Halves`] exists: the narrow tile keeps the default tile's
/// activation footprint and wins on block count alone. From it up the grid
/// picks the cadence — see [`Cadence`] and [`select_tiling`]. Must match the
/// kernel's `MMQF_GROUP_CADENCE_X`.
const GROUP_CADENCE_MIN_X: u32 = 40;

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
/// `Auto` is the production rule in [`select_tiling`]. The forced variants
/// exist for the kernel A/B in `examples/quant_shape_bench.rs`, which needs
/// every tiling at one shape; no production caller passes them. `Force`
/// names a feature tile on the two-half cadence. `ForceNarrowGroup` is the
/// narrow tile on the full-group cadence, which only its wide token tiles
/// compile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatTile {
    Auto,
    Force(u32),
    ForceNarrowGroup,
}

/// How a block walks the two 128-k halves of each 256-k activation group.
///
/// `Halves` stages one half at a time into one activation tile: four
/// barriers per group, and the second half's global load waits on the first
/// half's consumers. `Group` stages both halves into a doubled activation
/// tile: two barriers per group and one exposed round trip. The doubled tile
/// costs block co-residency, so `Group` wins only where nothing co-resides
/// anyway: a grid under one block per SM. The default feature tile has no
/// room for the doubled tile and is always `Halves`. The narrow tile
/// compiles `Group` from [`GROUP_CADENCE_MIN_X`] up, where per-group
/// activation staging is long enough to dominate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Cadence {
    Halves,
    Group,
}

/// One launch's tile geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Tiling {
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

/// 128-k halves of a 256-k group the activation tile holds at once: two on
/// [`Cadence::Group`] (`mmqf_accumulate_group` in the kernel file), each a
/// full record slice plus its scratch, else one. Must match the kernel's
/// `GROUP` template axis.
const fn act_halves(cadence: Cadence) -> u32 {
    match cadence {
        Cadence::Halves => 1,
        Cadence::Group => 2,
    }
}

/// Whether the narrow tile compiles [`Cadence::Group`] at `mmq_x`.
const fn group_compiled(feat_tile: u32, mmq_x: u32) -> bool {
    feat_tile == FEAT_TILE_NARROW && mmq_x >= GROUP_CADENCE_MIN_X
}

/// Dynamic shared memory one tiling needs: a `feat_tile`-row weight tile at
/// the format's stride, plus one term per staged 128-k half. Each term is an
/// `mmq_x`-row activation tile at `ACT_STRIDE` plus the format's per-token
/// activation scratch. Scratch is zero for every format whose minimum term
/// is no finer than the record's 32-value sub-block. The half count follows
/// the cadence.
///
/// Takes the whole descriptor rather than a stride so a format cannot be
/// launched with less shared memory than its kernel indexes.
pub(super) const fn smem_bytes(
    format: &FeatMajorFormat,
    feat_tile: u32,
    mmq_x: u32,
    cadence: Cadence,
) -> u32 {
    4 * (feat_tile * format.x_stride
        + act_halves(cadence) * mmq_x * (ACT_STRIDE + format.act_scratch_ints_per_token))
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
///
/// The fit test uses `cadence`'s shared-memory request. The caller that wants
/// the group cadence must ask with it: its doubled activation tile can push
/// a token tile over the limit that the halves cadence keeps.
pub(super) fn select_variant(
    m: u32,
    smem_limit: u32,
    format: &FeatMajorFormat,
    feat_tile: u32,
    cadence: Cadence,
) -> Option<u32> {
    let mut best: Option<(u32, u32)> = None;
    for &mmq_x in variants_at(feat_tile) {
        if smem_bytes(format, feat_tile, mmq_x, cadence) > smem_limit {
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
/// The default feature tile is tried first and is the answer for every
/// format without the narrow tile. Otherwise the narrow tile's own token
/// tile for `m` decides which lever is in play:
///
/// - Short token tile (below [`GROUP_CADENCE_MIN_X`]): only the two-half
///   cadence exists, with the default tile's activation footprint, so the
///   narrow tile is pure block count. It is taken exactly when the default
///   tiling is starved: fewer output tiles than two waves of SMs, the bound
///   `use_stream_k` draws. Below it the tile-parallel grid leaves SMs idle
///   for the whole K walk, and stream-k can only split what few tiles there
///   are.
/// - Wide token tile: per-group activation staging dominates, and the grid
///   picks the cadence. Under one block per SM nothing co-resides, so the
///   full-group cadence's halved barriers are pure gain. At or above it the
///   two-half cadence keeps the smaller activation tile and the
///   co-residency the group cadence would spend. It is taken while the
///   default tiling is still starved; past starvation the default tile
///   stays.
///
/// The starvation test runs on the default tiling's own token tile; the
/// narrow tiling re-selects its token tile for `m`. All of it runs before the
/// stream-k decision, which then applies to whichever tiling was chosen.
///
/// A forced tiling bypasses the rule. A forced tiling the format does not
/// compile is an error, not a silent fallback: a caller forcing a tiling is
/// measuring that tiling.
pub(super) fn select_tiling(
    m: u32,
    n: u32,
    smem_limit: u32,
    sms: u32,
    format: &FeatMajorFormat,
    feat_tile: FeatTile,
) -> Result<Option<Tiling>> {
    let at = |tile: u32, cadence: Cadence| {
        select_variant(m, smem_limit, format, tile, cadence).map(|mmq_x| Tiling {
            feat_tile: tile,
            mmq_x,
            cadence,
        })
    };
    let not_compiled = |what: String, hint: &str| Error::QuantError {
        reason: format!(
            "MMQ {what} is not compiled for {}; {hint}",
            format.kernel_infix
        ),
    };
    match feat_tile {
        FeatTile::Force(tile) if tile != FEAT_TILE_DEFAULT && tile != FEAT_TILE_NARROW => {
            Err(not_compiled(
                format!("feature tile {tile}"),
                &format!("force {FEAT_TILE_DEFAULT} or {FEAT_TILE_NARROW}"),
            ))
        }
        FeatTile::Force(tile) if tile == FEAT_TILE_NARROW && !format.narrow_tile => {
            Err(not_compiled(
                format!("feature tile {tile}"),
                &format!("force {FEAT_TILE_DEFAULT} or use the automatic tile"),
            ))
        }
        FeatTile::Force(tile) => Ok(at(tile, Cadence::Halves)),
        FeatTile::ForceNarrowGroup if !format.narrow_tile => Err(not_compiled(
            format!("feature tile {FEAT_TILE_NARROW}"),
            &format!("force {FEAT_TILE_DEFAULT} or use the automatic tile"),
        )),
        FeatTile::ForceNarrowGroup => {
            let Some(narrow) = at(FEAT_TILE_NARROW, Cadence::Group) else {
                return Ok(None);
            };
            if !group_compiled(narrow.feat_tile, narrow.mmq_x) {
                return Err(not_compiled(
                    format!(
                        "group cadence at feature tile {FEAT_TILE_NARROW} token tile {}",
                        narrow.mmq_x
                    ),
                    &format!("it exists from token tile {GROUP_CADENCE_MIN_X} up; raise m"),
                ));
            }
            Ok(Some(narrow))
        }
        FeatTile::Auto => {
            let Some(wide) = at(FEAT_TILE_DEFAULT, Cadence::Halves) else {
                return Ok(None);
            };
            if !format.narrow_tile {
                return Ok(Some(wide));
            }
            let Some(narrow) = at(FEAT_TILE_NARROW, Cadence::Halves) else {
                return Ok(Some(wide));
            };
            let starved = wide.tiles(m, n) < 2 * sms;
            if group_compiled(narrow.feat_tile, narrow.mmq_x) {
                // Same token tile on the group cadence, if its doubled
                // activation tile still fits the device.
                let group = Tiling {
                    cadence: Cadence::Group,
                    ..narrow
                };
                if narrow.tiles(m, n) < sms && group.smem_bytes(format) <= smem_limit {
                    return Ok(Some(group));
                }
            }
            Ok(Some(if starved { narrow } else { wide }))
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
