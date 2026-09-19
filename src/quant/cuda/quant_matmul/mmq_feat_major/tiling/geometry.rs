//! Tile-geometry constants, the caller-facing [`FeatTile`] choice, the
//! [`Cadence`]/[`Role`] axes, and the shared-memory math that everything
//! else in `tiling` builds on.

use super::super::formats::FeatMajorFormat;

/// Feature tile every format compiles, at every token tile in [`VARIANTS`].
pub(in crate::quant::cuda::quant_matmul::mmq_feat_major) const FEAT_TILE_DEFAULT: u32 = 128;

/// Narrow feature tile for the CTA-starved regime. Compiled only for the
/// formats whose `narrow_tile` flag is set, and only at [`NARROW_VARIANTS`].
pub(in crate::quant::cuda::quant_matmul::mmq_feat_major) const FEAT_TILE_NARROW: u32 = 64;

/// Single-warp feature tile for the decode regime: one 16-feature minitile
/// per block, four times the blocks of the narrow tile at the same per-warp
/// work. Compiled only for the formats whose `narrow_tile` flag is set, and
/// only at [`SMALL_VARIANTS`]. Must match `MMQF_Y_SMALL` in
/// `src/quant/cuda/kernels/quant_mmq_mma.cu`.
pub(in crate::quant::cuda::quant_matmul::mmq_feat_major) const FEAT_TILE_SMALL: u32 = 16;

/// Token tiles compiled at the single-warp feature tile. Must match the
/// `MMQ_FM_KERNEL_Y16` list in the kernel file. A batch wider than the last
/// one takes the narrow or default tile.
pub(in crate::quant::cuda::quant_matmul::mmq_feat_major) const SMALL_VARIANTS: &[u32] = &[8, 16];

/// Widest token tile the kernel stages both 128-k halves of an activation
/// group at once on every feature tile and cadence, prefetching the next
/// group where the format allows; the shared-memory request doubles its
/// activation term up to here. Must match `MMQF_SMALL_X` in the kernel file.
pub(in crate::quant::cuda::quant_matmul::mmq_feat_major) const SMALL_X: u32 = 32;

/// Compiled token-tile variants at the default feature tile, ascending. Below
/// 48 the tile steps by 8, at and above it by 16; the kernel's warp blocking
/// rejects every other value.
pub(in crate::quant::cuda::quant_matmul::mmq_feat_major) const VARIANTS: &[u32] =
    &[8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128];

/// Token tiles compiled at the narrow feature tile. Stops at 64: a batch that
/// wants a wider token tile already launches enough tiles at the default
/// feature tile. Must match the `MMQ_FM_KERNEL_Y64` list in
/// `src/quant/cuda/kernels/quant_mmq_mma.cu`.
pub(in crate::quant::cuda::quant_matmul::mmq_feat_major) const NARROW_VARIANTS: &[u32] =
    &[8, 16, 24, 32, 40, 48, 64];

/// Narrowest token tile the narrow tile compiles both cadences at. Below it
/// only [`Cadence::Halves`] exists: the narrow tile keeps the default tile's
/// activation footprint and wins on block count alone. From it up the grid
/// picks the cadence — see [`Cadence`] and `super::select::select_tiling`.
/// Must match the kernel's `MMQF_GROUP_CADENCE_X`.
pub(super) const GROUP_CADENCE_MIN_X: u32 = 40;

/// Activation row stride in the shared tile, in ints: 4 half2 scale pairs plus
/// 32 quant words. The same for every weight format.
pub(super) const ACT_STRIDE: u32 = 36;

/// Output features one warp owns per MMA minitile row: the `m` of
/// `mma.m16n8k32`. The kernel derives its warp count from this
/// (`MMQF_WARPS_OF`), so the launch's block size must too.
pub(super) const FEATURES_PER_WARP: u32 = 16;

pub(super) const WARP_SIZE: u32 = 32;

/// The caller's say over the feature tile.
///
/// `Auto` is the production rule in `select::select_tiling`. The forced
/// variants exist for the kernel A/B in `examples/quant_shape_bench.rs`,
/// which needs every tiling at one shape; no production caller passes them.
/// `Force` names a feature tile (128, 64 or 16) on the two-half cadence.
/// `ForceNarrowGroup` is the narrow tile on the full-group cadence, which
/// only its wide token tiles compile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatTile {
    Auto,
    Force(u32),
    ForceNarrowGroup,
}

/// The caller's say over the launch schedule once the split count is fixed.
///
/// `Auto` is the production rule in `split_k::use_split_launch`, read with
/// the format's measured `prefers_tile_parallel`. The forced variants exist
/// for `tests/quant_mmq_tile_parallel_tune.rs`, which runs both schedules on
/// one input and compares the bits; no production caller passes them.
/// `SplitK` errors when the split count is 1, since the pair does not exist
/// there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    Auto,
    SplitK,
    TileParallel,
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
pub(in crate::quant::cuda::quant_matmul::mmq_feat_major) enum Cadence {
    Halves,
    Group,
}

/// The three entry points each (format, tiling) compiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::quant::cuda::quant_matmul::mmq_feat_major) enum Role {
    /// One block per output tile, whole K in one range.
    TileParallel,
    /// One block per output tile, the split ranges back to back.
    Fused,
    /// One block per (output tile, split range).
    SplitK,
    /// Adds the split-K partials into the output, in range order.
    Fixup,
}

/// 128-k halves of a 256-k group the activation tile holds at once: two on
/// [`Cadence::Group`] (`mmqf_accumulate_group` in the kernel file) and at
/// every token tile up to [`SMALL_X`], each a full record slice plus its
/// scratch, else one. Must match the kernel's `GROUP` template axis and its
/// `MMQF_SMALL_X` rule.
const fn act_halves(cadence: Cadence, mmq_x: u32) -> u32 {
    match cadence {
        Cadence::Halves if mmq_x > SMALL_X => 1,
        _ => 2,
    }
}

/// Whether the narrow tile compiles [`Cadence::Group`] at `mmq_x`.
pub(super) const fn group_compiled(feat_tile: u32, mmq_x: u32) -> bool {
    feat_tile == FEAT_TILE_NARROW && mmq_x >= GROUP_CADENCE_MIN_X
}

/// Dynamic shared memory one tiling needs: a `feat_tile`-row weight tile at
/// the format's stride, plus one term per staged 128-k half. Each term is an
/// `mmq_x`-row activation tile at `ACT_STRIDE` plus the format's per-token
/// activation scratch. Scratch is zero for every format whose minimum term
/// is no finer than the record's 32-value sub-block. The half count follows
/// the cadence and the token tile ([`act_halves`]).
///
/// Takes the whole descriptor rather than a stride so a format cannot be
/// launched with less shared memory than its kernel indexes.
pub(in crate::quant::cuda::quant_matmul::mmq_feat_major) const fn smem_bytes(
    format: &FeatMajorFormat,
    feat_tile: u32,
    mmq_x: u32,
    cadence: Cadence,
) -> u32 {
    4 * (feat_tile * format.x_stride
        + act_halves(cadence, mmq_x) * mmq_x * (ACT_STRIDE + format.act_scratch_ints_per_token))
}

/// Per-block dynamic shared-memory ceiling this device grants on opt-in.
///
/// The per-block attribute reports only the static default, which every variant
/// above the smallest exceeds. The per-SM figure less the driver's reservation
/// is the bound that actually applies once a function opts in.
pub(in crate::quant::cuda::quant_matmul::mmq_feat_major) fn smem_opt_in_limit(
    shared_mem_per_unit: u32,
) -> u32 {
    shared_mem_per_unit.saturating_sub(1024)
}

/// Token tiles compiled at `feat_tile`.
pub(super) const fn variants_at(feat_tile: u32) -> &'static [u32] {
    if feat_tile == FEAT_TILE_SMALL {
        SMALL_VARIANTS
    } else if feat_tile == FEAT_TILE_NARROW {
        NARROW_VARIANTS
    } else {
        VARIANTS
    }
}
