//! Per-device measurement of the split-K pair against the tile-parallel
//! grid: the `prefers_tile_parallel` value `split_k::use_split_launch` reads.
//!
//! Both schedules sum the same K-range partials in the same order, so the
//! pick is speed only (`tests/quant_mmq_tile_parallel_tune.rs` and
//! `tests/quant_mmq_batch_invariance.rs` hold the bit checks). Which one wins
//! depends on the format's staging cost and on the part, so
//! [`prefers_tile_parallel`] times both on the device through
//! `numr::runtime::cuda::tune::tuned`, once per (device, format), at one
//! geometry where the flag decides the launch. The descriptor's
//! `prefers_tile_parallel_fallback` is the constant measured on one part; it
//! serves when tuning is off (`NUMR_CUDA_TUNE=0`) or the probe fails.
//!
//! Cost: one probe per (device, format) on the first dispatch that reaches
//! the pair-vs-grid decision: one warm-up and three timed launches of each
//! schedule on a weight of a few MB, a few ms in all; blazr's warmup absorbs
//! it. Later calls read the tune cache.

use numr::dtype::DType;
use numr::runtime::cuda::tune::{time_launches, tuned};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::runtime::{Device, RuntimeClient};
use numr::tensor::Tensor;

use crate::error::{Error, Result};
use crate::quant::cuda::kernels::{self, QUANT_MMQ_MMA_MODULE};
use crate::quant::cuda::quant_matmul::helpers::quantize_activation_q8_1_mmq;

use super::super::formats::FeatMajorFormat;
use super::super::launch::Launch;
use super::geometry::{FEAT_TILE_DEFAULT, FEAT_TILE_NARROW, FeatTile, VARIANTS, smem_opt_in_limit};
use super::select::select_tiling;
use super::split_k::{split_count, use_split_launch};
use super::tile::Tiling;

/// Depth of the probe's K walk: 16 staging groups, so the split count comes
/// from the SM count and the range floor never caps it. Rounded up to the
/// format's `k_multiple`, which every compiled format divides.
const PROBE_K: u32 = 4096;

/// Timed iterations per schedule; `time_launches` takes the minimum.
const PROBE_ITERS: usize = 3;

/// Batch sizes tried in order: the decode batch, then the smallest batch
/// with two token tiles at the default feature tile.
///
/// The flag only decides the launch where the pair is admitted (fewer
/// default feature tiles than SMs) AND the veto term fires (narrow-tile
/// equivalents past four thirds of the SM count). One token tile at the
/// default feature tile cannot meet both, so a format without the
/// single-warp tile is probed at two token tiles.
const PROBE_M: [u32; 2] = [8, VARIANTS[VARIANTS.len() - 1] + 1];

/// The geometry the probe times.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProbeShape {
    m: u32,
    n: u32,
    k: u32,
    tiling: Tiling,
    splits: u32,
}

/// Whether `fm`'s tile-parallel grid outruns its split-K pair on `client`'s
/// device.
///
/// The first call per (device, format) runs [`probe`]; later calls read the
/// tune cache. With tuning off, or when the probe returns `Err`, this is the
/// descriptor's `prefers_tile_parallel_fallback`.
pub(in crate::quant::cuda::quant_matmul) fn prefers_tile_parallel(
    client: &CudaClient,
    fm: &FeatMajorFormat,
) -> bool {
    tuned(
        client,
        fm.tile_parallel_key,
        fm.prefers_tile_parallel_fallback,
        || probe(client, fm).map_err(|e| numr::error::Error::Internal(e.to_string())),
    )
}

/// The `(m, n, k)` the probe times for `fm` on `client`'s device, so a test
/// can run both schedules on real inputs at the same geometry.
pub(in crate::quant::cuda::quant_matmul) fn tile_parallel_probe_shape(
    client: &CudaClient,
    fm: &FeatMajorFormat,
) -> Result<(u32, u32, u32)> {
    let profile = client.device().profile();
    let shape = probe_shape(
        fm,
        smem_opt_in_limit(profile.shared_mem_per_unit),
        profile.compute_units,
    )?;
    Ok((shape.m, shape.n, shape.k))
}

/// One geometry where the flag decides the launch: the tiling rule admits
/// the pair when the flag is off and vetoes it when the flag is on.
///
/// `n` puts the narrow-tile-equivalent tile count in the middle of the veto
/// band, five thirds of the SM count. The tiling is selected with the flag
/// off, so nothing here reads the value under measurement.
fn probe_shape(fm: &FeatMajorFormat, smem_limit: u32, sms: u32) -> Result<ProbeShape> {
    let fail = |why: &str| Error::QuantError {
        reason: format!("MMQ {} tile-parallel probe: {why}", fm.kernel_infix),
    };
    if sms == 0 {
        return Err(fail("device profile reports 0 compute units"));
    }
    let k = PROBE_K.next_multiple_of(fm.k_multiple);
    for m in PROBE_M {
        // `n` reaches the tiling rule only through its starvation test, so a
        // provisional width picks the tiling; the real width is re-checked
        // against it below.
        let Some(tiling) = select_tiling(
            m,
            FEAT_TILE_DEFAULT,
            k,
            smem_limit,
            sms,
            fm,
            FeatTile::Auto,
            false,
        )?
        else {
            continue;
        };
        // Narrow-tile equivalents per default feature tile of `n`: the unit
        // `use_split_launch` counts.
        let per_column =
            tiling.token_tiles(m) * (FEAT_TILE_DEFAULT / tiling.feat_tile.max(FEAT_TILE_NARROW));
        if per_column < 2 {
            continue;
        }
        let n = 5 * sms / (3 * per_column) * FEAT_TILE_DEFAULT;
        let Some(at_n) = select_tiling(m, n, k, smem_limit, sms, fm, FeatTile::Auto, false)? else {
            continue;
        };
        let tiles = at_n.tiles(m, n);
        let splits = split_count(k, n, sms);
        let pair = use_split_launch(splits, tiles, at_n.feat_tile, sms, false);
        let vetoed = !use_split_launch(splits, tiles, at_n.feat_tile, sms, true);
        if at_n == tiling && pair && vetoed {
            return Ok(ProbeShape {
                m,
                n,
                k,
                tiling: at_n,
                splits,
            });
        }
    }
    Err(fail(
        "no probed batch size reaches a geometry where the launch pick decides",
    ))
}

/// Time both schedules at the probe geometry and return whether the
/// tile-parallel grid ran no slower than the pair.
///
/// The buffers are left uninitialized: the kernels' cost does not depend on
/// the values, and nothing reads the product. The pair's time includes its
/// per-launch workspace allocation, as the dispatch pays it. Nothing on this
/// path calls [`prefers_tile_parallel`]: `tuned` holds its probe lock while
/// this runs.
fn probe(client: &CudaClient, fm: &FeatMajorFormat) -> Result<bool> {
    let device = client.device();
    let profile = device.profile();
    let ProbeShape {
        m,
        n,
        k,
        tiling,
        splits,
    } = probe_shape(
        fm,
        smem_opt_in_limit(profile.shared_mem_per_unit),
        profile.compute_units,
    )?;

    let act = Tensor::<CudaRuntime>::empty(&[m as usize, k as usize], DType::F32, device)?;
    let (q8, ntok) =
        quantize_activation_q8_1_mmq(client, &act, m as usize, k as usize, tiling.mmq_x as usize)?;
    let weight_bytes = fm.quant_format.storage_bytes(n as usize * k as usize)?;
    let weight = Tensor::<CudaRuntime>::empty(&[weight_bytes], DType::U8, device)?;
    let output = Tensor::<CudaRuntime>::empty(&[m as usize, n as usize], DType::F32, device)?;
    let module = kernels::get_or_load_module(client.context(), device.id(), QUANT_MMQ_MMA_MODULE)?;

    let launch = Launch {
        format: fm,
        client,
        module: &module,
        output_ptr: output.ptr(),
        q8_ptr: q8.ptr(),
        weight_ptr: weight.ptr(),
        m,
        k,
        n,
        ntok,
        tiling,
        smem: tiling.smem_bytes(fm),
        grid: (tiling.token_tiles(m), tiling.feat_tiles(n)),
        splits,
    };
    // `time_launches` takes numr's error type; the launch errors are only
    // reported through `tuned`'s fallback log.
    let to_numr = |e: Error| numr::error::Error::Internal(e.to_string());
    let pair_us = time_launches(client, PROBE_ITERS, || {
        launch.split_k(device).map_err(to_numr)
    })?;
    let grid_us = time_launches(client, PROBE_ITERS, || {
        launch.tile_parallel().map_err(to_numr)
    })?;
    tracing::debug!(
        m,
        n,
        k,
        splits,
        pair_us,
        grid_us,
        weight_format = fm.kernel_infix,
        "CUDA quant kernel: tile-parallel probe"
    );
    Ok(grid_us <= pair_us)
}

#[cfg(test)]
mod tests {
    use super::super::super::formats::{
        IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, PQ2_0, PTQ1_0, Q1_0, Q2_0,
        Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0,
    };
    use super::super::geometry::FEAT_TILE_SMALL;
    use super::*;
    use std::collections::HashSet;

    /// Every descriptor the family compiles.
    const ALL: [&FeatMajorFormat; 22] = [
        &Q8_0, &Q4_0, &Q4_1, &Q5_0, &Q5_1, &IQ4_NL, &PQ2_0, &Q2_0, &Q1_0, &PTQ1_0, &Q4_K, &Q5_K,
        &Q6_K, &Q3_K, &Q2_K, &IQ4_XS, &IQ2_XXS, &IQ2_XS, &IQ2_S, &IQ3_XXS, &IQ3_S, &IQ1_S,
    ];

    /// Ample limit: every compiled variant fits.
    const WIDE: u32 = 1 << 20;

    /// SM count the geometry cases below are written against.
    const SMS: u32 = 28;

    #[test]
    fn every_descriptor_has_its_own_key() {
        let mut seen = HashSet::new();
        for fm in ALL {
            assert_eq!(
                fm.tile_parallel_key,
                format!("mmq_feat_major.{}.prefers_tile_parallel", fm.kernel_infix)
            );
            assert!(
                seen.insert(fm.tile_parallel_key),
                "{} repeats",
                fm.kernel_infix
            );
        }
        assert_eq!(seen.len(), ALL.len());
    }

    #[test]
    fn the_second_probe_batch_is_the_first_with_two_token_tiles() {
        assert_eq!(PROBE_M, [8, 129]);
        assert!(VARIANTS.iter().all(|&x| 129_u32.div_ceil(x) >= 2));
    }

    #[test]
    fn a_single_warp_format_is_probed_at_the_decode_batch() {
        // Q4_K at m=8 takes the single-warp tile: 2 narrow-tile equivalents
        // per default feature tile, so 23 columns put 46 equivalents in the
        // veto band against the fixture's `SMS` (28). 23 feature tiles are
        // under the fixture's SM count, so the pair is admitted with 2
        // ranges of K=4096.
        let shape = probe_shape(&Q4_K, WIDE, SMS).expect("probe shape");
        assert_eq!((shape.m, shape.n, shape.k), (8, 2944, 4096));
        assert_eq!(shape.tiling.feat_tile, FEAT_TILE_SMALL);
        assert_eq!(shape.splits, 2);
    }

    #[test]
    fn a_default_tile_format_is_probed_at_two_token_tiles() {
        // Q8_0 has one feature tile: at m=8 one token tile times 23 feature
        // tiles never reaches the veto band, so the probe moves to m=129,
        // where x80 gives two token tiles and the same 46 equivalents.
        let shape = probe_shape(&Q8_0, WIDE, SMS).expect("probe shape");
        assert_eq!((shape.m, shape.n, shape.k), (129, 2944, 4096));
        assert_eq!(shape.tiling.feat_tile, FEAT_TILE_DEFAULT);
        assert_eq!(shape.tiling.token_tiles(shape.m), 2);
        assert_eq!(shape.splits, 2);
    }

    #[test]
    fn the_flag_decides_the_launch_at_every_probe_shape() {
        for fm in ALL {
            let shape = probe_shape(fm, WIDE, SMS).expect(fm.kernel_infix);
            let tiles = shape.tiling.tiles(shape.m, shape.n);
            assert!(use_split_launch(
                shape.splits,
                tiles,
                shape.tiling.feat_tile,
                SMS,
                false
            ));
            assert!(!use_split_launch(
                shape.splits,
                tiles,
                shape.tiling.feat_tile,
                SMS,
                true
            ));
            assert_eq!(shape.k % fm.k_multiple, 0, "{}", fm.kernel_infix);
        }
    }

    #[test]
    fn the_probe_weight_stays_under_sixteen_megabytes_on_the_measured_part() {
        for fm in ALL {
            let shape = probe_shape(fm, WIDE, SMS).expect(fm.kernel_infix);
            let bytes = fm
                .quant_format
                .storage_bytes(shape.n as usize * shape.k as usize)
                .expect("whole blocks");
            assert!(
                bytes < 16 << 20,
                "{} probe weight is {bytes} bytes",
                fm.kernel_infix
            );
        }
    }

    #[test]
    fn no_sm_count_is_an_error() {
        assert!(probe_shape(&Q8_0, WIDE, 0).is_err());
    }
}
