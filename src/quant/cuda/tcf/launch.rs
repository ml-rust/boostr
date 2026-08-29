//! Kernel launches for TCF native quantized weights on CUDA.
//!
//! Three entry points, one per shape of work: [`launch_dequant`] rebuilds a
//! whole tensor as f32, [`launch_gemv`] serves the decode and short-prefill
//! path, and [`launch_gemm`] serves a large batch. All three drive the same
//! device decoder in `kernels/tcf.cuh` and take the same plane offsets from
//! [`TcfLaunchArgs`], so no two of them can disagree about the layout.
//!
//! # What the GPU path does not check
//!
//! `tcf-core` rejects a payload carrying Section 13.2's reserved code or a
//! Section 13.1 invalid scale. A kernel cannot return that error per element
//! without a device-to-host round trip on every launch, so these kernels
//! decode a payload that has already been accepted — which is what reading a
//! TCF file produces, the reader having validated and digest-checked it. On a
//! payload `tcf-core` would reject, the CPU path errors and the CUDA path
//! returns numbers.
//!
//! # Rounding
//!
//! The weight reconstruction uses round-to-nearest intrinsics, so it is not
//! contracted into an FMA and not served by the approximate division that this
//! crate's `--use_fast_math` build otherwise selects. A reconstructed weight
//! therefore matches the CPU path's to the last bit. Accumulated dot products
//! still differ from the CPU's by summation order, as they do for every GGUF
//! format.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::runtime::cuda::CudaClient;

use crate::error::{Error, Result};
use crate::quant::TcfEncoding;

use super::super::kernels::{self, TCF_MODULE};
use super::layout::TcfLaunchArgs;

/// Execution tiles one dequantization block owns. One super-block, matching
/// `TCF_DEQUANT_TILES`.
const DEQUANT_TILES_PER_BLOCK: u32 = 4;
/// Threads per dequantization block: one per element of its four tiles.
const DEQUANT_BLOCK: u32 = 256;
/// Output columns one GEMV block covers, one per warp. Matches
/// `TCF_WARPS_PER_BLOCK`.
const GEMV_COLUMNS_PER_BLOCK: u32 = 8;
/// Threads per GEMV block: eight warps.
const GEMV_BLOCK: u32 = 256;
/// Output tile edge of the GEMM kernel, matching `TCF_GEMM_TM` / `TCF_GEMM_TN`.
const GEMM_TILE: u32 = 16;

/// Push the eleven layout arguments every TCF kernel takes, in the order the
/// kernel signatures declare them.
///
/// One function so the GEMV and GEMM launches cannot drift from the
/// dequantization launch on argument order — the failure that would produce is
/// a plausible tensor of wrong numbers, not an error.
macro_rules! push_layout {
    ($builder:expr, $args:expr) => {{
        $builder.arg(&$args.code_high_off);
        $builder.arg(&$args.scale_off);
        $builder.arg(&$args.min_off);
        $builder.arg(&$args.super_off);
        $builder.arg(&$args.super_min_off);
        $builder.arg(&$args.bits);
        $builder.arg(&$args.group);
        $builder.arg(&$args.groups_per_tile);
        $builder.arg(&$args.symmetric);
        $builder.arg(&$args.scale_form);
        $builder.arg(&$args.sub_block_bytes);
    }};
}

/// Dequantize a whole TCF payload into `output_ptr`, `product(shape)` f32.
///
/// # Errors
/// [`Error::QuantError`] when the layout is not one the kernels decode, when
/// the tile count exceeds `u32`, or when the launch fails.
pub(crate) fn launch_dequant(
    client: &CudaClient,
    device_index: usize,
    payload_ptr: u64,
    output_ptr: u64,
    encoding: TcfEncoding,
    shape: &[usize],
) -> Result<()> {
    let args = TcfLaunchArgs::new(encoding, shape)?;
    let tiles = u32::try_from(args.tiles).map_err(|_| Error::QuantError {
        reason: format!("{}: {} tiles exceed u32", encoding.name(), args.tiles),
    })?;
    if tiles == 0 {
        return Ok(());
    }

    let module = kernels::get_or_load_module(client.context(), device_index, TCF_MODULE)?;
    let func = kernels::get_kernel_function(&module, "tcf_dequant_f32")?;

    let cfg = LaunchConfig {
        grid_dim: (tiles.div_ceil(DEQUANT_TILES_PER_BLOCK), 1, 1),
        block_dim: (DEQUANT_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&payload_ptr);
        builder.arg(&output_ptr);
        builder.arg(&tiles);
        push_layout!(builder, args);
        builder.launch(cfg).map_err(|e| Error::QuantError {
            reason: format!(
                "CUDA tcf_dequant_f32 launch failed for {}: {e:?}",
                encoding.name()
            ),
        })?;
    }
    Ok(())
}

/// The `[m, n]` output geometry both matmul launches share.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MatmulShape {
    /// Activation rows.
    pub m: usize,
    /// Shared dimension, a whole number of execution tiles.
    pub k: usize,
    /// Weight rows, and output columns.
    pub n: usize,
}

/// `activation [M, K] x weight [N, K]^T -> output [M, N]`, one warp per output
/// column. The path for a decode step or a short prefill.
///
/// # Errors
/// Every error [`TcfLaunchArgs::new`] raises, plus [`Error::QuantError`] when
/// a dimension exceeds `u32` or the launch fails.
pub(crate) fn launch_gemv(
    client: &CudaClient,
    device_index: usize,
    act_ptr: u64,
    weight_ptr: u64,
    output_ptr: u64,
    encoding: TcfEncoding,
    at: MatmulShape,
) -> Result<()> {
    let (args, m, k, n) = matmul_setup(encoding, at)?;
    let module = kernels::get_or_load_module(client.context(), device_index, TCF_MODULE)?;
    let func = kernels::get_kernel_function(&module, "tcf_gemv_f32")?;

    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(GEMV_COLUMNS_PER_BLOCK), m, 1),
        block_dim: (GEMV_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&act_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&output_ptr);
        builder.arg(&m);
        builder.arg(&k);
        builder.arg(&n);
        push_layout!(builder, args);
        builder.launch(cfg).map_err(|e| Error::QuantError {
            reason: format!(
                "CUDA tcf_gemv_f32 launch failed for {}: {e:?}",
                encoding.name()
            ),
        })?;
    }
    Ok(())
}

/// `activation [M, K] x weight [N, K]^T -> output [M, N]`, a 16x16 output tile
/// per block with the weight decoded once per block into shared memory. The
/// path for a large batch.
///
/// # Errors
/// Every error [`launch_gemv`] raises.
pub(crate) fn launch_gemm(
    client: &CudaClient,
    device_index: usize,
    act_ptr: u64,
    weight_ptr: u64,
    output_ptr: u64,
    encoding: TcfEncoding,
    at: MatmulShape,
) -> Result<()> {
    let (args, m, k, n) = matmul_setup(encoding, at)?;
    let module = kernels::get_or_load_module(client.context(), device_index, TCF_MODULE)?;
    let func = kernels::get_kernel_function(&module, "tcf_gemm_f32")?;

    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(GEMM_TILE), m.div_ceil(GEMM_TILE), 1),
        block_dim: (GEMM_TILE, GEMM_TILE, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&act_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&output_ptr);
        builder.arg(&m);
        builder.arg(&k);
        builder.arg(&n);
        push_layout!(builder, args);
        builder.launch(cfg).map_err(|e| Error::QuantError {
            reason: format!(
                "CUDA tcf_gemm_f32 launch failed for {}: {e:?}",
                encoding.name()
            ),
        })?;
    }
    Ok(())
}

/// Layout arguments and `u32`-narrowed dimensions for a matmul launch.
///
/// `K` must be a whole number of execution tiles: the kernels walk a weight
/// row tile by tile, and a partial trailing tile would read a neighbouring
/// row's codes.
fn matmul_setup(encoding: TcfEncoding, at: MatmulShape) -> Result<(TcfLaunchArgs, u32, u32, u32)> {
    let name = encoding.name();
    let tile = encoding.tile();
    if tile == 0 || at.k == 0 || !at.k.is_multiple_of(tile) {
        return Err(Error::QuantError {
            reason: format!(
                "{name}: K={} is not a positive multiple of the tile width {tile}",
                at.k
            ),
        });
    }
    let args = TcfLaunchArgs::new(encoding, &[at.n, at.k])?;
    let narrow = |value: usize, label: &str| -> Result<u32> {
        u32::try_from(value).map_err(|_| Error::QuantError {
            reason: format!("{name}: {label}={value} exceeds u32"),
        })
    };
    Ok((
        args,
        narrow(at.m, "M")?,
        narrow(at.k, "K")?,
        narrow(at.n, "N")?,
    ))
}
