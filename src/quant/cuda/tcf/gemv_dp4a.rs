//! Token-batched dp4a GEMV launch for TCF `Q4AS32DT64`.
//!
//! # Why this exists beside `launch_gemv`
//!
//! [`super::launch_gemv`] reconstructs each weight as an f32 and multiplies it
//! by an f32 activation, so it re-reads and re-decodes the whole weight matrix
//! once per token. This path quantizes the activation to Q8_1 once, decodes a
//! 32-element weight group once per block, and runs the dot product on dp4a —
//! the same kernel family every GGUF format already has. The kernel body is
//! `kernels/gemv/tcf_ntok.cuh`.
//!
//! # What it changes about the numbers
//!
//! The activation is QUANTIZED here, where the f32 GEMV keeps it in f32, so a
//! shape moving onto this path moves off an element-wise agreement with the
//! CPU reference and onto a directional one. That is the same trade the MMQ
//! path already makes, and `tests/backend_parity/quant_tcf.rs` gates it with
//! `assert_cosine_parity` for exactly the shapes this path selects, through a
//! `takes_tcf_dp4a_gemv_path` helper that mirrors the dispatch condition.
//!
//! # Scope
//!
//! `Q4AS32DT64` only. The kernel's lane map assumes a 32-element quantization
//! group, which is the width of a Q8_1 activation block, and its nibble map
//! assumes 4-bit adjacent-pair codes. [`supports_dp4a_gemv`] is the one place
//! that is decided; every other encoding, and every K this path cannot serve,
//! stays on `tcf_gemv_f32`.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;
use tcf_core::NativeEncoding;

use crate::error::{Error, Result};
use crate::quant::TcfEncoding;
use crate::quant::cuda::quant_matmul::helpers::quantize_activation_q8_1;

use super::super::kernels::{self, GEMV_TCF_Q4AS32DT64_MODULE};
use super::launch::{MatmulShape, matmul_setup, push_layout};

/// Widest token tile the kernel family compiles, and so the largest `m` one
/// block covers in a single pass.
pub(crate) const DP4A_GEMV_MAX_TOKENS: usize = 4;

/// Whether `encoding` at `k` takes the dp4a GEMV rather than `tcf_gemv_f32`.
///
/// Two conditions, both structural:
/// - The encoding is `Q4AS32DT64`. The kernel's lane map is written for a
///   32-element quantization group and its unpack for 4-bit adjacent-pair
///   codes; no other v1 encoding has both.
/// - `K` is a whole number of 32-element blocks, which is what the Q8_1
///   activation record is cut into. The encoding's own 64-element execution
///   tile is a stricter multiple, and `matmul_setup` checks it for every TCF
///   matmul kernel alike, so it is not restated here.
///
/// Note what is NOT required: the `K % 256 == 0` the feature-major MMQ staging
/// map takes. This kernel resolves a super-block through `tcf_group_values`,
/// which addresses by the GLOBAL flattened tile number, so a feature row need
/// not start on a super-block boundary.
pub(crate) fn supports_dp4a_gemv(encoding: TcfEncoding, k: usize) -> bool {
    matches!(encoding.native(), NativeEncoding::Q4AS32DT64) && k.is_multiple_of(32)
}

/// `activation [M, K] x weight [N, K]^T -> output [M, N]` on dp4a, one output
/// column and `NTOK` token columns per block.
///
/// # Errors
/// Every error [`matmul_setup`] raises, plus [`Error::QuantError`] when the
/// encoding is not one [`supports_dp4a_gemv`] accepts, when the activation
/// quantization fails, or when the launch fails.
pub(crate) fn launch_gemv_dp4a(
    client: &CudaClient,
    device_index: usize,
    activation: &Tensor<CudaRuntime>,
    weight_ptr: u64,
    output_ptr: u64,
    encoding: TcfEncoding,
    at: MatmulShape,
) -> Result<()> {
    if !supports_dp4a_gemv(encoding, at.k) {
        return Err(Error::QuantError {
            reason: format!(
                "{}: no dp4a GEMV for this encoding at K={}",
                encoding.name(),
                at.k
            ),
        });
    }
    let (args, m, k, n) = matmul_setup(encoding, at)?;

    // The same producer the GGUF dp4a GEMV and the `quant_mmq_*_mma` kernels
    // use, so the activation records this kernel reads are byte for byte the
    // ones those read.
    let q8_buf = quantize_activation_q8_1(client, activation, at.m, at.k)?;
    let q8_ptr = q8_buf.ptr();

    // Narrowest tile that covers M in one block: a wider one idles its spare
    // columns, a narrower one needs a second pass over the weights.
    let tokens_per_block: u32 = match at.m {
        0..=1 => 1,
        2 => 2,
        _ => 4,
    };
    let kernel_name = match tokens_per_block {
        1 => "quant_gemv_tcf_q4as32dt64_q8_1_mwr",
        2 => "quant_gemv_tcf_q4as32dt64_q8_1_mwr_n2",
        _ => "quant_gemv_tcf_q4as32dt64_q8_1_mwr_n4",
    };

    // Warps per block follow the tile width through `mwr_nwarps_ntok` in
    // `kernels/gemv/common.cuh`, which returns 4 warps for every width up to
    // 4 — the widest this family compiles. The kernel reads that same function
    // for its `__launch_bounds__` and for its reduction's shared array, so a
    // block wider than it declares is rejected outright and a narrower one
    // would leave that array partly unwritten. All three must agree, so a
    // wider tile added there has to change this number too.
    let block_threads = 4 * 32;

    let cfg = LaunchConfig {
        grid_dim: (n, m.div_ceil(tokens_per_block), 1),
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let module =
        kernels::get_or_load_module(client.context(), device_index, GEMV_TCF_Q4AS32DT64_MODULE)?;
    let func = kernels::get_kernel_function(&module, kernel_name)?;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q8_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&output_ptr);
        builder.arg(&m);
        builder.arg(&k);
        builder.arg(&n);
        push_layout!(builder, args);
        builder.launch(cfg).map_err(|e| Error::QuantError {
            reason: format!("CUDA {kernel_name} launch failed: {e:?}"),
        })?;
    }
    Ok(())
}

/// The activation contract [`launch_gemv_dp4a`] satisfies.
///
/// This kernel quantizes the caller's activation to 8-bit codes per group of
/// 32 values along K and runs the dot product on dp4a, rescaling to f32
/// afterwards. The activation the dot product sees is NOT the activation the
/// caller handed in, which is exactly what a contract declaring exact f32
/// activations forbids.
///
/// Reassociates: dp4a accumulates int32 partials per lane, then a
/// `__shfl_down_sync` tree reduces across the warp.
pub(crate) const DP4A_GEMV_CONTRACT: crate::quant::KernelContract =
    crate::quant::KernelContract::dynamic_int8_activation("cuda tcf dp4a gemv");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declares_that_it_reassociates() {
        const { assert!(DP4A_GEMV_CONTRACT.reassociates) };
    }
}
