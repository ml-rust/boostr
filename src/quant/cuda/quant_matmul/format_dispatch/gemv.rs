//! GEMV dispatch for CUDA quantized matmul, taken when `m <= gemv_max_m`.

use crate::error::{Error, Result};
use crate::quant::cuda::kernels::{
    self, GEMV_IQ1_M_MODULE, GEMV_IQ1_S_MODULE, GEMV_IQ2_S_MODULE, GEMV_IQ2_XS_MODULE,
    GEMV_IQ2_XXS_MODULE, GEMV_IQ3_S_MODULE, GEMV_IQ3_XXS_MODULE, GEMV_IQ4_NL_MODULE,
    GEMV_IQ4_XS_MODULE, GEMV_Q2_K_MODULE, GEMV_Q3_K_MODULE, GEMV_Q4_1_MODULE, GEMV_Q5_0_MODULE,
    GEMV_Q5_1_MODULE, GEMV_Q5_K_MODULE, GEMV_Q8_1_MODULE, GEMV_Q8_K_MODULE, GEMV_TQ1_0_MODULE,
    GEMV_TQ2_0_MODULE, QUANT_GEMV_MODULE,
};
use crate::quant::cuda::quant_matmul::helpers::quantize_activation_q8_1;
use crate::quant::{QuantFormat, QuantTensor};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

/// Largest `m` for which the GEMV path beats the feature-major MMQ path.
///
/// A per-token GEMV re-reads the whole weight matrix once per token, so its
/// cost scales linearly with `m`. MMQ stages a weight tile once per token
/// tile, so its cost stays flat from `m = 1` up to the tile width. The
/// token-batched dp4a kernels close most of that gap: one block decodes a
/// weight block once and dot-products it against up to a format-specific
/// number of token columns, so the weight traffic no longer grows with `m`
/// inside a tile. GEMV wins while that batched read is still cheaper than
/// staging the MMQ tile.
///
/// The crossover is per-format because decode cost per weight block differs:
/// a format with a heavier per-block unpack crosses at a lower `m`, since the
/// MMQ tile's flat cost overtakes the batched read sooner. The values below
/// are measured (see the kernel-comparison example's `--gemv` flag to compare
/// both paths at a given shape) on one GPU architecture, not derived, and
/// will need re-measuring if either kernel changes or a materially different
/// architecture is targeted.
///
/// The MMQ path needs tensor-core int8 MMA (`caps.int8_mma_m16n8k32`). On a
/// device without it, MMQ isn't available at all, so every format falls back
/// to the old, higher GEMV threshold regardless of format.
pub(in crate::quant::cuda::quant_matmul) fn gemv_max_m(
    format: QuantFormat,
    device_index: usize,
) -> usize {
    let mma_crossover = match format {
        // Lightest per-block decode: batched GEMV stays ahead of the MMQ
        // tile out to a 4-token batch.
        QuantFormat::Q8_0 | QuantFormat::Q6K => 4,
        // Heavier decode crosses earlier: MMQ already wins at m = 3.
        //
        // The four legacy 32-element formats sit in the same weight class:
        // their per-block decode is bitfield unpacking, lighter than Q4_K's
        // 6-bit scale/min pairs. They have only the `_n2` tile, so 2 is also
        // the widest batch they can serve; at m = 1 they fall through to the
        // F32 GEMV, which is the only path they have there.
        //
        // IQ4_NL and IQ4_XS join them on the same terms: their decode is the
        // same nibble unpack plus one codebook permute, and they too have only
        // the `_n2` tile and no single-token dp4a sibling. Their 2 is
        // PROVISIONAL — set to match the weight class, not yet measured, so it
        // is the first thing to re-check with the command in this module's
        // doc note.
        //
        // The six grid-indexed IQ formats join on the same terms and with the
        // same PROVISIONAL 2, but they are the heaviest decode on this path by
        // some margin: each 8-element sub-group costs a codebook read into a
        // multi-kilobyte grid table on top of the sign and scale unpack. Q2_K
        // showed that a heavy per-block decode can make batching a LOSS at
        // shallow K, so these six are the first to re-measure and the most
        // likely to come back at 1.
        QuantFormat::Q4K
        | QuantFormat::Q5K
        | QuantFormat::Q4_0
        | QuantFormat::Q5_0
        | QuantFormat::Q4_1
        | QuantFormat::Q5_1
        | QuantFormat::IQ4NL
        | QuantFormat::IQ4XS
        | QuantFormat::IQ2XXS
        | QuantFormat::IQ2XS
        | QuantFormat::IQ2S
        | QuantFormat::IQ3XXS
        | QuantFormat::IQ3S
        | QuantFormat::IQ1S => 2,
        // Heaviest decode, and neither gains from batching. Q3_K only ties
        // the MMQ tile at m = 2 and loses above it. Q2_K's small gain at a
        // deep reduction reverses into a much larger loss at a shallow one,
        // so a single crossover cannot hold for it across shapes.
        QuantFormat::Q3K | QuantFormat::Q2K => 1,
        _ => 0,
    };
    let caps = numr::runtime::cuda::CudaDevice::new(device_index)
        .profile()
        .caps;
    if caps.int8_mma_m16n8k32 {
        mma_crossover
    } else {
        16
    }
}

/// GEMV dispatch for M <= 64 (decode + short prefill).
///
/// Chooses the dp4a MWR path for the formats that have such a kernel and the
/// F32 activation path for the rest. Q4_0, Q5_0, Q4_1, Q5_1, IQ4_NL, IQ4_XS
/// and the six grid-indexed IQ formats IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S
/// and IQ1_S have only the token-batched dp4a kernel, so they take the dp4a
/// path from `m = 2` up and the F32 path at `m = 1`. Returns `Ok(None)` if the
/// format has no dedicated kernel; callers fall back to
/// `quant_matmul_via_dequant`.
pub(in crate::quant::cuda::quant_matmul) fn dispatch_gemv(
    client: &CudaClient,
    act_contig: &Tensor<CudaRuntime>,
    weight: &QuantTensor<CudaRuntime>,
    output_ptr: u64,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Option<()>> {
    let device_index = act_contig.device().id();
    let format = weight.format()?;
    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;

    // The four legacy 32-element formats, the two IQ4 codebook formats and the
    // six grid-indexed IQ formats have a token-batched dp4a kernel but no
    // single-token one: at m = 1 the batched tile's spare column is pure
    // overhead, and the F32 path below already serves that shape. So they join
    // the dp4a branch only from m = 2 up.
    let dp4a_batched_only = matches!(
        format,
        QuantFormat::Q4_0
            | QuantFormat::Q5_0
            | QuantFormat::Q4_1
            | QuantFormat::Q5_1
            | QuantFormat::IQ4NL
            | QuantFormat::IQ4XS
            | QuantFormat::IQ2XXS
            | QuantFormat::IQ2XS
            | QuantFormat::IQ2S
            | QuantFormat::IQ3XXS
            | QuantFormat::IQ3S
            | QuantFormat::IQ1S
    );

    // Every dp4a kernel walks 32-element runs, so `k % 32 == 0` is the floor.
    // IQ4_XS and the six grid-indexed IQ formats resolve a run's or
    // sub-group's byte offset through a 256-element super-block, so a row whose
    // last super-block were partial is unaddressable — and has no on-disk
    // representation either. They therefore need the stricter gate; every other
    // format on this path has a 32-element block and does not.
    let k_aligned = if matches!(
        format,
        QuantFormat::IQ4XS
            | QuantFormat::IQ2XXS
            | QuantFormat::IQ2XS
            | QuantFormat::IQ2S
            | QuantFormat::IQ3XXS
            | QuantFormat::IQ3S
            | QuantFormat::IQ1S
    ) {
        k.is_multiple_of(256)
    } else {
        k.is_multiple_of(32)
    };

    // dp4a path: formats with Q8_1 activation + dp4a MWR kernels, aligned K.
    // `k_aligned` above carries the per-format K multiple.
    if (matches!(
        format,
        QuantFormat::Q4K
            | QuantFormat::Q6K
            | QuantFormat::Q8_0
            | QuantFormat::Q5K
            | QuantFormat::Q3K
            | QuantFormat::Q2K
    ) || (dp4a_batched_only && m >= 2))
        && k_aligned
    {
        tracing::debug!(
            ?format,
            m,
            k,
            n,
            path = "dp4a_gemv",
            "CUDA quant kernel: dp4a GEMV (optimized)"
        );
        let q8_buf = quantize_activation_q8_1(client, act_contig, m, k)?;
        let q8_ptr = q8_buf.ptr();
        let weight_ptr = weight.storage().ptr();

        let module_name = match format {
            QuantFormat::Q4K | QuantFormat::Q6K | QuantFormat::Q8_0 | QuantFormat::Q4_0 => {
                QUANT_GEMV_MODULE
            }
            QuantFormat::Q5K => GEMV_Q5_K_MODULE,
            QuantFormat::Q3K => GEMV_Q3_K_MODULE,
            QuantFormat::Q2K => GEMV_Q2_K_MODULE,
            QuantFormat::Q5_0 => GEMV_Q5_0_MODULE,
            QuantFormat::Q4_1 => GEMV_Q4_1_MODULE,
            QuantFormat::Q5_1 => GEMV_Q5_1_MODULE,
            QuantFormat::IQ4NL => GEMV_IQ4_NL_MODULE,
            QuantFormat::IQ4XS => GEMV_IQ4_XS_MODULE,
            QuantFormat::IQ2XXS => GEMV_IQ2_XXS_MODULE,
            QuantFormat::IQ2XS => GEMV_IQ2_XS_MODULE,
            QuantFormat::IQ2S => GEMV_IQ2_S_MODULE,
            QuantFormat::IQ3XXS => GEMV_IQ3_XXS_MODULE,
            QuantFormat::IQ3S => GEMV_IQ3_S_MODULE,
            QuantFormat::IQ1S => GEMV_IQ1_S_MODULE,
            _ => unreachable!(),
        };

        // Most formats on this path have a token-batched variant up to their
        // own `gemv_max_m`: one block covers `tokens_per_block` token columns,
        // so a weight block is loaded and decoded once and dot-producted
        // against all of them. The per-token kernel re-reads the whole weight
        // matrix for every token, which is what makes its cost scale with M.
        // Pick the narrowest tile that covers M in one block — a wider tile
        // would idle its spare columns, a narrower one would need two passes.
        // Which widths exist is per format: Q8_0 and Q6_K crossed over out to
        // a 4-token tile, so both `_n2` and `_n4` exist; Q4_K, Q5_K, the four
        // legacy 32-element formats, the two IQ4 codebook formats and the six
        // grid-indexed IQ formats have only `_n2`; Q3_K's and Q2_K's batched
        // read never wins, so they have neither and always use the per-token
        // kernel.
        //
        // The legacy four, the two IQ4 formats and the six grid-indexed IQ
        // formats have no per-token dp4a kernel at all, so the m = 1 row below
        // never applies to them: the branch guard above already routed m = 1 to
        // the F32 path.
        let tokens_per_block: u32 = match (format, m) {
            (QuantFormat::Q3K | QuantFormat::Q2K, _) => 1,
            (_, 0..=1) => 1,
            (QuantFormat::Q8_0 | QuantFormat::Q6K, m) if m >= 3 => 4,
            _ => 2,
        };
        let kernel_name = match (format, tokens_per_block) {
            (QuantFormat::Q3K, _) => "quant_gemv_q3_k_q8_1_mwr",
            (QuantFormat::Q2K, _) => "quant_gemv_q2_k_q8_1_mwr",
            (QuantFormat::Q4K, 1) => "quant_gemv_q4_k_q8_1_mwr",
            (QuantFormat::Q6K, 1) => "quant_gemv_q6_k_q8_1_mwr",
            (QuantFormat::Q8_0, 1) => "quant_gemv_q8_0_q8_1_mwr",
            (QuantFormat::Q5K, 1) => "quant_gemv_q5_k_q8_1_mwr",
            (QuantFormat::Q4K, _) => "quant_gemv_q4_k_q8_1_mwr_n2",
            (QuantFormat::Q5K, _) => "quant_gemv_q5_k_q8_1_mwr_n2",
            (QuantFormat::Q6K, 2) => "quant_gemv_q6_k_q8_1_mwr_n2",
            (QuantFormat::Q6K, _) => "quant_gemv_q6_k_q8_1_mwr_n4",
            (QuantFormat::Q8_0, 2) => "quant_gemv_q8_0_q8_1_mwr_n2",
            (QuantFormat::Q8_0, _) => "quant_gemv_q8_0_q8_1_mwr_n4",
            (QuantFormat::Q4_0, _) => "quant_gemv_q4_0_q8_1_mwr_n2",
            (QuantFormat::Q5_0, _) => "quant_gemv_q5_0_q8_1_mwr_n2",
            (QuantFormat::Q4_1, _) => "quant_gemv_q4_1_q8_1_mwr_n2",
            (QuantFormat::Q5_1, _) => "quant_gemv_q5_1_q8_1_mwr_n2",
            (QuantFormat::IQ4NL, _) => "quant_gemv_iq4_nl_q8_1_mwr_n2",
            (QuantFormat::IQ4XS, _) => "quant_gemv_iq4_xs_q8_1_mwr_n2",
            (QuantFormat::IQ2XXS, _) => "quant_gemv_iq2_xxs_q8_1_mwr_n2",
            (QuantFormat::IQ2XS, _) => "quant_gemv_iq2_xs_q8_1_mwr_n2",
            (QuantFormat::IQ2S, _) => "quant_gemv_iq2_s_q8_1_mwr_n2",
            (QuantFormat::IQ3XXS, _) => "quant_gemv_iq3_xxs_q8_1_mwr_n2",
            (QuantFormat::IQ3S, _) => "quant_gemv_iq3_s_q8_1_mwr_n2",
            (QuantFormat::IQ1S, _) => "quant_gemv_iq1_s_q8_1_mwr_n2",
            _ => unreachable!(),
        };

        // MWR: one output column per block. The warp count follows the tile
        // width — a wide tile holds one accumulator per token in registers, so
        // it drops to two warps to keep more blocks resident per SM. This
        // mirrors `mwr_nwarps_ntok` in gemv/common.cuh; the two must agree,
        // because the kernel sizes its reduction's shared array from it.
        let block_threads = if tokens_per_block >= 8 { 64 } else { 128 };
        let cfg = LaunchConfig {
            grid_dim: (n_u32, m_u32.div_ceil(tokens_per_block), 1),
            block_dim: (block_threads, 1, 1),
            shared_mem_bytes: 0,
        };

        let module = kernels::get_or_load_module(client.context(), device_index, module_name)?;
        let func = kernels::get_kernel_function(&module, kernel_name)?;

        unsafe {
            let mut builder = client.stream().launch_builder(&func);
            builder.arg(&q8_ptr);
            builder.arg(&weight_ptr);
            builder.arg(&output_ptr);
            builder.arg(&m_u32);
            builder.arg(&k_u32);
            builder.arg(&n_u32);
            builder.launch(cfg).map_err(|e| Error::QuantError {
                reason: format!("CUDA {} launch failed: {:?}", kernel_name, e),
            })?;
        }
        return Ok(Some(()));
    }

    // F32 activation path for formats with dedicated F32 GEMV kernels
    tracing::debug!(
        ?format,
        m,
        k,
        n,
        path = "f32_gemv",
        "CUDA quant kernel: F32 GEMV (optimized)"
    );
    let act_ptr = act_contig.ptr();
    let weight_ptr = weight.storage().ptr();

    let (kernel_name, module_name) = match format {
        QuantFormat::Q4_0 => ("quant_gemv_q4_0_f32", QUANT_GEMV_MODULE),
        QuantFormat::Q8_0 => ("quant_gemv_q8_0_f32", QUANT_GEMV_MODULE),
        QuantFormat::Q4K => ("quant_gemv_q4_k_f32", QUANT_GEMV_MODULE),
        QuantFormat::Q6K => ("quant_gemv_q6_k_f32", QUANT_GEMV_MODULE),
        QuantFormat::Q5K => ("quant_gemv_q5_k_f32", GEMV_Q5_K_MODULE),
        QuantFormat::Q3K => ("quant_gemv_q3_k_f32", GEMV_Q3_K_MODULE),
        QuantFormat::Q2K => ("quant_gemv_q2_k_f32", GEMV_Q2_K_MODULE),
        QuantFormat::Q5_0 => ("quant_gemv_q5_0_f32", GEMV_Q5_0_MODULE),
        QuantFormat::IQ4NL => ("quant_gemv_iq4_nl_f32", GEMV_IQ4_NL_MODULE),
        QuantFormat::IQ4XS => ("quant_gemv_iq4_xs_f32", GEMV_IQ4_XS_MODULE),
        QuantFormat::IQ3S => ("quant_gemv_iq3_s_f32", GEMV_IQ3_S_MODULE),
        QuantFormat::IQ2XS => ("quant_gemv_iq2_xs_f32", GEMV_IQ2_XS_MODULE),
        QuantFormat::Q4_1 => ("quant_gemv_q4_1_f32", GEMV_Q4_1_MODULE),
        QuantFormat::Q5_1 => ("quant_gemv_q5_1_f32", GEMV_Q5_1_MODULE),
        QuantFormat::Q8_1 => ("quant_gemv_q8_1_f32", GEMV_Q8_1_MODULE),
        QuantFormat::Q8K => ("quant_gemv_q8_k_f32", GEMV_Q8_K_MODULE),
        QuantFormat::IQ1S => ("quant_gemv_iq1_s_f32", GEMV_IQ1_S_MODULE),
        QuantFormat::IQ1M => ("quant_gemv_iq1_m_f32", GEMV_IQ1_M_MODULE),
        QuantFormat::IQ2XXS => ("quant_gemv_iq2_xxs_f32", GEMV_IQ2_XXS_MODULE),
        QuantFormat::IQ2S => ("quant_gemv_iq2_s_f32", GEMV_IQ2_S_MODULE),
        QuantFormat::IQ3XXS => ("quant_gemv_iq3_xxs_f32", GEMV_IQ3_XXS_MODULE),
        QuantFormat::TQ1_0 => ("quant_gemv_tq1_0_f32", GEMV_TQ1_0_MODULE),
        QuantFormat::TQ2_0 => ("quant_gemv_tq2_0_f32", GEMV_TQ2_0_MODULE),
    };

    let warps_per_block = 8u32;
    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(warps_per_block), m_u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let module = kernels::get_or_load_module(client.context(), device_index, module_name)?;
    let func = kernels::get_kernel_function(&module, kernel_name)?;

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&act_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&output_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.arg(&n_u32);
        builder.launch(cfg).map_err(|e| Error::QuantError {
            reason: format!("CUDA quant_gemv kernel launch failed: {:?}", e),
        })?;
    }

    Ok(Some(()))
}
