//! Kernel choice for a TCF-encoded weight inside `quant_matmul`.
//!
//! TCF weights take their own kernels: a GGUF kernel finds a block's codes and
//! its scale adjacent, while TCF spreads them over whole-tensor planes. The
//! choice between those kernels is its own decision with its own crossovers,
//! so it lives here rather than inline in the `QuantMatmulOps` impl.

use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::error::Result;
use crate::quant::{QuantTensor, TcfEncoding};

use super::super::tcf::{self as tcf_dispatch, MatmulShape};
use super::mmq_feat_major;

/// Smallest token count that takes the feature-major MMQ kernel rather than
/// the f32 GEMV.
///
/// The MMQ kernel is flat in `m` across a token tile, while the f32 GEMV
/// re-reads the weight per token, so the GEMV only wins where the tile would
/// be nearly empty.
///
/// Measured: the MMQ kernel wins at every token count from this constant
/// upward, on both benchmarked projection shapes. `m = 1` stays on the GEMV.
///
/// Re-measure: `cargo bench --features cuda --bench quant_throughput --
/// --backend cuda --filter Q8`, comparing the `tcf` rows against the `gguf`
/// rows at each `m`.
const TCF_FEAT_MAJOR_MIN_M: usize = 2;

/// Largest token count that takes the TCF dp4a GEMV rather than any kernel
/// above it.
///
/// This is the ONE place the dp4a-GEMV/MMQ boundary is set. Moving the
/// boundary is editing this constant and nothing else.
///
/// `None` is "no token count takes it". **Measured off.** The dp4a GEMV is compiled and parity-checked but wins no
/// token count, so it routes nowhere: `m = 1` keeps the f32 GEMV and every
/// larger `m` keeps the feature-major MMQ, which is where they were before the
/// kernel existed.
///
/// The reason is that the TCF GEMV is bound by memory access, not by
/// arithmetic, so replacing its f32 multiply-accumulate with an integer dp4a
/// changes the term that was never the cost. Both TCF GEMVs read the code
/// plane and the scale plane as two streams far apart in the tensor, while a
/// GGUF block holds its codes and its scale adjacent; at one token there is no
/// reuse to hide that distance, so TCF trails the GGUF kernel over the same
/// weight byte count. Until that access pattern changes, no arithmetic
/// substitution moves this constant.
///
/// Raising it above `tcf::DP4A_GEMV_MAX_TOKENS` asks for a token tile the
/// kernel family does not compile, which the assertion below refuses.
///
/// Re-measure: `cargo bench --features cuda --bench quant_throughput --
/// --backend cuda --filter Q4`, comparing the TCF row's `ns*` column with this
/// constant set to 4 and then to 0. Read `ns*`, NOT the `tcf/gguf` ratio: that
/// ratio is built from retired host instructions, which on a CUDA row count
/// kernel LAUNCH work rather than kernel work.
const TCF_DP4A_GEMV_MAX_M: Option<usize> = None;
/// Compile-time proof that the boundary never names a token tile the kernel
/// family does not compile: this subtraction underflows, and const evaluation
/// fails, if [`TCF_DP4A_GEMV_MAX_M`] is raised past the widest compiled width.
const _: usize = match TCF_DP4A_GEMV_MAX_M {
    Some(max) => tcf_dispatch::DP4A_GEMV_MAX_TOKENS - max,
    None => 0,
};

/// Run `act_contig [m, k] x weight [n, k]^T -> output_ptr [m, n]` on whichever
/// TCF kernel covers the shape.
///
/// TCF has three crossovers, not one, and none of them is the `M <= 64` the
/// GGUF path uses:
/// - [`TCF_DP4A_GEMV_MAX_M`] takes the encodings that have a token-batched
///   dp4a GEMV at the smallest token counts, ahead of everything else.
/// - [`TCF_FEAT_MAJOR_MIN_M`] gates the encodings that have a feature-major
///   kernel onto MMQ ahead of the f32 tiles.
/// - The `m <= 4` check below chooses `launch_gemv` against `launch_gemm` for
///   every encoding that is on neither of those — an encoding that has a
///   kernel falls back to it too when the MMQ dispatch declines.
///
/// The mechanism behind the GEMV/GEMM split, which is what carries across
/// devices: GEMV cost grows linearly in M because it re-reads the weights once
/// per row, while the register-blocked GEMM computes a 4x4 output patch per
/// thread and stays nearly flat in M, so the two cross at a small M on every
/// encoding.
///
/// Re-measure each crossover whenever its kernel changes — a speedup on one
/// side moves it, and all three values are measured constants, not derived
/// ones.
///
/// Every arm checks the weight's declared activation contract against the
/// contract the kernel it just selected satisfies, BEFORE launching. Section 9
/// defines no float fallback, so a mismatch stops the operation instead of
/// moving the weight to a kernel with different arithmetic.
///
/// # Errors
/// [`crate::error::Error::ActivationContractMismatch`] when the selected
/// kernel does not satisfy the weight's declared contract, and whatever the
/// selected launch raises otherwise.
pub(super) fn route_tcf(
    client: &CudaClient,
    encoding: TcfEncoding,
    act_contig: &Tensor<CudaRuntime>,
    weight: &QuantTensor<CudaRuntime>,
    output_ptr: u64,
    at: MatmulShape,
) -> Result<()> {
    let MatmulShape { m, k, n } = at;
    let device_index = act_contig.device().id();

    // The dp4a GEMV is checked FIRST, ahead of MMQ: at a small token count it
    // decodes each weight group once per block into int8 and never stages a
    // tile, which is the cheaper shape of work there. `supports_dp4a_gemv`
    // covers the encodings and K it serves; everything else keeps the f32
    // kernels below.
    if TCF_DP4A_GEMV_MAX_M.is_some_and(|max| m <= max)
        && tcf_dispatch::supports_dp4a_gemv(encoding, k)
    {
        // This kernel quantizes the activation, so it serves only a weight
        // whose declared contract asks for that. There is no float fallback
        // to drop to (Section 9): a weight declaring exact f32 activations is
        // refused here rather than routed on to the kernels below, which
        // would silently give it different arithmetic than the one it was
        // just found unfit for.
        weight.check_activation_contract(&tcf_dispatch::DP4A_GEMV_CONTRACT)?;
        return tcf_dispatch::launch_gemv_dp4a(
            client,
            device_index,
            act_contig,
            weight.storage().ptr(),
            output_ptr,
            encoding,
            at,
        );
    }

    if m >= TCF_FEAT_MAJOR_MIN_M {
        // An encoding with a `FeatMajorFormat` takes the feature-major
        // tensor-core family instead of the f32 FMA tile `launch_gemm` runs;
        // `mmq_feat_major::feat_major_format` is the one place that says which
        // those are. Every other encoding keeps that tile, as does any K the
        // format's staging map does not cover. `Ok(None)` means no compiled
        // variant fits the device, and `launch_gemm` still serves the shape.
        let feat_major = mmq_feat_major::feat_major_format(encoding.native())
            .filter(|format| k.is_multiple_of(format.k_multiple as usize))
            .filter(|_| {
                numr::runtime::cuda::CudaDevice::new(device_index)
                    .profile()
                    .caps
                    .int8_mma_m16n8k32
            });
        if let Some(format) = feat_major {
            // The feature-major family quantizes the activation to the 8-bit
            // dynamic record its MMA instructions consume. Checking BEFORE
            // the dispatch, not after, is what makes this a refusal: a
            // mismatch must not fall through to the f32 tiles below, because
            // that would answer "no kernel satisfies this contract" with a
            // different kernel rather than with an error.
            weight.check_activation_contract(&mmq_feat_major::CONTRACT)?;
            if mmq_feat_major::dispatch(format, client, act_contig, weight, output_ptr, m, k, n)?
                .is_some()
            {
                return Ok(());
            }
        }
    }

    // Both remaining kernels keep the activation in f32 and accumulate in
    // f32, so one check covers the pair.
    weight.check_activation_contract(&tcf_dispatch::F32_CONTRACT)?;
    let launch = if m <= 4 {
        tcf_dispatch::launch_gemv
    } else {
        tcf_dispatch::launch_gemm
    };
    launch(
        client,
        device_index,
        act_contig.ptr(),
        weight.storage().ptr(),
        output_ptr,
        encoding,
        at,
    )
}
