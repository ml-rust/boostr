//! Kernel choice for a TCF-encoded weight inside `quant_matmul`.
//!
//! TCF weights take their own kernels: a GGUF kernel finds a block's codes and
//! its scale adjacent, while TCF spreads them over whole-tensor planes. The
//! choice between those kernels is its own decision with its own crossovers,
//! so it lives here rather than inline in the `QuantMatmulOps` impl.
//!
//! # Resolution, not veto
//!
//! Section 9 makes a kernel resolve on `(weight_encoding, contract_digest,
//! execution_role)`, so the declared activation contract is an INPUT to the
//! choice. [`resolve`] walks the shape's candidate kernels in preference order
//! and returns the first one the declared contract accepts, so a weight
//! declaring exact f32 activations never selects an activation-quantizing
//! kernel in the first place — it lands on the f32 pair and runs.
//!
//! The refusal stays where the format puts it: when NO candidate satisfies the
//! declared contract, resolution returns the shape's preferred kernel, the
//! check in [`route_tcf`] rejects it, and the operation stops with
//! [`crate::error::Error::ActivationContractMismatch`]. Section 9 defines no
//! float fallback, so "no satisfying kernel" is an error and never a reroute
//! onto a kernel that computes something else.

use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::error::Result;
use crate::quant::{ActivationContract, KernelContract, QuantTensor, TcfEncoding};

use super::super::tcf::{self as tcf_dispatch, MatmulShape};
use super::mmq_feat_major::{self, FeatMajorFormat};

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
/// `None` is "no token count takes it". **Measured off.** The dp4a GEMV is
/// compiled and parity-checked but wins no token count, so it routes nowhere:
/// `m = 1` keeps the f32 GEMV and every larger `m` keeps the feature-major
/// MMQ, which is where they were before the kernel existed.
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
/// Contract resolution does not need it turned on: a weight declaring exact
/// f32 activations resolves to `tcf_gemv_f32` at `m = 1` on its own, and a
/// weight declaring the 8-bit dynamic contract picks this kernel up as soon as
/// the constant is `Some`.
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

/// One TCF matmul kernel a shape can be served by, together with what it
/// needs to launch.
///
/// [`TcfKernel::F32`] covers `tcf_gemv_f32` and `tcf_gemm_f32` as one
/// candidate: they differ in blocking, not in arithmetic, so they satisfy the
/// same contract and the choice between them is made at launch by `m`.
#[derive(Clone, Copy)]
enum TcfKernel {
    /// The token-batched dp4a GEMV. Quantizes the activation.
    Dp4aGemv,
    /// The feature-major tensor-core MMQ family, with the descriptor the
    /// encoding and K resolved to. Quantizes the activation.
    FeatMajorMmq(&'static FeatMajorFormat),
    /// The f32 GEMV/GEMM pair. Reads the activation as handed in.
    F32,
}

impl TcfKernel {
    /// What this kernel computes, as the kernel itself declares it.
    ///
    /// Every value comes from a constant sitting beside its kernel, so this
    /// restates nothing: a kernel whose arithmetic changes changes its own
    /// constant and this follows.
    fn contract(self) -> KernelContract {
        match self {
            Self::Dp4aGemv => tcf_dispatch::DP4A_GEMV_CONTRACT,
            Self::FeatMajorMmq(_) => mmq_feat_major::CONTRACT,
            Self::F32 => tcf_dispatch::F32_CONTRACT,
        }
    }

    /// Whether this kernel computes what `declared` requires.
    ///
    /// A weight with no contract accepts every candidate, which is what keeps
    /// a GGUF-sourced weight on the shape-only order below.
    fn accepted_by(self, declared: Option<&ActivationContract>) -> bool {
        declared.is_none_or(|contract| self.contract().satisfies(contract))
    }
}

/// The kernel `at` and `declared` resolve to.
///
/// Candidates are visited in the shape's preference order, and the first one
/// the declared contract accepts wins:
/// 1. [`TCF_DP4A_GEMV_MAX_M`] offers the token-batched dp4a GEMV at the
///    smallest token counts, for the encodings and K `supports_dp4a_gemv`
///    serves.
/// 2. [`TCF_FEAT_MAJOR_MIN_M`] offers the feature-major MMQ family, for an
///    encoding with a `FeatMajorFormat` on a K and a device it covers.
/// 3. The f32 pair, which serves every shape.
///
/// With no declared contract every candidate is accepted, so the first one the
/// shape offers wins and the order above IS the routing — unchanged from
/// before contracts existed.
///
/// When a contract is declared and no candidate satisfies it, this returns the
/// shape's preferred candidate so the caller's check names the kernel the
/// shape would have run and refuses. It never returns a candidate as a
/// fallback for a contract that candidate does not satisfy.
fn resolve(
    declared: Option<&ActivationContract>,
    encoding: TcfEncoding,
    at: MatmulShape,
    device_index: usize,
) -> TcfKernel {
    let MatmulShape { m, k, .. } = at;
    let mut preferred: Option<TcfKernel> = None;

    // The dp4a GEMV is offered FIRST, ahead of MMQ: at a small token count it
    // decodes each weight group once per block into int8 and never stages a
    // tile, which is the cheaper shape of work there. `supports_dp4a_gemv`
    // covers the encodings and K it serves.
    if TCF_DP4A_GEMV_MAX_M.is_some_and(|max| m <= max)
        && tcf_dispatch::supports_dp4a_gemv(encoding, k)
    {
        if TcfKernel::Dp4aGemv.accepted_by(declared) {
            return TcfKernel::Dp4aGemv;
        }
        preferred.get_or_insert(TcfKernel::Dp4aGemv);
    }

    if m >= TCF_FEAT_MAJOR_MIN_M {
        // An encoding with a `FeatMajorFormat` takes the feature-major
        // tensor-core family instead of the f32 FMA tile `launch_gemm` runs;
        // `mmq_feat_major::feat_major_format` is the one place that says which
        // those are. Every other encoding keeps that tile, as does any K the
        // format's staging map does not cover, and any device without the
        // integer MMA shape the family is built on.
        let feat_major = mmq_feat_major::feat_major_format(encoding.native())
            .filter(|format| k.is_multiple_of(format.k_multiple as usize))
            .filter(|_| {
                numr::runtime::cuda::CudaDevice::new(device_index)
                    .profile()
                    .caps
                    .int8_mma_m16n8k32
            });
        if let Some(format) = feat_major {
            if TcfKernel::FeatMajorMmq(format).accepted_by(declared) {
                return TcfKernel::FeatMajorMmq(format);
            }
            preferred.get_or_insert(TcfKernel::FeatMajorMmq(format));
        }
    }

    if TcfKernel::F32.accepted_by(declared) {
        return TcfKernel::F32;
    }
    // Nothing the shape offers satisfies the contract. The caller turns this
    // into the refusal; the f32 pair is the last resort only for naming which
    // kernel the shape would have run.
    preferred.unwrap_or(TcfKernel::F32)
}

/// Run `act_contig [m, k] x weight [n, k]^T -> output_ptr [m, n]` on whichever
/// TCF kernel the shape and the weight's declared contract resolve to.
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
/// [`resolve`] applies all three against the declared contract, so a crossover
/// only picks among the kernels that contract accepts.
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
/// # Errors
/// [`crate::error::Error::ActivationContractMismatch`] when no kernel the
/// shape offers satisfies the weight's declared contract, and whatever the
/// resolved launch raises otherwise.
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

    let declared = weight.activation_contract();
    let resolved = resolve(declared, encoding, at, device_index);
    // Passes by construction whenever a candidate satisfied the contract, and
    // raises the refusal when none did. Section 9 defines no float fallback,
    // so this is where the operation stops.
    weight.check_activation_contract(&resolved.contract())?;

    match resolved {
        TcfKernel::Dp4aGemv => {
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
        TcfKernel::FeatMajorMmq(format) => {
            if mmq_feat_major::dispatch(format, client, act_contig, weight, output_ptr, m, k, n)?
                .is_some()
            {
                return Ok(());
            }
            // `Ok(None)` is "no compiled variant fits this launch", which
            // resolution cannot see ahead of the dispatch. The f32 pair below
            // is a different kernel with different arithmetic, so it is
            // resolved afresh rather than inherited.
            weight.check_activation_contract(&TcfKernel::F32.contract())?;
        }
        TcfKernel::F32 => {}
    }

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
