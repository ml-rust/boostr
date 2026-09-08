//! The feature-major tensor-core GEMMs for the TCF encodings, against CPU.
//!
//! # What is being gated
//!
//! `Q8S32T64` and `Q4AS32DT64` are the TCF encodings with a `FeatMajorFormat`,
//! so above the GEMV crossover their weights go through
//! `quant_mmq_tcf_q8s32t64_q8_1_mma_*` and
//! `quant_mmq_tcf_q4as32dt64_q8_1_mma_*` rather than through `tcf_gemm_f32`.
//! Those kernels read the payload's planes directly — for `Q8S32T64` a dense
//! row-major int8 code plane and a dense row-major `binary16` scale plane, for
//! `Q4AS32DT64` a 4-bit code plane then four super-block planes — instead of
//! the block geometry every GGUF format has, and they stage them into the Q8_0
//! and Q4_K weight rows. A wrong plane offset or a wrong scale index there
//! produces plausible weights, not an error, so the only thing that catches it
//! is the CPU result.
//!
//! Ground truth is `tcf-core` through the CPU path, which CONFORMANCE.md makes
//! the definition of the semantics. Never the sibling TCF GPU kernels: they
//! would agree with a wrong reading of the same planes.
//!
//! # Shapes
//!
//! `Q8S32T64` takes every `k` that is a multiple of 64, which the encoding
//! requires and the dispatch guard re-checks.
//!
//! - `[128, 256]`: one whole feature tile, and `k / 32 = 8` blocks, exactly one
//!   256-k staging group with no tail.
//! - `[200, 320]`: an incomplete second feature tile, and 10 blocks, so the
//!   last staging group is a 2-block tail that runs the `CLAMP_K` path.
//! - `[257, 512]`: a third feature tile holding one row, and two whole staging
//!   groups.
//!
//! `m` covers several token tiles: 2 and 3 sit at and just above the
//! `TCF_FEAT_MAJOR_MIN_M` dispatch crossover, where the token tile (its
//! smallest compiled variant is 8) is mostly empty and the clamp is exercised
//! hardest; 8 and 40 pick the 8-token granularity, 100 and 128 the 16-token
//! one, and 100 is not a multiple of its tile.
//!
//! `Q4AS32DT64` takes a different set: its dispatch guard is `k % 256 == 0`,
//! because a super-block is indexed by the global flattened tile number and a
//! row therefore starts on a super-block boundary only at that multiple.
//!
//! - `[128, 256]`: one whole feature tile, and exactly ONE super-block per row,
//!   which is one staging group with no tail.
//! - `[200, 1024]`: an incomplete second feature tile, and four super-blocks
//!   per row, so the per-row plane stride is walked rather than assumed.
//! - `[257, 512]`: a third feature tile holding one row, and two super-blocks
//!   per row.
//!
//! Every shape has `n > 1`, which is what makes a wrong row stride in any of
//! the five planes visible: with a single row every stride agrees.
//!
//! # Tolerance
//!
//! Every case here has an `m` its encoding's dispatch actually sends to MMQ —
//! at or above `TCF_FEAT_MAJOR_MIN_M` for `Q8S32T64`, and above
//! `TCF_DP4A_GEMV_MAX_M` for `Q4AS32DT64`, whose token-batched dp4a GEMV is
//! checked first and would otherwise take the smallest batches — and a native
//! encoding with a feature-major kernel, so
//! every case takes an MMQ kernel under test and quantizes its activation to Q8_1
//! while the CPU reference keeps f32. An element-wise tolerance cannot bound
//! that gap (see
//! `helpers::assert_cosine_parity`), so this file uses the cosine gate for
//! every case rather than mixing gates.

#![cfg(feature = "cuda")]

use super::helpers::{assert_cosine_parity, with_cuda_backend};
use super::quant_tcf::{cpu_matmul, packed, source_values};
use boostr::QuantMatmulOps;
use boostr::quant::{QuantTensor, TcfEncoding};
use numr::tensor::Tensor;
use tcf_core::NativeEncoding;

/// `[n, k]` weight shapes: a whole feature tile, a partial one, and a feature
/// tile holding a single row.
const SHAPES: [(usize, usize); 3] = [(128, 256), (200, 320), (257, 512)];

/// `[n, k]` weight shapes whose `k` is a whole number of 256-element
/// super-blocks, which `Q4AS32DT64` requires: one super-block per row, four per
/// row, and two per row.
const SUPER_BLOCK_SHAPES: [(usize, usize); 3] = [(128, 256), (200, 1024), (257, 512)];

/// Batch sizes at or above `TCF_FEAT_MAJOR_MIN_M`, the GEMV/MMQ crossover
/// (not the plain `m <= 4` GEMV/GEMM split), spanning both warp granularities
/// and one that is not a whole token tile. `2` and `3` sit at and just above
/// the crossover, leaving the token tile mostly empty and so exercising the
/// tile clamp hardest.
const BATCHES: [usize; 6] = [2, 3, 8, 40, 100, 128];

/// `Q4AS32DT64` reaches MMQ at every entry above while
/// `TCF_DP4A_GEMV_MAX_M` is zero. Should that constant move, the token counts
/// at or below it would route to the dp4a GEMV instead, leaving the MMQ kernel
/// this file exists to gate untested there — give this encoding its own list
/// starting at the smallest `m` that still reaches MMQ, preferring an odd one
/// so the token tile runs with most slots clamped.

#[test]
fn tcf_q8s32t64_feat_major_gemm_matches_cpu() {
    let native = NativeEncoding::Q8S32T64;
    with_cuda_backend(|client, device| {
        for (n, k) in SHAPES {
            let weight_values = source_values(n * k, 0);
            let payload = packed(native, &weight_values, &[n, k]);
            let weight =
                QuantTensor::from_bytes(&payload, TcfEncoding::new(native), &[n, k], &device)
                    .expect("CUDA TCF QuantTensor");

            for m in BATCHES {
                let act = source_values(m * k, 53);
                let want = cpu_matmul(&act, &payload, native, m, k, n);

                let activation = Tensor::from_slice(&act, &[m, k], &device).expect("activation");
                let got = client
                    .quant_matmul(&activation, &weight)
                    .expect("CUDA quant_matmul")
                    .to_vec::<f32>();

                assert_cosine_parity(
                    &got,
                    &want,
                    &format!("Q8S32T64 feat-major gemm {m}x{k}x{n}"),
                );
            }
        }
    });
}

#[test]
fn tcf_q4as32dt64_feat_major_gemm_matches_cpu() {
    let native = NativeEncoding::Q4AS32DT64;
    with_cuda_backend(|client, device| {
        for (n, k) in SUPER_BLOCK_SHAPES {
            let weight_values = source_values(n * k, 0);
            let payload = packed(native, &weight_values, &[n, k]);
            let weight =
                QuantTensor::from_bytes(&payload, TcfEncoding::new(native), &[n, k], &device)
                    .expect("CUDA TCF QuantTensor");

            for m in BATCHES {
                let act = source_values(m * k, 53);
                let want = cpu_matmul(&act, &payload, native, m, k, n);

                let activation = Tensor::from_slice(&act, &[m, k], &device).expect("activation");
                let got = client
                    .quant_matmul(&activation, &weight)
                    .expect("CUDA quant_matmul")
                    .to_vec::<f32>();

                assert_cosine_parity(
                    &got,
                    &want,
                    &format!("Q4AS32DT64 feat-major gemm {m}x{k}x{n}"),
                );
            }
        }
    });
}
