//! Backend parity for TCF native quantized weights: CUDA against CPU.
//!
//! # What is being gated
//!
//! The CPU kernels delegate to `tcf-core`, which CONFORMANCE.md makes the
//! definition of the semantics. Device code cannot call it, so
//! `quant/cuda/kernels/tcf.cuh` holds a second, read-direction-only copy of
//! Section 14's bit positions and Section 13's reconstruction math. These
//! tests are the reason that copy is allowed to exist: every encoding, decoded
//! by the CUDA kernels and by the CPU path, must agree.
//!
//! # Shapes
//!
//! - `[3, 320]`: 15 tiles, five per row. A partial trailing super-block, and a
//!   row width that is not a whole number of super-blocks.
//! - `[5, 256]`: 20 tiles, four per row. Every row is one whole super-block.
//! - `[2, 448]`: 14 tiles, seven per row. An odd row width, partial trailing
//!   super-block again.
//!
//! Every one of those is under eight tiles per row, which is the GEMV's
//! `TCF_RUN_TILES`. They therefore gate only its tile-at-a-time tail.
//! `tcf_cuda_gemv_run_path_matches_cpu` adds the widths that reach the
//! eight-tile run the GEMV actually spends its time in.
//!
//! # Tolerances
//!
//! Dequantization is compared BIT FOR BIT. The device decoder reconstructs a
//! weight with round-to-nearest intrinsics, so neither FMA contraction nor the
//! approximate division of this crate's `--use_fast_math` build can move it off
//! the CPU value. A matmul is compared with a relative tolerance instead: both
//! sides accumulate in f32 and their summation orders differ, exactly as they
//! do for every GGUF format.
//!
//! TWO dispatch conditions move a case off the f32 kernels and onto one that
//! quantizes the activation to Q8_1 (`quant/cuda/quant_matmul/tcf_route.rs`):
//!
//! - `TCF_DP4A_GEMV_MAX_M` and below, `Q4AS32DT64` takes the token-batched
//!   dp4a GEMV. This one is checked FIRST, so it wins wherever both apply,
//!   and it covers `m = 1` — the shape that used to be the one place a
//!   feature-major encoding kept an element-wise bound.
//! - From `TCF_FEAT_MAJOR_MIN_M` up, an encoding with a `FeatMajorFormat`
//!   takes the MMQ kernel rather than `tcf_gemm_f32` / `tcf_gemv_f32`.
//!
//! A case either condition selects takes `helpers::assert_cosine_parity`;
//! `takes_mmq_path` and `takes_tcf_dp4a_gemv_path` are the mirrors. Every
//! other encoding at any `m`, and a feature-major encoding below both
//! constants, stays on the f32 path and keeps the element-wise gate.
//!
//! The branch conditions here must track those two constants, not the
//! separate `m <= 4` split choosing `launch_gemv` against `launch_gemm` for
//! the encodings on the f32 path. All three thresholds differ.

#![cfg(feature = "cuda")]

use super::helpers::{
    assert_cosine_parity, assert_parity_f32_tol, setup_cpu, takes_mmq_path,
    takes_tcf_dp4a_gemv_path, with_cuda_backend,
};
use boostr::quant::{QuantTensor, TcfEncoding};
use boostr::{DequantOps, QuantMatmulOps};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use tcf_core::{NativeEncoding, pack, quantize};

/// Every v1 native quantized encoding, the two two-level forms included.
const ENCODINGS: [NativeEncoding; 7] = [
    NativeEncoding::Q4S32T64,
    NativeEncoding::Q4AS32T64,
    NativeEncoding::Q4AS64T64,
    NativeEncoding::Q6S32T64,
    NativeEncoding::Q8S32T64,
    NativeEncoding::Q6S16DT64,
    NativeEncoding::Q4AS32DT64,
];

/// `[n, k]` weight shapes covering a partial trailing super-block, a whole one,
/// and a row width that is not a multiple of the super-block.
const SHAPES: [(usize, usize); 3] = [(3, 320), (5, 256), (2, 448)];

/// A deterministic input with sign changes, a flat run, and a spike, so a
/// group's scale and minimum both move between groups.
pub fn source_values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let x = (i + seed) as f32;
            match (i + seed) % 6 {
                0 => 0.75,
                2 => -(x * 0.011).sin() * 2.5,
                4 => (x * 0.037).cos() * 1.5,
                _ => (x * 0.023).sin() * 1.1 - 0.2,
            }
        })
        .collect()
}

/// Pack a tensor with `tcf-core`'s own writer, so the bytes under test are the
/// bytes the format defines.
pub fn packed(native: NativeEncoding, values: &[f32], shape: &[usize]) -> Vec<u8> {
    let dims: Vec<u64> = shape.iter().map(|d| *d as u64).collect();
    let tiles = quantize(values, &dims, 2, native.layout()).expect("quantizes");
    pack(&tiles, native.layout()).expect("packs")
}

/// The CPU path's dequantization, which runs `tcf_core::unpack` followed by
/// `tcf_core::dequantize`.
fn cpu_dequant(payload: &[u8], native: NativeEncoding, shape: &[usize]) -> Vec<f32> {
    let device = CpuDevice::new();
    let (client, _) = setup_cpu();
    let qt =
        QuantTensor::<CpuRuntime>::from_bytes(payload, TcfEncoding::new(native), shape, &device)
            .expect("CPU TCF QuantTensor");
    client
        .dequantize(&qt, DType::F32)
        .expect("CPU dequantize")
        .to_vec::<f32>()
}

/// The CPU path's fused matmul against a packed TCF weight.
pub fn cpu_matmul(
    act: &[f32],
    payload: &[u8],
    native: NativeEncoding,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let device = CpuDevice::new();
    let (client, _) = setup_cpu();
    let activation = Tensor::<CpuRuntime>::from_slice(act, &[m, k], &device).expect("activation");
    let weight =
        QuantTensor::<CpuRuntime>::from_bytes(payload, TcfEncoding::new(native), &[n, k], &device)
            .expect("CPU TCF QuantTensor");
    client
        .quant_matmul(&activation, &weight)
        .expect("CPU quant_matmul")
        .to_vec::<f32>()
}

/// Compare two f32 slices bit for bit, naming the first disagreement.
fn assert_bit_identical(got: &[f32], want: &[f32], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (index, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label} at {index}: CUDA {a} (0x{:08x}) vs CPU {b} (0x{:08x})",
            a.to_bits(),
            b.to_bits(),
        );
    }
}

/// THE dequantization gate. Every encoding, every shape, bit for bit.
#[test]
fn tcf_cuda_dequant_matches_cpu_bit_for_bit() {
    with_cuda_backend(|client, device| {
        for (n, k) in SHAPES {
            let shape = [n, k];
            let values = source_values(n * k, 0);
            for native in ENCODINGS {
                let payload = packed(native, &values, &shape);
                let want = cpu_dequant(&payload, native, &shape);

                let qt =
                    QuantTensor::from_bytes(&payload, TcfEncoding::new(native), &shape, &device)
                        .expect("CUDA TCF QuantTensor");
                let got = client
                    .dequantize(&qt, DType::F32)
                    .expect("CUDA dequantize")
                    .to_vec::<f32>();

                assert_bit_identical(
                    &got,
                    &want,
                    &format!("{} dequant {n}x{k}", TcfEncoding::new(native).name()),
                );
            }
        }
    });
}

/// `M = 1` runs the GEMV path, one warp per output column — or, for an
/// encoding `takes_tcf_dp4a_gemv_path` selects, the single-token tile of the
/// dp4a GEMV. `M = 8` is past the `m <= 4` GEMV/GEMM boundary in
/// `quant/cuda/quant_matmul/tcf_route.rs`, so it runs the GEMM path instead —
/// the register-blocked f32 tile, except for an encoding `takes_mmq_path`
/// selects, which instead takes the MMQ kernel under test in
/// `quant_tcf_feat_major.rs`. Both of those quantize the activation to Q8_1.
///
/// `M = 2` and `M = 3` exist for the dp4a GEMV's token tiles: 2 reaches the
/// `_n2` kernel and 3 the `_n4` one with its last slot clamped, so a wired
/// tile width cannot ship without a parity case that reaches it. For every
/// other encoding both are ordinary GEMV cases.
#[test]
fn tcf_cuda_gemv_matches_cpu() {
    with_cuda_backend(|client, device| {
        for (n, k) in SHAPES {
            for m in [1usize, 2, 3, 8] {
                let weight_values = source_values(n * k, 0);
                let act = source_values(m * k, 17);
                for native in ENCODINGS {
                    let payload = packed(native, &weight_values, &[n, k]);
                    let want = cpu_matmul(&act, &payload, native, m, k, n);

                    let activation =
                        Tensor::from_slice(&act, &[m, k], &device).expect("activation");
                    let weight = QuantTensor::from_bytes(
                        &payload,
                        TcfEncoding::new(native),
                        &[n, k],
                        &device,
                    )
                    .expect("CUDA TCF QuantTensor");
                    let got = client
                        .quant_matmul(&activation, &weight)
                        .expect("CUDA quant_matmul")
                        .to_vec::<f32>();

                    let label = format!("{} gemv {m}x{k}x{n}", TcfEncoding::new(native).name());
                    if takes_mmq_path(native, m) || takes_tcf_dp4a_gemv_path(native, m, k) {
                        assert_cosine_parity(&got, &want, &label);
                    } else {
                        assert_parity_f32_tol(&got, &want, &label, 1e-3, 1e-5);
                    }
                }
            }
        }
    });
}

/// The GEMV's run paths, which `SHAPES` is too narrow to reach.
///
/// The GEMV walks a row at three granularities: whole scale-resolve runs of
/// `TCF_GEMV_RUN_TILES` tiles, then whole eight-tile code reads
/// (`TCF_RUN_TILES`), then single tiles. A row of fewer than eight tiles runs
/// entirely on the last. These widths reach each of the three, and cover both
/// the case where a granularity divides the row exactly and the case where it
/// leaves a tail behind:
///
/// - `[3, 512]`: 8 tiles per row, one whole code read, no tail.
/// - `[2, 704]`: 11 tiles per row, one code read and a 3-tile tail. The row
///   width is not a whole number of super-blocks either, so a read's eight
///   tiles straddle super-block boundaries and the two-level forms resolve
///   across them.
/// - `[5, 1024]`: 16 tiles per row, two whole code reads.
/// - `[3, 2048]`: 32 tiles per row, one whole scale-resolve run at the
///   committed width, no tail of either kind.
/// - `[2, 2752]`: 43 tiles per row — one scale-resolve run, then one whole
///   code read, then a 3-tile tail, so all three granularities run in one row
///   and their partial sums must still agree with the CPU.
///
/// Widening `TCF_GEMV_RUN_TILES` past 32 leaves the last two too narrow to
/// reach the widened run; extend them with it.
///
/// The GEMV/GEMM dispatch boundary in `quant/cuda/quant_matmul/tcf_route.rs`
/// is `m > 4`, not 16, so `M = 16` here already runs the GEMM path — the
/// eight-tile run is exercised through `M = 1` at these wider `k` values, and
/// `M = 16` additionally checks GEMM (or, for an encoding `takes_mmq_path`
/// selects, the Q8_1-activation MMQ kernel) parity at the same shapes.
///
/// These `k` values carry the dp4a GEMV's own geometry for the encoding
/// `takes_tcf_dp4a_gemv_path` selects, which resolves a run of 32-element
/// groups and then walks it eight groups per step: 512 is 16 groups, a partial
/// run of two whole steps; 704 is 22 groups, two steps and a 6-group tail that
/// runs the bounds check; 2752 is 86 groups, one whole run followed by a
/// partial one that ends on that same 6-group tail. Neither 704 nor a row of
/// 512 at `n = 3` starts on a super-block boundary, so its global-tile scale
/// resolution is exercised across super-blocks.
#[test]
fn tcf_cuda_gemv_run_path_matches_cpu() {
    with_cuda_backend(|client, device| {
        for (n, k) in [
            (3usize, 512usize),
            (2, 704),
            (5, 1024),
            (3, 2048),
            (2, 2752),
        ] {
            for m in [1usize, 16] {
                let weight_values = source_values(n * k, 0);
                let act = source_values(m * k, 41);
                for native in ENCODINGS {
                    let payload = packed(native, &weight_values, &[n, k]);
                    let want = cpu_matmul(&act, &payload, native, m, k, n);

                    let activation =
                        Tensor::from_slice(&act, &[m, k], &device).expect("activation");
                    let weight = QuantTensor::from_bytes(
                        &payload,
                        TcfEncoding::new(native),
                        &[n, k],
                        &device,
                    )
                    .expect("CUDA TCF QuantTensor");
                    let got = client
                        .quant_matmul(&activation, &weight)
                        .expect("CUDA quant_matmul")
                        .to_vec::<f32>();

                    let label = format!("{} gemv run {m}x{k}x{n}", TcfEncoding::new(native).name());
                    if takes_mmq_path(native, m) || takes_tcf_dp4a_gemv_path(native, m, k) {
                        assert_cosine_parity(&got, &want, &label);
                    } else {
                        assert_parity_f32_tol(&got, &want, &label, 1e-3, 1e-5);
                    }
                }
            }
        }
    });
}

/// The GEMM path (`m > 4` in `quant/cuda/quant_matmul/tcf_route.rs`). A 16x16
/// output tile with the weight staged in shared memory, including an M that
/// is not a multiple of the tile edge. An encoding `takes_mmq_path` selects
/// instead takes the MMQ kernel under test in `quant_tcf_feat_major.rs` and
/// quantizes its activation to Q8_1.
#[test]
fn tcf_cuda_gemm_matches_cpu() {
    with_cuda_backend(|client, device| {
        for (n, k) in SHAPES {
            for m in [65usize, 128] {
                let weight_values = source_values(n * k, 0);
                let act = source_values(m * k, 29);
                for native in ENCODINGS {
                    let payload = packed(native, &weight_values, &[n, k]);
                    let want = cpu_matmul(&act, &payload, native, m, k, n);

                    let activation =
                        Tensor::from_slice(&act, &[m, k], &device).expect("activation");
                    let weight = QuantTensor::from_bytes(
                        &payload,
                        TcfEncoding::new(native),
                        &[n, k],
                        &device,
                    )
                    .expect("CUDA TCF QuantTensor");
                    let got = client
                        .quant_matmul(&activation, &weight)
                        .expect("CUDA quant_matmul")
                        .to_vec::<f32>();

                    let label = format!("{} gemm {m}x{k}x{n}", TcfEncoding::new(native).name());
                    if takes_mmq_path(native, m) || takes_tcf_dp4a_gemv_path(native, m, k) {
                        assert_cosine_parity(&got, &want, &label);
                    } else {
                        assert_parity_f32_tol(&got, &want, &label, 1e-3, 1e-5);
                    }
                }
            }
        }
    });
}

/// A TCF weight reaching `quant_swiglu` must take the two-matmul path rather
/// than the GGUF fused kernel, which reads a block layout TCF does not have.
///
/// `m` is 4, at or past the `TCF_FEAT_MAJOR_MIN_M` crossover in
/// `quant/cuda/quant_matmul/tcf_route.rs`, so an encoding `takes_mmq_path`
/// selects routes through the MMQ kernel for both of its matmuls here and
/// quantizes its activation to Q8_1 each time; the other encodings stay on
/// the f32 tile. SwiGLU then applies a sigmoid and a product on top, so that
/// encoding's output carries the Q8_1 error through a nonlinearity rather
/// than a linear combination, and only `assert_cosine_parity` bounds it; the
/// other encodings keep the element-wise gate.
///
/// The nonlinearity costs roughly one digit of cosine headroom against the
/// plain matmul cases, which is measured and expected. The score still clears
/// the floor by a wide margin, so the gate keeps its power to catch a decode
/// defect: a wrong plane offset or scale index scrambles direction and
/// collapses the score toward zero regardless of the sigmoid.
#[test]
fn tcf_cuda_swiglu_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (n, k, m) = (5usize, 256usize, 4usize);
        let gate_values = source_values(n * k, 0);
        let up_values = source_values(n * k, 41);
        let act = source_values(m * k, 7);

        for native in ENCODINGS {
            let gate_payload = packed(native, &gate_values, &[n, k]);
            let up_payload = packed(native, &up_values, &[n, k]);

            let cpu_device = CpuDevice::new();
            let (cpu_client, _) = setup_cpu();
            let cpu_act =
                Tensor::<CpuRuntime>::from_slice(&act, &[m, k], &cpu_device).expect("activation");
            let cpu_gate = QuantTensor::<CpuRuntime>::from_bytes(
                &gate_payload,
                TcfEncoding::new(native),
                &[n, k],
                &cpu_device,
            )
            .expect("CPU gate");
            let cpu_up = QuantTensor::<CpuRuntime>::from_bytes(
                &up_payload,
                TcfEncoding::new(native),
                &[n, k],
                &cpu_device,
            )
            .expect("CPU up");
            let want = cpu_client
                .quant_swiglu(&cpu_act, &cpu_gate, &cpu_up)
                .expect("CPU quant_swiglu")
                .to_vec::<f32>();

            let cuda_act = Tensor::from_slice(&act, &[m, k], &device).expect("activation");
            let cuda_gate =
                QuantTensor::from_bytes(&gate_payload, TcfEncoding::new(native), &[n, k], &device)
                    .expect("CUDA gate");
            let cuda_up =
                QuantTensor::from_bytes(&up_payload, TcfEncoding::new(native), &[n, k], &device)
                    .expect("CUDA up");
            let got = client
                .quant_swiglu(&cuda_act, &cuda_gate, &cuda_up)
                .expect("CUDA quant_swiglu")
                .to_vec::<f32>();

            let label = format!("{} swiglu", TcfEncoding::new(native).name());
            if takes_mmq_path(native, m) || takes_tcf_dp4a_gemv_path(native, m, k) {
                assert_cosine_parity(&got, &want, &label);
            } else {
                assert_parity_f32_tol(&got, &want, &label, 1e-3, 1e-5);
            }
        }
    });
}
