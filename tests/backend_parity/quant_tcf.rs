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

#![cfg(feature = "cuda")]

use super::helpers::{assert_parity_f32_tol, setup_cpu, with_cuda_backend};
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
fn source_values(count: usize, seed: usize) -> Vec<f32> {
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
fn packed(native: NativeEncoding, values: &[f32], shape: &[usize]) -> Vec<u8> {
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
fn cpu_matmul(
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

/// The GEMV path, M <= 16. One warp per output column.
#[test]
fn tcf_cuda_gemv_matches_cpu() {
    with_cuda_backend(|client, device| {
        for (n, k) in SHAPES {
            for m in [1usize, 8] {
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

                    assert_parity_f32_tol(
                        &got,
                        &want,
                        &format!("{} gemv {m}x{k}x{n}", TcfEncoding::new(native).name()),
                        1e-3,
                        1e-5,
                    );
                }
            }
        }
    });
}

/// The GEMV's eight-tile run path, which `SHAPES` is too narrow to reach.
///
/// `TCF_RUN_TILES` is 8, so a row of fewer than eight tiles runs entirely on
/// the tail. These widths reach the run itself, and cover both the case where
/// it divides the row exactly and the case where it leaves a tail behind:
///
/// - `[3, 512]`: 8 tiles per row, one whole run, no tail.
/// - `[2, 704]`: 11 tiles per row, one run and a 3-tile tail. The row width is
///   not a whole number of super-blocks either, so a run's eight tiles straddle
///   super-block boundaries and the two-level forms resolve across them.
/// - `[5, 1024]`: 16 tiles per row, two whole runs.
///
/// `M = 16` is the GEMV/GEMM dispatch boundary in `quant/cuda/quant_matmul`,
/// so it is pinned here rather than assumed.
#[test]
fn tcf_cuda_gemv_run_path_matches_cpu() {
    with_cuda_backend(|client, device| {
        for (n, k) in [(3usize, 512usize), (2, 704), (5, 1024)] {
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

                    assert_parity_f32_tol(
                        &got,
                        &want,
                        &format!("{} gemv run {m}x{k}x{n}", TcfEncoding::new(native).name()),
                        1e-3,
                        1e-5,
                    );
                }
            }
        }
    });
}

/// The GEMM path, M > 16. A 16x16 output tile with the weight staged in shared
/// memory, including an M that is not a multiple of the tile edge.
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

                    assert_parity_f32_tol(
                        &got,
                        &want,
                        &format!("{} gemm {m}x{k}x{n}", TcfEncoding::new(native).name()),
                        1e-3,
                        1e-5,
                    );
                }
            }
        }
    });
}

/// A TCF weight reaching `quant_swiglu` must take the two-matmul path rather
/// than the GGUF fused kernel, which reads a block layout TCF does not have.
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

            assert_parity_f32_tol(
                &got,
                &want,
                &format!("{} swiglu", TcfEncoding::new(native).name()),
                1e-3,
                1e-5,
            );
        }
    });
}
