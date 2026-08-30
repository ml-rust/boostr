//! Backend parity for TCF native quantized weights: WebGPU against CPU.
//!
//! # What is being gated
//!
//! The CPU kernels delegate to `tcf-core`, which CONFORMANCE.md makes the
//! definition of the semantics. A WGSL shader cannot call it, so
//! `quant/wgpu/shaders/tcf.rs` holds a second, read-direction-only copy of
//! Section 14's bit positions and Section 13's reconstruction math. These
//! tests are the reason that copy is allowed to exist: every encoding, decoded
//! by the WebGPU shaders and by the CPU path, must agree.
//!
//! # Shapes
//!
//! - `[3, 320]`: 15 tiles, five per row. A partial trailing super-block, and a
//!   row width that is not a whole number of super-blocks.
//! - `[6, 256]`: 24 tiles, four per row. Every row is one whole super-block.
//! - `[2, 448]`: 14 tiles, seven per row. An odd row width, partial trailing
//!   super-block again.
//!
//! Those three all have an `n` below the matmul kernels' 16-wide output tile,
//! which is enough for the decode but leaves the workgroup-tiled kernel's
//! interior untested. `tcf_wgpu_tiled_matmul_matches_cpu` adds two shapes that
//! exceed the tile in both `M` and `N`, one an exact multiple of it and one
//! not, so the boundary masking is covered rather than assumed.
//!
//! They also all have a row of 4, 5 or 7 execution tiles, which is BELOW the
//! small-M GEMV kernel's eight-tile run, so they only ever reach that kernel's
//! tail. `tcf_wgpu_gemv_matmul_matches_cpu` adds three shapes whose rows
//! exceed the run width, one an exact multiple of it and two not, so the run
//! path and the transition into the tail are both covered. A CUDA fast path
//! shipped once whose every test shape was under its new run width; the
//! kernel was never executed by its own gate.
//!
//! # Tolerances
//!
//! Dequantization is compared BIT FOR BIT, the same gate the CUDA path meets.
//! The shader decodes every binary16 by integer bit manipulation, and it also
//! divides a two-level scale or minimum in integers: WGSL specifies f32 `/` at
//! 2.5 ULP rather than correctly rounded, and an adapter this suite was run on
//! returned the neighbouring float for `Q6S16D_T64`. So no operation the shader
//! reaches for is looser than the CPU's. A matmul is compared with a relative
//! tolerance instead: both sides accumulate in f32 and their summation orders
//! differ, exactly as they do for every GGUF format.

#![cfg(feature = "wgpu")]

use super::helpers::{assert_parity_f32_tol, setup_cpu, with_wgpu_backend};
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
const SHAPES: [(usize, usize); 3] = [(3, 320), (6, 256), (2, 448)];

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
            "{label} at {index}: WebGPU {a} (0x{:08x}) vs CPU {b} (0x{:08x})",
            a.to_bits(),
            b.to_bits(),
        );
    }
}

/// THE dequantization gate. Every encoding, every shape, bit for bit.
#[test]
fn tcf_wgpu_dequant_matches_cpu_bit_for_bit() {
    with_wgpu_backend(|client, device| {
        for (n, k) in SHAPES {
            let shape = [n, k];
            let values = source_values(n * k, 0);
            for native in ENCODINGS {
                let payload = packed(native, &values, &shape);
                let want = cpu_dequant(&payload, native, &shape);

                let qt =
                    QuantTensor::from_bytes(&payload, TcfEncoding::new(native), &shape, &device)
                        .expect("WebGPU TCF QuantTensor");
                let got = client
                    .dequantize(&qt, DType::F32)
                    .expect("WebGPU dequantize")
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

/// The fused matmul, over an M below and above the 16-row workgroup edge.
#[test]
fn tcf_wgpu_matmul_matches_cpu() {
    with_wgpu_backend(|client, device| {
        for (n, k) in SHAPES {
            for m in [1usize, 8, 65] {
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
                    .expect("WebGPU TCF QuantTensor");
                    let got = client
                        .quant_matmul(&activation, &weight)
                        .expect("WebGPU quant_matmul")
                        .to_vec::<f32>();

                    assert_parity_f32_tol(
                        &got,
                        &want,
                        &format!("{} matmul {m}x{k}x{n}", TcfEncoding::new(native).name()),
                        1e-3,
                        1e-5,
                    );
                }
            }
        }
    });
}

/// The workgroup-tiled matmul, which `dispatch_matmul` selects above
/// `MATMUL_GEMV_MAX_M` activation rows.
///
/// # Why this test exists separately
///
/// `tcf_wgpu_matmul_matches_cpu` runs `n` of 2, 3 and 6, all below the kernel's
/// 16-wide output tile, so every workgroup there is a boundary workgroup in `N`
/// and no invocation with `lid.x` past the weight count ever carries a real
/// value. That would have gated the kernel without ever exercising its interior
/// — the exact hole a CUDA fast path shipped through, whose tests all used
/// shapes below its blocking factor.
///
/// So both shapes here exceed the 16x16 output tile in BOTH `M` and `N`:
///
/// - `m=32, n=32`: an exact multiple of the tile in both, four full
///   workgroups, no masking anywhere.
/// - `m=37, n=19`: a multiple of neither, so the last workgroup row overhangs
///   `M` by 11 and the last column overhangs `N` by 13. Those invocations must
///   still reach every barrier and stage `0.0` rather than return.
#[test]
fn tcf_wgpu_tiled_matmul_matches_cpu() {
    // `(m, k, n)`. Both `m` are at or above the tiled kernel's threshold, so
    // both take the tiled path.
    const TILED_SHAPES: [(usize, usize, usize); 2] = [(32, 256, 32), (37, 192, 19)];

    with_wgpu_backend(|client, device| {
        for (m, k, n) in TILED_SHAPES {
            let weight_values = source_values(n * k, 3);
            let act = source_values(m * k, 23);
            for native in ENCODINGS {
                let payload = packed(native, &weight_values, &[n, k]);
                let want = cpu_matmul(&act, &payload, native, m, k, n);

                let activation = Tensor::from_slice(&act, &[m, k], &device).expect("activation");
                let weight =
                    QuantTensor::from_bytes(&payload, TcfEncoding::new(native), &[n, k], &device)
                        .expect("WebGPU TCF QuantTensor");
                let got = client
                    .quant_matmul(&activation, &weight)
                    .expect("WebGPU quant_matmul")
                    .to_vec::<f32>();

                assert_parity_f32_tol(
                    &got,
                    &want,
                    &format!(
                        "{} tiled matmul {m}x{k}x{n}",
                        TcfEncoding::new(native).name()
                    ),
                    1e-3,
                    1e-5,
                );
            }
        }
    });
}

/// The small-M GEMV, which `dispatch_matmul` selects at `MATMUL_GEMV_MAX_M`
/// activation rows and below.
///
/// # Why this test exists separately
///
/// The kernel walks a weight row EIGHT execution tiles at a time and finishes
/// whatever a whole run cannot cover on a tile-at-a-time tail. Every `k` in
/// `SHAPES` is 4, 5 or 7 tiles per row, so those tests enter the tail
/// immediately and the run path — where the group parameters are resolved
/// once per 512 weights rather than once per 64 — never executes. A CUDA fast
/// path shipped through exactly that hole.
///
/// So every `k` here exceeds the run width in tiles per row, and the three
/// cover both sides of the tail:
///
/// - `m=1, k=1024, n=32`: 16 tiles per row, an exact multiple of the run, so
///   two whole runs and NO tail. `n` is a multiple of the workgroup's eight
///   columns, so no column is masked either. This is the `M = 1` case the
///   kernel exists for.
/// - `m=4, k=704, n=19`: 11 tiles per row, one whole run plus a three-tile
///   tail. `n` is not a multiple of eight, so the last workgroup overhangs `N`
///   by five and those invocations must still reach every barrier.
/// - `m=8, k=576, n=8`: 9 tiles per row, one run plus a one-tile tail, at the
///   band's ceiling, where eight accumulators per invocation are live.
#[test]
fn tcf_wgpu_gemv_matmul_matches_cpu() {
    // `(m, k, n)`. Every `m` is at or below the GEMV band's ceiling, so every
    // shape takes the GEMV path.
    const GEMV_SHAPES: [(usize, usize, usize); 3] = [(1, 1024, 32), (4, 704, 19), (8, 576, 8)];

    with_wgpu_backend(|client, device| {
        for (m, k, n) in GEMV_SHAPES {
            let weight_values = source_values(n * k, 5);
            let act = source_values(m * k, 29);
            for native in ENCODINGS {
                let payload = packed(native, &weight_values, &[n, k]);
                let want = cpu_matmul(&act, &payload, native, m, k, n);

                let activation = Tensor::from_slice(&act, &[m, k], &device).expect("activation");
                let weight =
                    QuantTensor::from_bytes(&payload, TcfEncoding::new(native), &[n, k], &device)
                        .expect("WebGPU TCF QuantTensor");
                let got = client
                    .quant_matmul(&activation, &weight)
                    .expect("WebGPU quant_matmul")
                    .to_vec::<f32>();

                assert_parity_f32_tol(
                    &got,
                    &want,
                    &format!(
                        "{} gemv matmul {m}x{k}x{n}",
                        TcfEncoding::new(native).name()
                    ),
                    1e-3,
                    1e-5,
                );
            }
        }
    });
}

/// A TCF weight reaching `quant_swiglu` must take the two-matmul path rather
/// than the GGUF fused kernel, which reads a block layout TCF does not have.
#[test]
fn tcf_wgpu_swiglu_matches_cpu() {
    with_wgpu_backend(|client, device| {
        let (n, k, m) = (6usize, 256usize, 4usize);
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

            let wgpu_act = Tensor::from_slice(&act, &[m, k], &device).expect("activation");
            let wgpu_gate =
                QuantTensor::from_bytes(&gate_payload, TcfEncoding::new(native), &[n, k], &device)
                    .expect("WebGPU gate");
            let wgpu_up =
                QuantTensor::from_bytes(&up_payload, TcfEncoding::new(native), &[n, k], &device)
                    .expect("WebGPU up");
            let got = client
                .quant_swiglu(&wgpu_act, &wgpu_gate, &wgpu_up)
                .expect("WebGPU quant_swiglu")
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
