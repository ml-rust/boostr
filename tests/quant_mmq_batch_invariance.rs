//! A row's CUDA quantized matmul result does not depend on how many rows
//! share the launch or on which token tile it lands in.
//!
//! Every weight with a feature-major MMQ kernel takes it at every M. The
//! split count is fixed by K, N and the device, and every schedule — the
//! split-K pair, the fused tile-parallel grid, at the 16-, 64- and 128-row
//! feature tiles — sums the same K-range partials in the same order. So row
//! `r` of an M-row product must be the same bits as the M=1 product of row
//! `r` alone, at every M the dispatch serves with a different tiling or
//! schedule, through `quant_matmul`, `quant_matmul_batch` and `quant_swiglu`.
//!
//! At M=1, Q8_0, PQ2_0, Q2_0, Q1_0 and PTQ1_0 take the single-token kernel
//! `quant_mmq_<fmt>_q8_1_gemv1` (`mmq_feat_major::gemv1`), a third schedule
//! that forms the same bits as the tensor-core kernels, so the "alone"
//! reference every batched row is held to below IS that kernel's output for
//! those five formats, and every batched M runs the tensor-core kernels
//! against it. The comparison is on the bit pattern (`to_bits`), never a
//! tolerance.
//!
//! The pair-vs-grid pick is measured per device at first use, so the run
//! below covers whichever schedule this device picks; the second run pins
//! every format to its fallback pick.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test quant_mmq_batch_invariance
//!   cd boostr && NUMR_CUDA_TUNE=0 cargo test --features cuda --test quant_mmq_batch_invariance

#![cfg(feature = "cuda")]

use boostr::quant::traits::QuantMatmulOps;
use boostr::quant::{QuantFormat, QuantTensor, QuantizeOps};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

/// Row counts to check: the LM decode batches, the edges of the single-warp
/// tile, the DiT CFG pairs (22 per item), and enough rows to reach the wide
/// token tiles and the second and third token tile.
const ROW_COUNTS: [usize; 11] = [1, 2, 3, 4, 8, 16, 17, 22, 44, 88, 176];

/// Output widths: one feature tile, a DiT projection, and a wide LM one.
const WIDTHS: [usize; 3] = [64, 1536, 4096];

/// K walks: a DiT projection under the split gate, the gate itself, and the
/// 256-multiples above it.
const DEPTHS: [usize; 4] = [1024, 2048, 4096, 6144];

/// A K one block past the deepest whole-group walk, so it is not a whole
/// number of 256-k groups and the last split takes a ragged tail. One block
/// is the format's own: 32 for Q8_0 and Q4_0, 64 for Q2_0, 128 for PQ2_0,
/// Q1_0 and PTQ1_0.
fn ragged_depth(format: QuantFormat) -> usize {
    6144 + format.block_size()
}

fn cuda() -> Option<(CudaClient, CudaDevice)> {
    let device = CudaDevice::new(0);
    let client = CudaClient::new(device.clone()).ok()?;
    Some((client, device))
}

/// Packed weight bytes for `format` at `[n, k]`, through the CPU quantizer.
/// PQ2_0, Q2_0, Q1_0 and PTQ1_0 have no CPU quantize kernel, so their blocks
/// are built directly: every bit pattern of their code run is a valid block
/// (`gguf_base3_trit` maps every byte to a trit at every level).
fn packed_weight(format: QuantFormat, n: usize, k: usize, salt: f32) -> Vec<u8> {
    if matches!(
        format,
        QuantFormat::PQ2_0 | QuantFormat::Q2_0 | QuantFormat::Q1_0 | QuantFormat::PTQ1_0
    ) {
        return lowbit_weight(format, n, k, salt);
    }
    let values: Vec<f32> = (0..n * k)
        .map(|i| ((i % 977) as f32 * 0.031 + salt).sin() + ((i / 977) as f32 * 0.17).cos() * 0.25)
        .collect();
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let input = Tensor::<CpuRuntime>::from_slice(&values, &[n, k], &device).expect("weight");
    client
        .quantize(&input, format)
        .expect("quantize")
        .to_bytes()
        .expect("weight bytes")
}

/// A lowbit weight built byte by byte: f16 `d` and the code run, both varied
/// by block index and `salt` so no two blocks or weights match. `d` sits at
/// byte 0 for PQ2_0, Q2_0 and Q1_0 and at the END of the block (byte 26) for
/// PTQ1_0; the code run fills the other `block_bytes - 2` bytes.
fn lowbit_weight(format: QuantFormat, n: usize, k: usize, salt: f32) -> Vec<u8> {
    let block_bytes = format.block_bytes();
    let bpr = k / format.block_size();
    let (d_off, qs_off) = if format == QuantFormat::PTQ1_0 {
        (block_bytes - 2, 0)
    } else {
        (0, 2)
    };
    let mut out = vec![0u8; n * bpr * block_bytes];
    for block in 0..n * bpr {
        let base = block * block_bytes;
        let d = half::f16::from_f32(0.01 + ((block as f32 * 0.003) + salt * 0.1) % 0.5);
        out[base + d_off..base + d_off + 2].copy_from_slice(&d.to_le_bytes());
        for pos in 0..block_bytes - 2 {
            out[base + qs_off + pos] =
                ((block * 131 + pos * 17 + (salt * 100.0) as usize) % 251) as u8;
        }
    }
    out
}

/// Activation rows with no two alike, so a row landing in the wrong slot
/// changes the answer.
fn activation_rows(rows: usize, k: usize) -> Vec<f32> {
    (0..rows * k)
        .map(|i| {
            let r = (i / k) as f32;
            ((i % k) as f32 * 0.013 + r * 0.7).sin() * 0.4 + (r * 0.31).cos() * 0.05
        })
        .collect()
}

fn act(device: &CudaDevice, rows: &[f32], m: usize, k: usize) -> Tensor<CudaRuntime> {
    Tensor::<CudaRuntime>::from_slice(&rows[..m * k], &[m, k], device).expect("act")
}

/// The three entry points a model reaches, each as `[m, n]` outputs.
fn run_all(
    client: &CudaClient,
    device: &CudaDevice,
    rows: &[f32],
    m: usize,
    k: usize,
    weight: &QuantTensor<CudaRuntime>,
    up: &QuantTensor<CudaRuntime>,
) -> [Vec<f32>; 3] {
    let a = act(device, rows, m, k);
    let single = client
        .quant_matmul(&a, weight)
        .expect("quant_matmul")
        .to_vec::<f32>();
    let batched = client
        .quant_matmul_batch(&a, &[weight, up])
        .expect("quant_matmul_batch")
        .swap_remove(0)
        .to_vec::<f32>();
    let swiglu = client
        .quant_swiglu(&a, weight, up)
        .expect("quant_swiglu")
        .to_vec::<f32>();
    [single, batched, swiglu]
}

fn check_rows(
    what: &str,
    format: QuantFormat,
    n: usize,
    k: usize,
    m: usize,
    out: &[f32],
    alone: &[f32],
) {
    for r in 0..m {
        for c in 0..n {
            let got = out[r * n + c];
            let want = alone[r * n + c];
            assert!(
                got.to_bits() == want.to_bits(),
                "{} {what} N={n} K={k}: row {r} col {c} at M={m} is {got:e} ({:#010x}), \
                 alone it is {want:e} ({:#010x})",
                format.name(),
                got.to_bits(),
                want.to_bits()
            );
        }
    }
}

fn check_shape(client: &CudaClient, device: &CudaDevice, format: QuantFormat, n: usize, k: usize) {
    let max_m = *ROW_COUNTS.iter().max().expect("row counts");
    let weight =
        QuantTensor::from_bytes(&packed_weight(format, n, k, 0.0), format, &[n, k], device)
            .expect("weight");
    let up = QuantTensor::from_bytes(&packed_weight(format, n, k, 1.3), format, &[n, k], device)
        .expect("up weight");
    let rows = activation_rows(max_m, k);

    // Every row on its own: the reference each batched row must reproduce.
    // For the five formats with a single-token kernel this is that kernel's
    // output; `ROW_COUNTS` starts at 1, so the M=1 pass below also checks
    // that entry against itself through each public op.
    let mut alone: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for r in 0..max_m {
        let one = run_all(client, device, &rows[r * k..], 1, k, &weight, &up);
        for (dst, src) in alone.iter_mut().zip(one) {
            dst.extend(src);
        }
    }
    // The batched call and the single call agree at M=1 too: one row through
    // either entry is the same row.
    check_rows("batch vs single", format, n, k, max_m, &alone[1], &alone[0]);

    for &m in &ROW_COUNTS {
        let out = run_all(client, device, &rows, m, k, &weight, &up);
        for (what, got, want) in [
            ("quant_matmul", &out[0], &alone[0]),
            ("quant_matmul_batch", &out[1], &alone[1]),
            ("quant_swiglu", &out[2], &alone[2]),
        ] {
            check_rows(what, format, n, k, m, got, want);
        }
    }
}

fn check_format(format: QuantFormat, ragged: bool) {
    let Some((client, device)) = cuda() else {
        eprintln!("CUDA not available, skipping");
        return;
    };
    let mut depths = DEPTHS.to_vec();
    if ragged {
        depths.push(ragged_depth(format));
    }
    for &k in &depths {
        for &n in &WIDTHS {
            check_shape(&client, &device, format, n, k);
        }
    }
}

#[test]
fn q4_k_rows_do_not_depend_on_the_batch() {
    check_format(QuantFormat::Q4K, false);
}

#[test]
fn q5_k_rows_do_not_depend_on_the_batch() {
    check_format(QuantFormat::Q5K, false);
}

#[test]
fn q6_k_rows_do_not_depend_on_the_batch() {
    check_format(QuantFormat::Q6K, false);
}

#[test]
fn q8_0_rows_do_not_depend_on_the_batch() {
    check_format(QuantFormat::Q8_0, true);
}

#[test]
fn q4_0_rows_do_not_depend_on_the_batch() {
    check_format(QuantFormat::Q4_0, true);
}

#[test]
fn q2_k_rows_do_not_depend_on_the_batch() {
    check_format(QuantFormat::Q2K, false);
}

#[test]
fn pq2_0_rows_do_not_depend_on_the_batch() {
    check_format(QuantFormat::PQ2_0, true);
}

#[test]
fn q2_0_rows_do_not_depend_on_the_batch() {
    check_format(QuantFormat::Q2_0, true);
}

#[test]
fn q1_0_rows_do_not_depend_on_the_batch() {
    check_format(QuantFormat::Q1_0, true);
}

#[test]
fn ptq1_0_rows_do_not_depend_on_the_batch() {
    check_format(QuantFormat::PTQ1_0, true);
}
