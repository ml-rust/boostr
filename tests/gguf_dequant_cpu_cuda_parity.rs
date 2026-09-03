//! `DequantOps::dequantize` CPU vs CUDA, on identical GGUF block bytes, for
//! every GGUF format both backends decode.
//!
//! # Why this file exists
//!
//! GGUF packs element `j` and element `j + 16` of a 32-element 4-bit block
//! into ONE byte: the low nibble is the FIRST half of the block, the high
//! nibble the SECOND half. `src/quant/cpu/kernels/dequant_simple.rs` documents
//! it and the CPU kernels implement it:
//!
//! ```text
//! out[j] = low * d;   out[j + 16] = high * d;
//! ```
//!
//! Several CUDA kernels instead write `out[i*2]` / `out[i*2 + 1]`, which
//! permutes every weight inside each block. Nothing errors — shape, block
//! count and even the RMS of the tensor stay correct — so a model loads, runs,
//! and silently produces wrong numbers. `quant_matmul_generic.cu` in the SAME
//! directory gets it right and cites llama.cpp's `dequantize_row_q4_0` while
//! doing so, which is how the two orderings came to coexist.
//!
//! # The coverage gap this closes
//!
//! No other test compares CPU against CUDA at the value level for any GGUF
//! format. The other quant tests compare a CUDA GEMM against a CUDA GEMV —
//! both wrong together — or against a reference that mirrors the kernel's
//! own dequantisation logic, which proves nothing.
//!
//! # Format coverage
//!
//! `CpuClient::dequantize` (`src/quant/cpu/dequant.rs`) and
//! `CudaClient::dequantize` (`src/quant/cuda/dequant.rs`) both handle ALL 23
//! GGUF formats — CUDA through a dedicated kernel in `kernels/dequant.cu` for
//! twelve of them and `kernels/dequant_generic.cu` for the rest. Nothing is
//! excluded. TCF is not a `QuantFormat` and has its own parity gate in
//! `tests/backend_parity/quant_tcf.rs`.
//!
//! # Fixtures
//!
//! Every fixture uses THREE blocks with a distinct f16 scale per block, and
//! every fixture varies its payload with the element index. A block of
//! uniform bytes decodes identically under either nibble ordering, so a
//! uniform fixture passes while the backend is broken — that is exactly how
//! this defect survived. Each test states how its bytes defeat a permutation.
//!
//! # Tolerance
//!
//! Both backends evaluate the same integer decode and the same f32 scale
//! arithmetic, so agreement is bit-exact except where nvcc contracts a
//! multiply-add into an FMA (one rounding instead of two, at most one ulp).
//! The bound is `1e-6 * (1.0 + max(|cpu|, |cuda|))` — roughly eight f32 ulps,
//! wide enough for FMA contraction and about six orders of magnitude too
//! narrow to hide a permuted element.
//!
//! # Device memory: synchronize between cases
//!
//! Every case here must drop its CUDA tensors and then call
//! `client.synchronize()` before the next one allocates, exactly as
//! `tests/backend_parity/helpers.rs::with_cuda_backend` does. numr's CUDA
//! allocator issues `cuMemFreeAsync` on the shared stream
//! (`src/runtime/cuda/allocator.rs`), so a dropped tensor's memory is NOT
//! reclaimed until the stream reaches that free. A file with this many cases
//! and no synchronize accumulates un-reclaimed stream-ordered frees and then
//! fails `Error::OutOfMemory` on allocations of a few hundred bytes, in an
//! order-dependent way that looks like a per-format defect and is not one.
//! The next person adding a format here inherits that trap.
//!
//! An allocation error is a FAILURE, never a skip: a format that fails to
//! run its comparison must not report `ok`.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test gguf_dequant_cpu_cuda_parity

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::quant::{DequantOps, QuantFormat, QuantMatmulOps, QuantTensor};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

static CUDA_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn cuda_lock() -> std::sync::MutexGuard<'static, ()> {
    CUDA_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

fn cuda_available() -> bool {
    numr::runtime::cuda::is_cuda_available()
}

/// Prints the skip banner to stdout AND stderr, then returns. Deliberately
/// noisy: a silent skip is how the nibble-order defect survived.
fn loud_skip(label: &str, reason: &str) {
    let banner = format!(
        "\n\
         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\
         !! GGUF_DEQUANT_SKIPPED  test=\"{label}\"\n\
         !! REASON: {reason}\n\
         !! NOTHING WAS VERIFIED. This test reported success WITHOUT comparing\n\
         !! the CUDA dequant kernel against the CPU one. Treat the format as\n\
         !! UNTESTED, not as green.\n\
         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    );
    println!("{banner}");
    eprintln!("{banner}");
}

/// Three blocks everywhere: one block cannot expose a per-block indexing or
/// stride error, and two cannot distinguish "off by one block" from "reversed".
const BLOCKS: usize = 3;

/// Per-block f16 scale bits: 1.0, 0.5, 2.0. Exact powers of two, so the scale
/// contributes no rounding of its own, and DISTINCT per block so a block-index
/// error changes the magnitude of every element in that block.
const D_BITS: [u16; BLOCKS] = [0x3C00, 0x3800, 0x4000];

/// Per-block f16 min/second-scale bits: 0.25, 1.0, 0.5. Distinct from the
/// matching `D_BITS` entry so a kernel that swaps the two fields is caught.
const M_BITS: [u16; BLOCKS] = [0x3400, 0x3C00, 0x3800];

/// Per-block 32-bit high-bit words for Q5_0 / Q5_1. `qh` bit `j` is the fifth
/// bit of element `j` and bit `j + 16` that of element `j + 16`, so the first
/// entry — low half all ones, high half all zeros — separates the split-half
/// reading from the interleaved `i*2` / `i*2 + 1` one by a full 16 units on
/// every element. The other two are asymmetric bit patterns, which no
/// permutation of the 32 bit positions maps onto itself.
const QH_WORDS: [u32; BLOCKS] = [0x0000_FFFF, 0xF0F0_0F0F, 0x9C3A_5A73];

/// Deterministic payload byte for a bulk field. Distinct for every index
/// inside a block (the multiplier is odd, so `i * 31` has period 256 and no
/// field here is longer than 256 bytes) and shifted per block, so neither an
/// element permutation nor a block permutation can coincide.
fn payload(i: usize, b: usize) -> u8 {
    ((i as u32)
        .wrapping_mul(31)
        .wrapping_add(b as u32 * 97)
        .wrapping_add(11)
        & 0xFF) as u8
}

/// A byte whose LOW nibble is `(j + b) % 16` and whose HIGH nibble is
/// `(15 - j + b) % 16`. The two nibbles are never equal (they differ by an odd
/// residue), and both vary with `j`, so the 32 decoded elements are all
/// distinct and no reordering of them reproduces the split-half result.
fn nibble_byte(j: usize, b: usize) -> u8 {
    let low = ((j + b) % 16) as u8;
    let high = ((15 - j + b) % 16) as u8;
    low | (high << 4)
}

fn cpu_setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

fn cuda_setup() -> (CudaClient, CudaDevice) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    (client, device)
}

/// Dequantizes `bytes` on both backends and asserts element-wise agreement.
///
/// Always prints one flat `GGUF_DEQUANT_DIAG` line, pass or fail, so a run
/// that is green still records what each format actually produced.
fn assert_dequant_parity(label: &str, format: QuantFormat, bytes: &[u8]) {
    if !cuda_available() {
        loud_skip(label, "CUDA is not available on this machine");
        return;
    }
    let _lock = cuda_lock();

    let numel = BLOCKS * format.block_size();
    assert_eq!(
        bytes.len(),
        BLOCKS * format.block_bytes(),
        "{label}: fixture must be exactly {BLOCKS} blocks"
    );

    let (cpu_client, cpu_device) = cpu_setup();
    let cpu_qt =
        QuantTensor::<CpuRuntime>::from_bytes(bytes, format, &[numel], &cpu_device).unwrap();
    let cpu_out = cpu_client
        .dequantize(&cpu_qt, DType::F32)
        .unwrap()
        .to_vec::<f32>();

    let (cuda_client, cuda_device) = cuda_setup();
    // Drain the previous case's stream-ordered frees before allocating.
    cuda_client.synchronize();

    // Scoped so both device tensors drop before the synchronize below, which
    // is what actually returns their memory to the pool.
    let cuda_out = {
        let cuda_qt = QuantTensor::<CudaRuntime>::from_bytes(bytes, format, &[numel], &cuda_device)
            .unwrap_or_else(|e| {
                panic!(
                    "{label}: allocating {} device bytes for the {format} fixture failed: {e}. \
                     An allocation error here is a FAILURE, not a skip — {format} was NOT \
                     compared. See the module note on stream-ordered frees.",
                    bytes.len()
                )
            });
        cuda_client
            .dequantize(&cuda_qt, DType::F32)
            .unwrap()
            .to_vec::<f32>()
    };
    cuda_client.synchronize();

    assert_eq!(cpu_out.len(), numel, "{label}: CPU element count");
    assert_eq!(cuda_out.len(), numel, "{label}: CUDA element count");

    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut first_mismatch: i64 = -1;
    let mut report = String::new();
    for i in 0..numel {
        let (a, b) = (cpu_out[i], cuda_out[i]);
        let diff = (a - b).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        // Eight-ish f32 ulps at this magnitude: absorbs an FMA contraction on
        // the GPU, far below the O(1) gap a permuted element produces.
        let tol = 1e-6 * (1.0 + a.abs().max(b.abs()));
        if !a.is_finite() || !b.is_finite() || diff > tol {
            if first_mismatch < 0 {
                first_mismatch = i as i64;
            }
            mismatches += 1;
            // The PATTERN of the first mismatches is what separates a nibble
            // permutation from a scale error, so name indices and both values.
            if mismatches <= 16 {
                report.push_str(&format!("\n  idx {i:5}  cpu {a:14.6}  cuda {b:14.6}"));
            }
        }
    }

    println!(
        "GGUF_DEQUANT_DIAG format={format} elements={numel} mismatches={mismatches} \
         max_abs={max_abs:.6e} first_mismatch_idx={first_mismatch}"
    );

    assert_eq!(
        mismatches, 0,
        "{label}: CUDA dequantize disagrees with CPU on {mismatches} of {numel} elements \
         (first at index {first_mismatch}). First mismatches:{report}\n\
         A run of mismatches whose CUDA values are the CPU values of a different \
         position inside the same 32-element block is the split-half vs interleaved \
         nibble ordering; a uniform ratio between the two columns is a scale error."
    );
}

/// Same weight bytes and same activations through `QuantMatmulOps` on both
/// backends. The dequant tests localise a decode defect; this one is the
/// user-visible consequence — a quantized `Linear` producing different logits
/// on GPU than on CPU.
///
/// Tolerance here is looser than the dequant one because the two backends
/// accumulate the K-dimension sum in a different ORDER, which is a legitimate
/// f32 difference. `1e-4` relative to the largest reference magnitude stays
/// orders of magnitude below the change a permuted weight makes to a dot
/// product.
fn assert_matmul_parity(label: &str, format: QuantFormat, weight_bytes: &[u8], n: usize, k: usize) {
    assert_matmul_parity_m(label, format, weight_bytes, 2, n, k);
}

/// Same as `assert_matmul_parity`, with the batch dimension `m` exposed as a
/// parameter. `m` is what selects the CUDA kernel family in
/// `src/quant/cuda/quant_matmul/impl_ops.rs`: `m <= 16` dispatches the GEMV
/// kernel, anything larger dispatches the GEMM kernel — so tests calling this
/// with `m = 2` and tests calling it with `m = 32` exercise different CUDA
/// kernels entirely.
fn assert_matmul_parity_m(
    label: &str,
    format: QuantFormat,
    weight_bytes: &[u8],
    m: usize,
    n: usize,
    k: usize,
) {
    if !cuda_available() {
        loud_skip(label, "CUDA is not available on this machine");
        return;
    }
    let _lock = cuda_lock();

    let act: Vec<f32> = (0..m * k)
        .map(|i| {
            let x = i as f32 * 0.017 + 0.31;
            x.sin() * 0.9 + (x * 2.3).cos() * 0.4
        })
        .collect();

    let (cpu_client, cpu_device) = cpu_setup();
    let cpu_w =
        QuantTensor::<CpuRuntime>::from_bytes(weight_bytes, format, &[n, k], &cpu_device).unwrap();
    let cpu_act = Tensor::<CpuRuntime>::from_slice(&act, &[m, k], &cpu_device).unwrap();
    let cpu_out = cpu_client
        .quant_matmul(&cpu_act, &cpu_w)
        .unwrap()
        .to_vec::<f32>();

    let (cuda_client, cuda_device) = cuda_setup();
    cuda_client.synchronize();

    let cuda_out = {
        let cuda_w =
            QuantTensor::<CudaRuntime>::from_bytes(weight_bytes, format, &[n, k], &cuda_device)
                .unwrap_or_else(|e| {
                    panic!(
                        "{label}: allocating {} device bytes for the {format} weight failed: \
                         {e}. An allocation error here is a FAILURE, not a skip — {format} was \
                         NOT compared. See the module note on stream-ordered frees.",
                        weight_bytes.len()
                    )
                });
        let cuda_act = Tensor::<CudaRuntime>::from_slice(&act, &[m, k], &cuda_device).unwrap();
        cuda_client
            .quant_matmul(&cuda_act, &cuda_w)
            .unwrap()
            .to_vec::<f32>()
    };
    cuda_client.synchronize();

    let scale = cpu_out.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let tol = 1e-4 * (1.0 + scale);

    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut first_mismatch: i64 = -1;
    let mut report = String::new();
    for i in 0..m * n {
        let (a, b) = (cpu_out[i], cuda_out[i]);
        let diff = (a - b).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        if !a.is_finite() || !b.is_finite() || diff > tol {
            if first_mismatch < 0 {
                first_mismatch = i as i64;
            }
            mismatches += 1;
            if mismatches <= 16 {
                report.push_str(&format!("\n  idx {i:5}  cpu {a:14.6}  cuda {b:14.6}"));
            }
        }
    }

    println!(
        "GGUF_DEQUANT_DIAG format={format} path=quant_matmul elements={} mismatches={mismatches} \
         max_abs={max_abs:.6e} first_mismatch_idx={first_mismatch}",
        m * n
    );

    assert_eq!(
        mismatches,
        0,
        "{label}: CUDA quant_matmul disagrees with CPU on {mismatches} of {} outputs \
         (first at index {first_mismatch}). First mismatches:{report}",
        m * n
    );
}

// ── 32-element block formats ─────────────────────────────────────────

/// Q4_0: `d` (f16) + 16 nibble bytes.
///
/// Fixture: `qs[j]`'s low nibble is `(j + b) % 16`, its high nibble
/// `(15 - j + b) % 16`. The two nibbles of a byte are never equal and both
/// move with `j`, so all 32 decoded elements of a block are distinct and the
/// interleaved `out[2i]` / `out[2i+1]` ordering cannot coincide with the
/// split-half `out[j]` / `out[j+16]` one at more than the fixed points.
#[test]
fn q4_0_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 18];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 18..(b + 1) * 18];
        blk[0..2].copy_from_slice(&D_BITS[b].to_le_bytes());
        for j in 0..16 {
            blk[2 + j] = nibble_byte(j, b);
        }
    }
    assert_dequant_parity("q4_0_dequant_matches_cpu", QuantFormat::Q4_0, &data);
}

/// Q4_1: `d` (f16) + `m` (f16) + 16 nibble bytes.
///
/// Same distinct-nibble payload as Q4_0, plus a per-block `m` that differs
/// from that block's `d`, so a kernel that reads the two scale fields in the
/// wrong order is caught alongside the ordering.
#[test]
fn q4_1_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 20];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 20..(b + 1) * 20];
        blk[0..2].copy_from_slice(&D_BITS[b].to_le_bytes());
        blk[2..4].copy_from_slice(&M_BITS[b].to_le_bytes());
        for j in 0..16 {
            blk[4 + j] = nibble_byte(j, b);
        }
    }
    assert_dequant_parity("q4_1_dequant_matches_cpu", QuantFormat::Q4_1, &data);
}

/// Q5_0: `d` (f16) + `qh` (u32) + 16 nibble bytes.
///
/// The nibble payload is the Q4_0 one. `qh` adds a second, independent
/// split-half field: bit `j` is element `j`'s fifth bit and bit `j + 16` is
/// element `j + 16`'s. Block 0's `qh` is `0x0000FFFF` — every first-half
/// element gets the fifth bit, every second-half element does not — which
/// separates the two readings by 16 quantisation steps on every element.
#[test]
fn q5_0_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 22];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 22..(b + 1) * 22];
        blk[0..2].copy_from_slice(&D_BITS[b].to_le_bytes());
        blk[2..6].copy_from_slice(&QH_WORDS[b].to_le_bytes());
        for j in 0..16 {
            blk[6 + j] = nibble_byte(j, b);
        }
    }
    assert_dequant_parity("q5_0_dequant_matches_cpu", QuantFormat::Q5_0, &data);
}

/// Q5_1: `d` (f16) + `m` (f16) + `qh` (u32) + 16 nibble bytes.
///
/// Q5_0's fixture with the extra `m` field, distinct from `d` per block.
#[test]
fn q5_1_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 24];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 24..(b + 1) * 24];
        blk[0..2].copy_from_slice(&D_BITS[b].to_le_bytes());
        blk[2..4].copy_from_slice(&M_BITS[b].to_le_bytes());
        blk[4..8].copy_from_slice(&QH_WORDS[b].to_le_bytes());
        for j in 0..16 {
            blk[8 + j] = nibble_byte(j, b);
        }
    }
    assert_dequant_parity("q5_1_dequant_matches_cpu", QuantFormat::Q5_1, &data);
}

/// Q8_0: `d` (f16) + 32 i8 codes.
///
/// One byte per element, so there is no nibble to permute — but the codes are
/// still all distinct within a block (`i * 37` is a bijection mod 256) and
/// shifted per block, so any reordering or block-stride error still shows.
#[test]
fn q8_0_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 34];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 34..(b + 1) * 34];
        blk[0..2].copy_from_slice(&D_BITS[b].to_le_bytes());
        for i in 0..32 {
            blk[2 + i] = (i as u8).wrapping_mul(37).wrapping_add(b as u8 * 13);
        }
    }
    assert_dequant_parity("q8_0_dequant_matches_cpu", QuantFormat::Q8_0, &data);
}

/// Q8_1: `d` (f16) + `s` (f16) + 32 i8 codes.
///
/// `s` is llama.cpp's precomputed `sum(qs) * d`, used by dot-product kernels
/// and NOT part of the dequantized value. It is deliberately NON-ZERO here:
/// an implementation that folds `s` into the output is only detectable when
/// the field is set, and a zero `s` hides exactly that class of defect the
/// same way uniform nibbles hide the ordering one.
#[test]
fn q8_1_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 36];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 36..(b + 1) * 36];
        blk[0..2].copy_from_slice(&D_BITS[b].to_le_bytes());
        blk[2..4].copy_from_slice(&M_BITS[b].to_le_bytes());
        for i in 0..32 {
            blk[4 + i] = (i as u8).wrapping_mul(37).wrapping_add(b as u8 * 13);
        }
    }
    assert_dequant_parity("q8_1_dequant_matches_cpu", QuantFormat::Q8_1, &data);
}

/// IQ4_NL: `d` (f16) + 16 nibble bytes indexing the non-linear codebook.
///
/// Same distinct-nibble construction as Q4_0. The codebook is strictly
/// increasing, so distinct nibbles stay distinct values and the split-half
/// versus interleaved orderings remain separable.
#[test]
fn iq4_nl_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 18];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 18..(b + 1) * 18];
        blk[0..2].copy_from_slice(&D_BITS[b].to_le_bytes());
        for j in 0..16 {
            blk[2 + j] = nibble_byte(j, b);
        }
    }
    assert_dequant_parity("iq4_nl_dequant_matches_cpu", QuantFormat::IQ4NL, &data);
}

// ── 256-element super-block formats ──────────────────────────────────

/// Q2_K: 16 scale/min bytes + 64 code bytes + `d` + `dmin`.
///
/// Each scale byte carries a different low nibble (sub-block scale) and high
/// nibble (sub-block min), so all 16 sub-blocks decode with different
/// constants and a sub-block permutation cannot coincide. The 64 code bytes
/// are all distinct within a block, and each is read at four different 2-bit
/// shifts, so a shift-order error shows too.
#[test]
fn q2k_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 84];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 84..(b + 1) * 84];
        for (i, slot) in blk.iter_mut().take(16).enumerate() {
            *slot = (i as u8) | (((15 - i) as u8) << 4);
        }
        for i in 0..64 {
            blk[16 + i] = payload(i, b);
        }
        blk[80..82].copy_from_slice(&D_BITS[b].to_le_bytes());
        blk[82..84].copy_from_slice(&M_BITS[b].to_le_bytes());
    }
    assert_dequant_parity("q2k_dequant_matches_cpu", QuantFormat::Q2K, &data);
}

/// Q3_K: 32 hmask bytes + 64 code bytes + 12 packed 6-bit scales + `d`.
///
/// All three payload fields carry index-varying bytes, so the running hmask
/// bit, the 2-bit code shift and the 6-bit scale unpacking each contribute a
/// different value per sub-block. A uniform hmask makes the `-4` branch
/// constant and hides any indexing error in it.
#[test]
fn q3k_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 110];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 110..(b + 1) * 110];
        for (i, slot) in blk.iter_mut().take(32).enumerate() {
            *slot = payload(i, b);
        }
        for i in 0..64 {
            blk[32 + i] = payload(i + 32, b);
        }
        for i in 0..12 {
            blk[96 + i] = payload(i + 96, b);
        }
        blk[108..110].copy_from_slice(&D_BITS[b].to_le_bytes());
    }
    assert_dequant_parity("q3k_dequant_matches_cpu", QuantFormat::Q3K, &data);
}

/// Q4_K: `d` + `dmin` + 12 packed scale bytes + 128 nibble bytes.
///
/// Q4_K splits a byte between two ADJACENT sub-blocks (low nibble to
/// sub-block `2c`, high nibble to `2c + 1`) rather than across halves of one
/// block, so this fixture pins that split: every payload byte has an
/// index-varying value, and the 12 scale bytes give all eight sub-blocks
/// different scale/min pairs, so swapping any two sub-blocks changes the
/// output.
#[test]
fn q4k_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 144];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 144..(b + 1) * 144];
        blk[0..2].copy_from_slice(&D_BITS[b].to_le_bytes());
        blk[2..4].copy_from_slice(&M_BITS[b].to_le_bytes());
        for i in 0..12 {
            blk[4 + i] = payload(i, b);
        }
        for i in 0..128 {
            blk[16 + i] = payload(i + 12, b);
        }
    }
    assert_dequant_parity("q4k_dequant_matches_cpu", QuantFormat::Q4K, &data);
}

/// Q5_K: `d` + `dmin` + 12 scale bytes + 32 `qh` bytes + 128 nibble bytes.
///
/// The fifth bit of element `j * 32 + l` is bit `j` of `qh[l]` — an
/// index-varying `qh` therefore gives a different fifth bit to each of the
/// eight sub-blocks at the same `l`, which a kernel addressing `qh` by a flat
/// element index cannot reproduce. The nibble payload varies with the index
/// as well, so the low/high nibble split across sub-blocks is pinned too.
#[test]
fn q5k_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 176];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 176..(b + 1) * 176];
        blk[0..2].copy_from_slice(&D_BITS[b].to_le_bytes());
        blk[2..4].copy_from_slice(&M_BITS[b].to_le_bytes());
        for i in 0..12 {
            blk[4 + i] = payload(i, b);
        }
        for i in 0..32 {
            blk[16 + i] = payload(i + 12, b);
        }
        for i in 0..128 {
            blk[48 + i] = payload(i + 44, b);
        }
    }
    assert_dequant_parity("q5k_dequant_matches_cpu", QuantFormat::Q5K, &data);
}

/// Q6_K: 128 `ql` bytes + 64 `qh` bytes + 16 i8 scales + `d`.
///
/// Four elements share one `qh` byte at four different 2-bit shifts and two
/// `ql` bytes at two nibbles, and the 16 scales are all different, so both
/// the shift assignment and the `l / 16` scale index are pinned by
/// index-varying payloads.
#[test]
fn q6k_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 210];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 210..(b + 1) * 210];
        for (i, slot) in blk.iter_mut().take(128).enumerate() {
            *slot = payload(i, b);
        }
        for i in 0..64 {
            blk[128 + i] = payload(i + 128, b);
        }
        for i in 0..16 {
            blk[192 + i] = payload(i + 192, b);
        }
        blk[208..210].copy_from_slice(&D_BITS[b].to_le_bytes());
    }
    assert_dequant_parity("q6k_dequant_matches_cpu", QuantFormat::Q6K, &data);
}

/// Q8_K: f32 `d` + 256 i8 codes + 16 i16 `bsums`.
///
/// The only GGUF format with an f32 scale, so `d` is written as f32 bits
/// (0.5, 0.25, 1.0 — exact) rather than f16. Codes are distinct per index.
/// `bsums` are a dot-product accelerator and must not reach the output; they
/// are set NON-ZERO so a kernel that folds them in is detected.
#[test]
fn q8k_dequant_matches_cpu() {
    let d_f32: [f32; BLOCKS] = [0.5, 0.25, 1.0];
    let mut data = vec![0u8; BLOCKS * 292];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 292..(b + 1) * 292];
        blk[0..4].copy_from_slice(&d_f32[b].to_le_bytes());
        for i in 0..256 {
            blk[4 + i] = payload(i, b);
        }
        for i in 0..32 {
            blk[260 + i] = payload(i + 7, b);
        }
    }
    assert_dequant_parity("q8k_dequant_matches_cpu", QuantFormat::Q8K, &data);
}

/// IQ1_S: `d` (f16) + 32 `qs` bytes + 16 `qh` sign bytes.
///
/// Each of the 16 sub-blocks takes its 12-bit grid index from a different
/// `qs` pair and its signs from a different `qh` byte, so index-varying
/// payloads give all 16 sub-blocks different values and different sign
/// patterns. A uniform payload makes every sub-block identical and hides
/// any sub-block indexing error.
#[test]
fn iq1s_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 50];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 50..(b + 1) * 50];
        blk[0..2].copy_from_slice(&D_BITS[b].to_le_bytes());
        for i in 0..48 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_dequant_parity("iq1s_dequant_matches_cpu", QuantFormat::IQ1S, &data);
}

/// IQ1_M: `d` (f16) + 6 packed 3-bit scale bytes + 32 `qs` + 16 `qh`.
///
/// The 3-bit scales straddle byte boundaries, so an index-varying scale field
/// gives the 16 sub-blocks 16 different scales and pins the bit-offset
/// arithmetic; grid and sign payloads vary as in IQ1_S.
#[test]
fn iq1m_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 56];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 56..(b + 1) * 56];
        blk[0..2].copy_from_slice(&D_BITS[b].to_le_bytes());
        for i in 0..54 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_dequant_parity("iq1m_dequant_matches_cpu", QuantFormat::IQ1M, &data);
}

/// IQ2_XXS: `d` (f16) + eight 8-byte groups.
///
/// Each group's u64 supplies four grid indices, four 7-bit sign fields and a
/// 4-bit sub-scale. An index-varying payload gives all eight groups different
/// sub-scales and all 32 sub-groups different grids and signs, so a group or
/// sub-group permutation changes the output.
#[test]
fn iq2xxs_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 66];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 66..(b + 1) * 66];
        blk[0..2].copy_from_slice(&D_BITS[b].to_le_bytes());
        for i in 0..64 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_dequant_parity("iq2xxs_dequant_matches_cpu", QuantFormat::IQ2XXS, &data);
}

/// IQ2_XS: `d` (f16) + 16 signed scale bytes + 56 `qs` bytes.
///
/// The 16 scale bytes are read as i8, so an index-varying field gives both
/// positive and negative sub-block scales — a sub-block permutation then
/// changes signs as well as magnitudes. Grid index and sign field come from
/// the same 16-bit word, both varying per sub-block.
#[test]
fn iq2xs_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 74];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 74..(b + 1) * 74];
        blk[0..2].copy_from_slice(&D_BITS[b].to_le_bytes());
        for i in 0..72 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_dequant_parity("iq2xs_dequant_matches_cpu", QuantFormat::IQ2XS, &data);
}

/// IQ2_S: `d` (f16) + 32 `qs` + 4 `qh` + 16 sign bytes + 28 scale bytes.
///
/// Four separately-addressed fields, each index-varying, so a kernel that
/// mis-places any field boundary decodes different values.
#[test]
fn iq2s_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 82];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 82..(b + 1) * 82];
        blk[0..2].copy_from_slice(&D_BITS[b].to_le_bytes());
        for i in 0..80 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_dequant_parity("iq2s_dequant_matches_cpu", QuantFormat::IQ2S, &data);
}

/// IQ3_XXS: `d` (f16) + eight 12-byte groups (8 grid bytes + 4 sign/scale).
///
/// Index-varying payload gives each group a different sub-scale and each
/// sub-group a different 16-bit grid index and sign byte.
#[test]
fn iq3xxs_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 98];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 98..(b + 1) * 98];
        blk[0..2].copy_from_slice(&D_BITS[b].to_le_bytes());
        for i in 0..96 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_dequant_parity("iq3xxs_dequant_matches_cpu", QuantFormat::IQ3XXS, &data);
}

/// IQ3_S: `d` (f16) + 32 `qs` + 4 `qh` + 32 sign bytes + 8 scale bytes.
///
/// Each of the 8 sub-blocks takes 32 elements from four `qs` bytes at four
/// 2-bit shifts, a high bit from a flat 256-bit `qh` field, a sign byte per 8
/// elements and its own 4-bit scale. Index-varying payloads pin all four
/// indexings independently; in particular the 2-bit shift within a `qs` byte
/// is only observable when neighbouring 2-bit fields differ.
#[test]
fn iq3s_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 110];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 110..(b + 1) * 110];
        blk[0..2].copy_from_slice(&D_BITS[b].to_le_bytes());
        for i in 0..108 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_dequant_parity("iq3s_dequant_matches_cpu", QuantFormat::IQ3S, &data);
}

/// IQ4_XS: `d` (f16) + `scales_h` (u16) + 4 `scales_l` bytes + 128 nibble bytes.
///
/// Two permutations at once. `scales_h` is a TWO-byte field at offset 2..4
/// carrying 2 high scale bits for all EIGHT sub-blocks, with `scales_l` at
/// 4..8 — reading `scales_h` as one byte shifts `scales_l` by one and drops
/// the high bits of sub-blocks 4..8, so the value here (`0xB1E4`) gives all
/// eight sub-blocks different 2-bit fields and `scales_l` gives all eight
/// different low nibbles. The 128 `qs` bytes use the Q4_0 distinct-nibble
/// construction, pinning the split-half order within each 32-element
/// sub-block.
#[test]
fn iq4xs_dequant_matches_cpu() {
    let scales_l: [u8; 4] = [0x21, 0x43, 0x65, 0x87];
    let mut data = vec![0u8; BLOCKS * 136];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 136..(b + 1) * 136];
        blk[0..2].copy_from_slice(&D_BITS[b].to_le_bytes());
        blk[2..4].copy_from_slice(&0xB1E4u16.to_le_bytes());
        blk[4..8].copy_from_slice(&scales_l);
        for sb in 0..8 {
            for j in 0..16 {
                blk[8 + sb * 16 + j] = nibble_byte(j, b + sb);
            }
        }
    }
    assert_dequant_parity("iq4xs_dequant_matches_cpu", QuantFormat::IQ4XS, &data);
}

/// TQ1_0: `d` (f16) + 52 bytes of base-3 packed ternary codes.
///
/// Five elements per byte via repeated `% 3` / `/= 3`, so an index-varying
/// payload makes consecutive elements differ and pins the unpacking order.
#[test]
fn tq1_0_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 54];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 54..(b + 1) * 54];
        blk[0..2].copy_from_slice(&D_BITS[b].to_le_bytes());
        for i in 0..52 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_dequant_parity("tq1_0_dequant_matches_cpu", QuantFormat::TQ1_0, &data);
}

/// TQ2_0: `d` (f16) + 64 bytes of 2-bit packed ternary codes.
///
/// Four elements per byte at four 2-bit shifts; index-varying bytes make the
/// four fields of a byte differ, so a shift-order error changes the output.
#[test]
fn tq2_0_dequant_matches_cpu() {
    let mut data = vec![0u8; BLOCKS * 66];
    for b in 0..BLOCKS {
        let blk = &mut data[b * 66..(b + 1) * 66];
        blk[0..2].copy_from_slice(&D_BITS[b].to_le_bytes());
        for i in 0..64 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_dequant_parity("tq2_0_dequant_matches_cpu", QuantFormat::TQ2_0, &data);
}

// ── The user-visible consequence: quantized matmul ───────────────────
//
// A permuted weight inside a block is invisible to shape checks and to any
// summary statistic of the tensor, but it changes every dot product the
// weight takes part in. These run the SAME bytes as the dequant fixtures
// above through `QuantMatmulOps::quant_matmul`, which is the path a
// quantized `Linear` actually takes, on both backends with identical
// activations. They cover the formats whose CUDA dequant kernels use the
// interleaved nibble order.

/// Q4_0 weight `[3, 64]` — 2 blocks per row, 6 blocks total.
#[test]
fn q4_0_quant_matmul_matches_cpu() {
    let (n, k) = (3usize, 64usize);
    let blocks = n * k / 32;
    let mut data = vec![0u8; blocks * 18];
    for b in 0..blocks {
        let blk = &mut data[b * 18..(b + 1) * 18];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        for j in 0..16 {
            blk[2 + j] = nibble_byte(j, b);
        }
    }
    assert_matmul_parity(
        "q4_0_quant_matmul_matches_cpu",
        QuantFormat::Q4_0,
        &data,
        n,
        k,
    );
}

/// Q4_1 weight `[3, 64]`.
#[test]
fn q4_1_quant_matmul_matches_cpu() {
    let (n, k) = (3usize, 64usize);
    let blocks = n * k / 32;
    let mut data = vec![0u8; blocks * 20];
    for b in 0..blocks {
        let blk = &mut data[b * 20..(b + 1) * 20];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        blk[2..4].copy_from_slice(&M_BITS[b % BLOCKS].to_le_bytes());
        for j in 0..16 {
            blk[4 + j] = nibble_byte(j, b);
        }
    }
    assert_matmul_parity(
        "q4_1_quant_matmul_matches_cpu",
        QuantFormat::Q4_1,
        &data,
        n,
        k,
    );
}

/// Q5_0 weight `[3, 64]`.
#[test]
fn q5_0_quant_matmul_matches_cpu() {
    let (n, k) = (3usize, 64usize);
    let blocks = n * k / 32;
    let mut data = vec![0u8; blocks * 22];
    for b in 0..blocks {
        let blk = &mut data[b * 22..(b + 1) * 22];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        blk[2..6].copy_from_slice(&QH_WORDS[b % BLOCKS].to_le_bytes());
        for j in 0..16 {
            blk[6 + j] = nibble_byte(j, b);
        }
    }
    assert_matmul_parity(
        "q5_0_quant_matmul_matches_cpu",
        QuantFormat::Q5_0,
        &data,
        n,
        k,
    );
}

/// Q5_1 weight `[3, 64]`.
#[test]
fn q5_1_quant_matmul_matches_cpu() {
    let (n, k) = (3usize, 64usize);
    let blocks = n * k / 32;
    let mut data = vec![0u8; blocks * 24];
    for b in 0..blocks {
        let blk = &mut data[b * 24..(b + 1) * 24];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        blk[2..4].copy_from_slice(&M_BITS[b % BLOCKS].to_le_bytes());
        blk[4..8].copy_from_slice(&QH_WORDS[b % BLOCKS].to_le_bytes());
        for j in 0..16 {
            blk[8 + j] = nibble_byte(j, b);
        }
    }
    assert_matmul_parity(
        "q5_1_quant_matmul_matches_cpu",
        QuantFormat::Q5_1,
        &data,
        n,
        k,
    );
}

/// IQ4_NL weight `[3, 64]`.
#[test]
fn iq4_nl_quant_matmul_matches_cpu() {
    let (n, k) = (3usize, 64usize);
    let blocks = n * k / 32;
    let mut data = vec![0u8; blocks * 18];
    for b in 0..blocks {
        let blk = &mut data[b * 18..(b + 1) * 18];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        for j in 0..16 {
            blk[2 + j] = nibble_byte(j, b);
        }
    }
    assert_matmul_parity(
        "iq4_nl_quant_matmul_matches_cpu",
        QuantFormat::IQ4NL,
        &data,
        n,
        k,
    );
}

/// IQ1_S weight `[3, 512]` — 2 blocks per row, 6 blocks total.
#[test]
fn iq1_s_quant_matmul_matches_cpu() {
    let (n, k) = (3usize, 512usize);
    let blocks = n * k / 256;
    let mut data = vec![0u8; blocks * 50];
    for b in 0..blocks {
        let blk = &mut data[b * 50..(b + 1) * 50];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        for i in 0..48 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_matmul_parity(
        "iq1_s_quant_matmul_matches_cpu",
        QuantFormat::IQ1S,
        &data,
        n,
        k,
    );
}

/// IQ1_M weight `[3, 512]` — 2 blocks per row, 6 blocks total.
#[test]
fn iq1_m_quant_matmul_matches_cpu() {
    let (n, k) = (3usize, 512usize);
    let blocks = n * k / 256;
    let mut data = vec![0u8; blocks * 56];
    for b in 0..blocks {
        let blk = &mut data[b * 56..(b + 1) * 56];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        for i in 0..54 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_matmul_parity(
        "iq1_m_quant_matmul_matches_cpu",
        QuantFormat::IQ1M,
        &data,
        n,
        k,
    );
}

/// IQ2_XXS weight `[3, 512]` — 2 blocks per row, 6 blocks total.
#[test]
fn iq2_xxs_quant_matmul_matches_cpu() {
    let (n, k) = (3usize, 512usize);
    let blocks = n * k / 256;
    let mut data = vec![0u8; blocks * 66];
    for b in 0..blocks {
        let blk = &mut data[b * 66..(b + 1) * 66];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        for i in 0..64 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_matmul_parity(
        "iq2_xxs_quant_matmul_matches_cpu",
        QuantFormat::IQ2XXS,
        &data,
        n,
        k,
    );
}

/// IQ2_XS weight `[3, 512]` — 2 blocks per row, 6 blocks total.
#[test]
fn iq2_xs_quant_matmul_matches_cpu() {
    let (n, k) = (3usize, 512usize);
    let blocks = n * k / 256;
    let mut data = vec![0u8; blocks * 74];
    for b in 0..blocks {
        let blk = &mut data[b * 74..(b + 1) * 74];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        for i in 0..72 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_matmul_parity(
        "iq2_xs_quant_matmul_matches_cpu",
        QuantFormat::IQ2XS,
        &data,
        n,
        k,
    );
}

/// IQ2_S weight `[3, 512]` — 2 blocks per row, 6 blocks total.
#[test]
fn iq2_s_quant_matmul_matches_cpu() {
    let (n, k) = (3usize, 512usize);
    let blocks = n * k / 256;
    let mut data = vec![0u8; blocks * 82];
    for b in 0..blocks {
        let blk = &mut data[b * 82..(b + 1) * 82];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        for i in 0..80 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_matmul_parity(
        "iq2_s_quant_matmul_matches_cpu",
        QuantFormat::IQ2S,
        &data,
        n,
        k,
    );
}

/// IQ3_XXS weight `[3, 512]` — 2 blocks per row, 6 blocks total.
#[test]
fn iq3_xxs_quant_matmul_matches_cpu() {
    let (n, k) = (3usize, 512usize);
    let blocks = n * k / 256;
    let mut data = vec![0u8; blocks * 98];
    for b in 0..blocks {
        let blk = &mut data[b * 98..(b + 1) * 98];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        for i in 0..96 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_matmul_parity(
        "iq3_xxs_quant_matmul_matches_cpu",
        QuantFormat::IQ3XXS,
        &data,
        n,
        k,
    );
}

/// IQ3_S weight `[3, 512]` — 2 blocks per row, 6 blocks total.
#[test]
fn iq3_s_quant_matmul_matches_cpu() {
    let (n, k) = (3usize, 512usize);
    let blocks = n * k / 256;
    let mut data = vec![0u8; blocks * 110];
    for b in 0..blocks {
        let blk = &mut data[b * 110..(b + 1) * 110];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        for i in 0..108 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_matmul_parity(
        "iq3_s_quant_matmul_matches_cpu",
        QuantFormat::IQ3S,
        &data,
        n,
        k,
    );
}

/// IQ4_XS weight `[3, 512]` — 2 super-blocks per row, 6 total.
#[test]
fn iq4_xs_quant_matmul_matches_cpu() {
    let (n, k) = (3usize, 512usize);
    let blocks = n * k / 256;
    let scales_l: [u8; 4] = [0x21, 0x43, 0x65, 0x87];
    let mut data = vec![0u8; blocks * 136];
    for b in 0..blocks {
        let blk = &mut data[b * 136..(b + 1) * 136];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        blk[2..4].copy_from_slice(&0xB1E4u16.to_le_bytes());
        blk[4..8].copy_from_slice(&scales_l);
        for sb in 0..8 {
            for j in 0..16 {
                blk[8 + sb * 16 + j] = nibble_byte(j, b + sb);
            }
        }
    }
    assert_matmul_parity(
        "iq4_xs_quant_matmul_matches_cpu",
        QuantFormat::IQ4XS,
        &data,
        n,
        k,
    );
}

// ── GEMM path (m = 32) ────────────────────────────────────────────────
//
// Every case above runs with `m = 2`, which is at or below the `m <= 16`
// threshold in `src/quant/cuda/quant_matmul/impl_ops.rs`, so it dispatches
// the GEMV kernel. The GEMM kernels in `src/quant/cuda/kernels/gemm/` are
// otherwise exercised by no test at all. These cases repeat each weight
// fixture above with `m = 32` to force the GEMM path instead.

/// Q4_0 weight `[3, 64]` — 2 blocks per row, 6 blocks total. `m = 32` is
/// above the `m <= 16` threshold, so this forces the GEMM path rather than
/// GEMV.
#[test]
fn q4_0_quant_matmul_gemm_matches_cpu() {
    let (n, k) = (3usize, 64usize);
    let blocks = n * k / 32;
    let mut data = vec![0u8; blocks * 18];
    for b in 0..blocks {
        let blk = &mut data[b * 18..(b + 1) * 18];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        for j in 0..16 {
            blk[2 + j] = nibble_byte(j, b);
        }
    }
    assert_matmul_parity_m(
        "q4_0_quant_matmul_gemm_matches_cpu",
        QuantFormat::Q4_0,
        &data,
        32,
        n,
        k,
    );
}

/// Q4_1 weight `[3, 64]` — 2 blocks per row, 6 blocks total. `m = 32` is
/// above the `m <= 16` threshold, so this forces the GEMM path rather than
/// GEMV.
#[test]
fn q4_1_quant_matmul_gemm_matches_cpu() {
    let (n, k) = (3usize, 64usize);
    let blocks = n * k / 32;
    let mut data = vec![0u8; blocks * 20];
    for b in 0..blocks {
        let blk = &mut data[b * 20..(b + 1) * 20];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        blk[2..4].copy_from_slice(&M_BITS[b % BLOCKS].to_le_bytes());
        for j in 0..16 {
            blk[4 + j] = nibble_byte(j, b);
        }
    }
    assert_matmul_parity_m(
        "q4_1_quant_matmul_gemm_matches_cpu",
        QuantFormat::Q4_1,
        &data,
        32,
        n,
        k,
    );
}

/// Q5_0 weight `[3, 64]` — 2 blocks per row, 6 blocks total. `m = 32` is
/// above the `m <= 16` threshold, so this forces the GEMM path rather than
/// GEMV.
#[test]
fn q5_0_quant_matmul_gemm_matches_cpu() {
    let (n, k) = (3usize, 64usize);
    let blocks = n * k / 32;
    let mut data = vec![0u8; blocks * 22];
    for b in 0..blocks {
        let blk = &mut data[b * 22..(b + 1) * 22];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        blk[2..6].copy_from_slice(&QH_WORDS[b % BLOCKS].to_le_bytes());
        for j in 0..16 {
            blk[6 + j] = nibble_byte(j, b);
        }
    }
    assert_matmul_parity_m(
        "q5_0_quant_matmul_gemm_matches_cpu",
        QuantFormat::Q5_0,
        &data,
        32,
        n,
        k,
    );
}

/// Q5_1 weight `[3, 64]` — 2 blocks per row, 6 blocks total. `m = 32` is
/// above the `m <= 16` threshold, so this forces the GEMM path rather than
/// GEMV.
#[test]
fn q5_1_quant_matmul_gemm_matches_cpu() {
    let (n, k) = (3usize, 64usize);
    let blocks = n * k / 32;
    let mut data = vec![0u8; blocks * 24];
    for b in 0..blocks {
        let blk = &mut data[b * 24..(b + 1) * 24];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        blk[2..4].copy_from_slice(&M_BITS[b % BLOCKS].to_le_bytes());
        blk[4..8].copy_from_slice(&QH_WORDS[b % BLOCKS].to_le_bytes());
        for j in 0..16 {
            blk[8 + j] = nibble_byte(j, b);
        }
    }
    assert_matmul_parity_m(
        "q5_1_quant_matmul_gemm_matches_cpu",
        QuantFormat::Q5_1,
        &data,
        32,
        n,
        k,
    );
}

/// IQ4_NL weight `[3, 64]` — 2 blocks per row, 6 blocks total. `m = 32` is
/// above the `m <= 16` threshold, so this forces the GEMM path rather than
/// GEMV.
#[test]
fn iq4_nl_quant_matmul_gemm_matches_cpu() {
    let (n, k) = (3usize, 64usize);
    let blocks = n * k / 32;
    let mut data = vec![0u8; blocks * 18];
    for b in 0..blocks {
        let blk = &mut data[b * 18..(b + 1) * 18];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        for j in 0..16 {
            blk[2 + j] = nibble_byte(j, b);
        }
    }
    assert_matmul_parity_m(
        "iq4_nl_quant_matmul_gemm_matches_cpu",
        QuantFormat::IQ4NL,
        &data,
        32,
        n,
        k,
    );
}

/// IQ1_S weight `[3, 512]` — 2 blocks per row, 6 blocks total. `m = 32` is
/// above the `m <= 16` threshold, so this forces the GEMM path rather than
/// GEMV.
#[test]
fn iq1_s_quant_matmul_gemm_matches_cpu() {
    let (n, k) = (3usize, 512usize);
    let blocks = n * k / 256;
    let mut data = vec![0u8; blocks * 50];
    for b in 0..blocks {
        let blk = &mut data[b * 50..(b + 1) * 50];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        for i in 0..48 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_matmul_parity_m(
        "iq1_s_quant_matmul_gemm_matches_cpu",
        QuantFormat::IQ1S,
        &data,
        32,
        n,
        k,
    );
}

/// IQ1_M weight `[3, 512]` — 2 blocks per row, 6 blocks total. `m = 32` is
/// above the `m <= 16` threshold, so this forces the GEMM path rather than
/// GEMV.
#[test]
fn iq1_m_quant_matmul_gemm_matches_cpu() {
    let (n, k) = (3usize, 512usize);
    let blocks = n * k / 256;
    let mut data = vec![0u8; blocks * 56];
    for b in 0..blocks {
        let blk = &mut data[b * 56..(b + 1) * 56];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        for i in 0..54 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_matmul_parity_m(
        "iq1_m_quant_matmul_gemm_matches_cpu",
        QuantFormat::IQ1M,
        &data,
        32,
        n,
        k,
    );
}

/// IQ2_XXS weight `[3, 512]` — 2 blocks per row, 6 blocks total. `m = 32`
/// is above the `m <= 16` threshold, so this forces the GEMM path rather
/// than GEMV.
#[test]
fn iq2_xxs_quant_matmul_gemm_matches_cpu() {
    let (n, k) = (3usize, 512usize);
    let blocks = n * k / 256;
    let mut data = vec![0u8; blocks * 66];
    for b in 0..blocks {
        let blk = &mut data[b * 66..(b + 1) * 66];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        for i in 0..64 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_matmul_parity_m(
        "iq2_xxs_quant_matmul_gemm_matches_cpu",
        QuantFormat::IQ2XXS,
        &data,
        32,
        n,
        k,
    );
}

/// IQ2_XS weight `[3, 512]` — 2 blocks per row, 6 blocks total. `m = 32`
/// is above the `m <= 16` threshold, so this forces the GEMM path rather
/// than GEMV.
#[test]
fn iq2_xs_quant_matmul_gemm_matches_cpu() {
    let (n, k) = (3usize, 512usize);
    let blocks = n * k / 256;
    let mut data = vec![0u8; blocks * 74];
    for b in 0..blocks {
        let blk = &mut data[b * 74..(b + 1) * 74];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        for i in 0..72 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_matmul_parity_m(
        "iq2_xs_quant_matmul_gemm_matches_cpu",
        QuantFormat::IQ2XS,
        &data,
        32,
        n,
        k,
    );
}

/// IQ2_S weight `[3, 512]` — 2 blocks per row, 6 blocks total. `m = 32` is
/// above the `m <= 16` threshold, so this forces the GEMM path rather than
/// GEMV.
#[test]
fn iq2_s_quant_matmul_gemm_matches_cpu() {
    let (n, k) = (3usize, 512usize);
    let blocks = n * k / 256;
    let mut data = vec![0u8; blocks * 82];
    for b in 0..blocks {
        let blk = &mut data[b * 82..(b + 1) * 82];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        for i in 0..80 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_matmul_parity_m(
        "iq2_s_quant_matmul_gemm_matches_cpu",
        QuantFormat::IQ2S,
        &data,
        32,
        n,
        k,
    );
}

/// IQ3_XXS weight `[3, 512]` — 2 blocks per row, 6 blocks total. `m = 32`
/// is above the `m <= 16` threshold, so this forces the GEMM path rather
/// than GEMV.
#[test]
fn iq3_xxs_quant_matmul_gemm_matches_cpu() {
    let (n, k) = (3usize, 512usize);
    let blocks = n * k / 256;
    let mut data = vec![0u8; blocks * 98];
    for b in 0..blocks {
        let blk = &mut data[b * 98..(b + 1) * 98];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        for i in 0..96 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_matmul_parity_m(
        "iq3_xxs_quant_matmul_gemm_matches_cpu",
        QuantFormat::IQ3XXS,
        &data,
        32,
        n,
        k,
    );
}

/// IQ3_S weight `[3, 512]` — 2 blocks per row, 6 blocks total. `m = 32` is
/// above the `m <= 16` threshold, so this forces the GEMM path rather than
/// GEMV.
#[test]
fn iq3_s_quant_matmul_gemm_matches_cpu() {
    let (n, k) = (3usize, 512usize);
    let blocks = n * k / 256;
    let mut data = vec![0u8; blocks * 110];
    for b in 0..blocks {
        let blk = &mut data[b * 110..(b + 1) * 110];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        for i in 0..108 {
            blk[2 + i] = payload(i, b);
        }
    }
    assert_matmul_parity_m(
        "iq3_s_quant_matmul_gemm_matches_cpu",
        QuantFormat::IQ3S,
        &data,
        32,
        n,
        k,
    );
}

/// IQ4_XS weight `[3, 512]` — 2 super-blocks per row, 6 total. `m = 32` is
/// above the `m <= 16` threshold, so this forces the GEMM path rather than
/// GEMV.
#[test]
fn iq4_xs_quant_matmul_gemm_matches_cpu() {
    let (n, k) = (3usize, 512usize);
    let blocks = n * k / 256;
    let scales_l: [u8; 4] = [0x21, 0x43, 0x65, 0x87];
    let mut data = vec![0u8; blocks * 136];
    for b in 0..blocks {
        let blk = &mut data[b * 136..(b + 1) * 136];
        blk[0..2].copy_from_slice(&D_BITS[b % BLOCKS].to_le_bytes());
        blk[2..4].copy_from_slice(&0xB1E4u16.to_le_bytes());
        blk[4..8].copy_from_slice(&scales_l);
        for sb in 0..8 {
            for j in 0..16 {
                blk[8 + sb * 16 + j] = nibble_byte(j, b + sb);
            }
        }
    }
    assert_matmul_parity_m(
        "iq4_xs_quant_matmul_gemm_matches_cpu",
        QuantFormat::IQ4XS,
        &data,
        32,
        n,
        k,
    );
}
