//! Proves `quant_mmq_q8_0_q8_1_mma` is bit-identical to `quant_mmq_q8_0_q8_1`.
//!
//! Both kernels stage the same Q8_0/Q8_1 bytes into shared memory, accumulate
//! the same int8 products in int32, and apply the same two f16 scales once
//! per 32-element block in the same block order. Nothing about that changes
//! between dp4a and `mma.sync.aligned.m16n8k32` — only the instruction that
//! does the multiply-accumulate. The outputs must match to the bit, not to a
//! float tolerance.
//!
//! The PQ2_0, Q2_0 and Q1_0 tests at the bottom check the feature-major
//! tensor-core kernel against the token-batched dp4a GEMV on one Q8_1
//! activation. Those two do NOT share a float order, so they are held to a
//! magnitude-relative tolerance; see [`lowbit_mma_matches_gemv`]. PTQ1_0 has
//! no dp4a GEMV, so its feature-major kernel is checked against the F32 GEMV
//! `quant_gemv_ptq1_0_f32` on the Q8_1 activation dequantized back to f32;
//! see [`ptq1_0_mma_matches_f32_gemv`].
//!
//! The `*_gemv1_matches_mma_at_one_token` tests at the bottom check the
//! single-token kernel `quant_mmq_<fmt>_q8_1_gemv1`, which the public
//! `quant_matmul` takes at M = 1, against the feature-major tensor-core
//! kernel forced through `quant_matmul_forced_schedule` on the same input.
//! Those two DO share a float order — same int8 lanes, same scales, same
//! expression, same chunk order, same K ranges — so they are held to bit
//! equality; see [`gemv1_matches_mma`].
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test quant_mmq_mma_parity

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::quant::cuda::kernels::{
    self, GEMV_PQ2_0_MODULE, GEMV_PTQ1_0_MODULE, GEMV_Q1_0_MODULE, GEMV_Q2_0_MODULE,
    QUANT_GEMV_MODULE, QUANT_MMQ_MMA_MODULE,
};
use boostr::quant::cuda::quant_matmul::forced_tile::quant_matmul_forced_schedule;
use boostr::quant::cuda::quant_matmul::mmq_feat_major::Schedule;
use boostr::quant::traits::QuantMatmulOps;
use boostr::quant::{QuantFormat, QuantTensor};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::runtime::Device;
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

/// Deterministic pseudo-random `i8` quant, varied by block index and
/// position so no permutation of the payload coincides with another.
fn quant_byte(block: usize, pos: usize) -> i8 {
    (((block * 131 + pos * 17) % 251) as i32 - 125) as i8
}

/// Plausible per-block f16 scale, varied by block index.
fn block_scale(block: usize) -> half::f16 {
    half::f16::from_f32(0.01 + (block as f32 * 0.003) % 0.5)
}

/// Builds a Q8_0 weight buffer: `n * (k / 32)` blocks of 34 bytes, half scale
/// at byte 0, 32 `i8` quants at byte 2.
fn build_q8_0_weight(n: usize, k: usize) -> Vec<u8> {
    let bpr = k / 32;
    let mut out = vec![0u8; n * bpr * 34];
    for row in 0..n {
        for b in 0..bpr {
            let block = row * bpr + b;
            let base = block * 34;
            out[base..base + 2].copy_from_slice(&block_scale(block).to_le_bytes());
            for pos in 0..32 {
                out[base + 2 + pos] = quant_byte(block, pos) as u8;
            }
        }
    }
    out
}

/// Builds a Q8_1 activation buffer: `m * (k / 32)` blocks of 36 bytes, half
/// scale at byte 0, block sum at byte 2 (unused by the kernel), 32 `i8`
/// quants at byte 4.
fn build_q8_1_activation(m: usize, k: usize) -> Vec<u8> {
    let bpr = k / 32;
    let mut out = vec![0u8; m * bpr * 36];
    for row in 0..m {
        for b in 0..bpr {
            let block = row * bpr + b;
            let base = block * 36;
            out[base..base + 2].copy_from_slice(&block_scale(block).to_le_bytes());
            out[base + 2..base + 4].copy_from_slice(&0i16.to_le_bytes());
            for pos in 0..32 {
                // Offset the position stream so activation and weight quants
                // never coincide, even for the same block index.
                out[base + 4 + pos] = quant_byte(block, pos + 1000) as u8;
            }
        }
    }
    out
}

/// Builds a Q4_K weight buffer: `n * (k / 256)` super-blocks of 144 bytes —
/// f16 `d` at byte 0, f16 `dmin` at byte 2, 12 bytes of packed 6-bit
/// scales/minimums at byte 4, 128 bytes of nibble-packed quants at byte 16.
fn build_q4_k_weight(n: usize, k: usize) -> Vec<u8> {
    let supers = k / 256;
    let mut out = vec![0u8; n * supers * 144];
    for row in 0..n {
        for s in 0..supers {
            let block = row * supers + s;
            let base = block * 144;
            out[base..base + 2].copy_from_slice(&block_scale(block).to_le_bytes());
            out[base + 2..base + 4].copy_from_slice(
                &half::f16::from_f32(0.02 + (block as f32 * 0.005) % 0.3).to_le_bytes(),
            );
            for i in 0..12 {
                out[base + 4 + i] = (((block * 71 + i * 13) % 256) as i32 - 128) as u8;
            }
            for i in 0..128 {
                out[base + 16 + i] = (((block * 197 + i * 29) % 256) as i32 - 128) as u8;
            }
        }
    }
    out
}

/// Builds a Q6_K weight buffer: `n * (k / 256)` super-blocks of 210 bytes —
/// `ql` at byte 0 (128 bytes), `qh` at byte 128 (64 bytes), 16 signed `sc`
/// bytes at byte 192, f16 `d` at byte 208.
fn build_q6_k_weight(n: usize, k: usize) -> Vec<u8> {
    let supers = k / 256;
    let mut out = vec![0u8; n * supers * 210];
    for row in 0..n {
        for s in 0..supers {
            let block = row * supers + s;
            let base = block * 210;
            for i in 0..128 {
                out[base + i] = (((block * 89 + i * 31) % 256) as i32 - 128) as u8;
            }
            for i in 0..64 {
                out[base + 128 + i] = (((block * 113 + i * 41) % 256) as i32 - 128) as u8;
            }
            for i in 0..16 {
                // Kept small and signed, as a real Q6_K scale is.
                out[base + 192 + i] = (((block * 7 + i * 5) % 63) as i32 - 31) as i8 as u8;
            }
            out[base + 208..base + 210].copy_from_slice(&block_scale(block * 3 + 1).to_le_bytes());
        }
    }
    out
}

#[test]
fn q6_k_mma_kernel_matches_dp4a_kernel() {
    if !numr::runtime::cuda::is_cuda_available() {
        println!(
            "!! q6_k_mma_kernel_matches_dp4a_kernel SKIPPED: CUDA is not available on this \
             machine. NOTHING WAS VERIFIED."
        );
        eprintln!(
            "!! q6_k_mma_kernel_matches_dp4a_kernel SKIPPED: CUDA is not available on this \
             machine. NOTHING WAS VERIFIED."
        );
        return;
    }
    let _lock = cuda_lock();

    // `m16n8k16` needs sm_80. `caps.bf16` marks that floor, so a pre-Ampere
    // device skips here instead of failing to load the module.
    if !CudaDevice::new(0).profile().caps.bf16 {
        println!(
            "!! q6_k_mma_kernel_matches_dp4a_kernel SKIPPED: this GPU predates sm_80, which \
             `mma.sync.aligned.m16n8k16` requires. NOTHING WAS VERIFIED."
        );
        eprintln!(
            "!! q6_k_mma_kernel_matches_dp4a_kernel SKIPPED: this GPU predates sm_80, which \
             `mma.sync.aligned.m16n8k16` requires. NOTHING WAS VERIFIED."
        );
        return;
    }

    // K must be a multiple of 256 for Q6_K's super-block layout.
    let m: usize = 64;
    let k: usize = 512;
    let n: usize = 96;

    let weight_bytes = build_q6_k_weight(n, k);
    let act_bytes = build_q8_1_activation(m, k);

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    client.synchronize();
    let device_index = device.id();

    let weight =
        Tensor::<CudaRuntime>::from_slice(&weight_bytes, &[weight_bytes.len()], &device).unwrap();
    let act = Tensor::<CudaRuntime>::from_slice(&act_bytes, &[act_bytes.len()], &device).unwrap();
    let out_dp4a = Tensor::<CudaRuntime>::from_slice(&vec![0f32; m * n], &[m, n], &device).unwrap();
    let out_mma = Tensor::<CudaRuntime>::from_slice(&vec![0f32; m * n], &[m, n], &device).unwrap();

    let weight_ptr = weight.ptr();
    let act_ptr = act.ptr();
    let out_dp4a_ptr = out_dp4a.ptr();
    let out_mma_ptr = out_mma.ptr();
    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;

    let cfg = LaunchConfig {
        grid_dim: (n_u32.div_ceil(64), m_u32.div_ceil(128), 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let dp4a_module =
        kernels::get_or_load_module(client.context(), device_index, QUANT_GEMV_MODULE).unwrap();
    let dp4a_func = kernels::get_kernel_function(&dp4a_module, "quant_mmq_q6_k_q8_1").unwrap();

    let mma_module =
        kernels::get_or_load_module(client.context(), device_index, QUANT_MMQ_MMA_MODULE).unwrap();
    let mma_func = kernels::get_kernel_function(&mma_module, "quant_mmq_q6_k_q8_1_mma").unwrap();

    unsafe {
        let mut builder = client.stream().launch_builder(&dp4a_func);
        builder.arg(&act_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&out_dp4a_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.arg(&n_u32);
        builder.launch(cfg).unwrap();
    }

    unsafe {
        let mut builder = client.stream().launch_builder(&mma_func);
        builder.arg(&act_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&out_mma_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.arg(&n_u32);
        builder.launch(cfg).unwrap();
    }

    client.synchronize();

    let dp4a_host = out_dp4a.to_vec::<f32>();
    let mma_host = out_mma.to_vec::<f32>();

    for row in 0..m {
        for col in 0..n {
            let idx = row * n + col;
            let a = dp4a_host[idx];
            let b = mma_host[idx];
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "mismatch at (row={row}, col={col}): dp4a={a}, mma={b}. The two kernels must \
                 agree bitwise: they accumulate the same int32 products per 16-element half and \
                 apply the same two scales in the same order."
            );
        }
    }
}

#[test]
fn q4_k_mma_kernel_matches_dp4a_kernel() {
    if !numr::runtime::cuda::is_cuda_available() {
        println!(
            "!! q4_k_mma_kernel_matches_dp4a_kernel SKIPPED: CUDA is not available on this \
             machine. NOTHING WAS VERIFIED."
        );
        eprintln!(
            "!! q4_k_mma_kernel_matches_dp4a_kernel SKIPPED: CUDA is not available on this \
             machine. NOTHING WAS VERIFIED."
        );
        return;
    }
    let _lock = cuda_lock();

    // `m16n8k32` needs sm_80. `caps.bf16` marks that floor, so a pre-Ampere
    // device skips here instead of failing to load the module.
    if !CudaDevice::new(0).profile().caps.bf16 {
        println!(
            "!! q4_k_mma_kernel_matches_dp4a_kernel SKIPPED: this GPU predates sm_80, which \
             `mma.sync.aligned.m16n8k32` requires. NOTHING WAS VERIFIED."
        );
        eprintln!(
            "!! q4_k_mma_kernel_matches_dp4a_kernel SKIPPED: this GPU predates sm_80, which \
             `mma.sync.aligned.m16n8k32` requires. NOTHING WAS VERIFIED."
        );
        return;
    }

    // K must be a multiple of 256 for Q4_K's super-block layout.
    let m: usize = 64;
    let k: usize = 512;
    let n: usize = 96;

    let weight_bytes = build_q4_k_weight(n, k);
    let act_bytes = build_q8_1_activation(m, k);

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    client.synchronize();
    let device_index = device.id();

    let weight =
        Tensor::<CudaRuntime>::from_slice(&weight_bytes, &[weight_bytes.len()], &device).unwrap();
    let act = Tensor::<CudaRuntime>::from_slice(&act_bytes, &[act_bytes.len()], &device).unwrap();
    let out_dp4a = Tensor::<CudaRuntime>::from_slice(&vec![0f32; m * n], &[m, n], &device).unwrap();
    let out_mma = Tensor::<CudaRuntime>::from_slice(&vec![0f32; m * n], &[m, n], &device).unwrap();

    let weight_ptr = weight.ptr();
    let act_ptr = act.ptr();
    let out_dp4a_ptr = out_dp4a.ptr();
    let out_mma_ptr = out_mma.ptr();
    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;

    let cfg = LaunchConfig {
        grid_dim: (n_u32.div_ceil(64), m_u32.div_ceil(128), 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let dp4a_module =
        kernels::get_or_load_module(client.context(), device_index, QUANT_GEMV_MODULE).unwrap();
    let dp4a_func = kernels::get_kernel_function(&dp4a_module, "quant_mmq_q4_k_q8_1").unwrap();

    let mma_module =
        kernels::get_or_load_module(client.context(), device_index, QUANT_MMQ_MMA_MODULE).unwrap();
    let mma_func = kernels::get_kernel_function(&mma_module, "quant_mmq_q4_k_q8_1_mma").unwrap();

    unsafe {
        let mut builder = client.stream().launch_builder(&dp4a_func);
        builder.arg(&act_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&out_dp4a_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.arg(&n_u32);
        builder.launch(cfg).unwrap();
    }

    unsafe {
        let mut builder = client.stream().launch_builder(&mma_func);
        builder.arg(&act_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&out_mma_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.arg(&n_u32);
        builder.launch(cfg).unwrap();
    }

    client.synchronize();

    let dp4a_host = out_dp4a.to_vec::<f32>();
    let mma_host = out_mma.to_vec::<f32>();

    for row in 0..m {
        for col in 0..n {
            let idx = row * n + col;
            let a = dp4a_host[idx];
            let b = mma_host[idx];
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "mismatch at (row={row}, col={col}): dp4a={a}, mma={b}. The two kernels must \
                 agree bitwise: they accumulate the same int32 products and row sums and apply \
                 the same scales in the same order."
            );
        }
    }
}

#[test]
fn mma_kernel_matches_dp4a_kernel() {
    if !numr::runtime::cuda::is_cuda_available() {
        println!(
            "!! mma_kernel_matches_dp4a_kernel SKIPPED: CUDA is not available on this machine. \
             NOTHING WAS VERIFIED."
        );
        eprintln!(
            "!! mma_kernel_matches_dp4a_kernel SKIPPED: CUDA is not available on this machine. \
             NOTHING WAS VERIFIED."
        );
        return;
    }
    let _lock = cuda_lock();

    // `m16n8k32` needs sm_80. `caps.bf16` marks that floor, so a pre-Ampere
    // device skips here instead of failing to load the module.
    if !CudaDevice::new(0).profile().caps.bf16 {
        println!(
            "!! mma_kernel_matches_dp4a_kernel SKIPPED: this GPU predates sm_80, which \
             `mma.sync.aligned.m16n8k32` requires. NOTHING WAS VERIFIED."
        );
        eprintln!(
            "!! mma_kernel_matches_dp4a_kernel SKIPPED: this GPU predates sm_80, which \
             `mma.sync.aligned.m16n8k32` requires. NOTHING WAS VERIFIED."
        );
        return;
    }

    // M, K, N deliberately do NOT divide the 128x64 MMQ tile evenly, so the
    // ragged-edge guards in both kernels are exercised.
    let m: usize = 64;
    let k: usize = 256;
    let n: usize = 96;

    let weight_bytes = build_q8_0_weight(n, k);
    let act_bytes = build_q8_1_activation(m, k);

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    client.synchronize();
    let device_index = device.id();

    let weight =
        Tensor::<CudaRuntime>::from_slice(&weight_bytes, &[weight_bytes.len()], &device).unwrap();
    let act = Tensor::<CudaRuntime>::from_slice(&act_bytes, &[act_bytes.len()], &device).unwrap();
    let out_dp4a = Tensor::<CudaRuntime>::from_slice(&vec![0f32; m * n], &[m, n], &device).unwrap();
    let out_mma = Tensor::<CudaRuntime>::from_slice(&vec![0f32; m * n], &[m, n], &device).unwrap();

    let weight_ptr = weight.ptr();
    let act_ptr = act.ptr();
    let out_dp4a_ptr = out_dp4a.ptr();
    let out_mma_ptr = out_mma.ptr();
    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;

    let cfg = LaunchConfig {
        grid_dim: (n_u32.div_ceil(64), m_u32.div_ceil(128), 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let dp4a_module =
        kernels::get_or_load_module(client.context(), device_index, QUANT_GEMV_MODULE).unwrap();
    let dp4a_func = kernels::get_kernel_function(&dp4a_module, "quant_mmq_q8_0_q8_1").unwrap();

    let mma_module =
        kernels::get_or_load_module(client.context(), device_index, QUANT_MMQ_MMA_MODULE).unwrap();
    let mma_func = kernels::get_kernel_function(&mma_module, "quant_mmq_q8_0_q8_1_mma").unwrap();

    unsafe {
        let mut builder = client.stream().launch_builder(&dp4a_func);
        builder.arg(&act_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&out_dp4a_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.arg(&n_u32);
        builder.launch(cfg).unwrap();
    }

    unsafe {
        let mut builder = client.stream().launch_builder(&mma_func);
        builder.arg(&act_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&out_mma_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.arg(&n_u32);
        builder.launch(cfg).unwrap();
    }

    client.synchronize();

    let dp4a_host = out_dp4a.to_vec::<f32>();
    let mma_host = out_mma.to_vec::<f32>();

    for row in 0..m {
        for col in 0..n {
            let idx = row * n + col;
            let a = dp4a_host[idx];
            let b = mma_host[idx];
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "mismatch at (row={row}, col={col}): dp4a={a}, mma={b}. The two kernels must \
                 agree bitwise: they accumulate the same int8 products in the same int32 order \
                 and apply the same scale once per block."
            );
        }
    }
}

/// Builds a lowbit weight buffer: `n * (k / block_elems)` blocks of
/// `block_bytes`, f16 `d` at byte 0 then `block_bytes - 2` bytes of packed
/// codes. Every bit pattern of the code run is a valid block, so the quant
/// stream is used as raw bytes.
fn build_lowbit_weight(n: usize, k: usize, block_elems: usize, block_bytes: usize) -> Vec<u8> {
    let bpr = k / block_elems;
    let mut out = vec![0u8; n * bpr * block_bytes];
    for row in 0..n {
        for b in 0..bpr {
            let block = row * bpr + b;
            let base = block * block_bytes;
            out[base..base + 2].copy_from_slice(&block_scale(block).to_le_bytes());
            for pos in 0..block_bytes - 2 {
                out[base + 2 + pos] = quant_byte(block, pos) as u8;
            }
        }
    }
    out
}

/// PQ2_0: 34-byte blocks of 128 elements, 32 bytes of 2-bit codes at byte 2.
fn build_pq2_0_weight(n: usize, k: usize) -> Vec<u8> {
    build_lowbit_weight(n, k, 128, 34)
}

/// Q2_0: 18-byte blocks of 64 elements, 16 bytes of 2-bit codes at byte 2.
fn build_q2_0_weight(n: usize, k: usize) -> Vec<u8> {
    build_lowbit_weight(n, k, 64, 18)
}

/// Q1_0: 18-byte blocks of 128 elements, 16 bytes of sign bits at byte 2.
fn build_q1_0_weight(n: usize, k: usize) -> Vec<u8> {
    build_lowbit_weight(n, k, 128, 18)
}

/// The lowbit value maps, on the code run of one block. `code2` is
/// `(code - 1)`, low bits first; `sign` is `+1` for a set bit, `-1` clear.
fn lowbit_code2(qs: &[u8], elem: usize) -> i32 {
    i32::from((qs[elem >> 2] >> ((elem & 3) * 2)) & 0x03) - 1
}

fn lowbit_sign(qs: &[u8], elem: usize) -> i32 {
    if (qs[elem >> 3] >> (elem & 7)) & 1 == 1 {
        1
    } else {
        -1
    }
}

/// The geometry of one lowbit format, as the tests below need it.
struct LowbitCase {
    gemv_kernel: &'static str,
    gemv_module: &'static str,
    mma_kernel: &'static str,
    block_elems: usize,
    block_bytes: usize,
    decode: fn(&[u8], usize) -> i32,
}

/// f64 reference for one output element of `activation x weight^T`, and the
/// sum of the magnitudes of its 32-element block terms. Each block term is
/// exact — an integer dot times two f16 scales — so the magnitude bounds the
/// float error of any accumulation order.
fn lowbit_reference(
    case: &LowbitCase,
    weight: &[u8],
    act: &[u8],
    token: usize,
    feat: usize,
    k: usize,
) -> (f64, f64) {
    let bpr = k / 32;
    let wpr = k / case.block_elems;
    let chunks = case.block_elems / 32;
    let mut sum = 0.0f64;
    let mut magnitude = 0.0f64;
    for b in 0..bpr {
        let wb = (feat * wpr + b / chunks) * case.block_bytes;
        let dw = f64::from(half::f16::from_le_bytes([weight[wb], weight[wb + 1]]).to_f32());
        let qs = &weight[wb + 2..wb + case.block_bytes];
        let ab = (token * bpr + b) * 36;
        let da = f64::from(half::f16::from_le_bytes([act[ab], act[ab + 1]]).to_f32());
        let mut dot = 0i64;
        for pos in 0..32 {
            let wq = i64::from((case.decode)(qs, (b % chunks) * 32 + pos));
            let aq = i64::from(act[ab + 4 + pos] as i8);
            dot += wq * aq;
        }
        let term = dw * da * dot as f64;
        sum += term;
        magnitude += term.abs();
    }
    (sum, magnitude)
}

/// Repacks the per-token Q8_1 buffer into the layout the feature-major
/// kernels read: a 144-byte record per (128-k group, token) holding four
/// header words — f16 `d` in the low half, the int16 quant sum in the high
/// half — then 128 int8, records indexed `kgroup * ntok + token`. Mirrors
/// `quantize_f32_q8_1_mmq` in `src/quant/cuda/kernels/quant_act.cu`; `d` is
/// copied from the per-token header, so both kernels see one activation.
fn repack_q8_1_mmq(act: &[u8], m: usize, k: usize, ntok: usize) -> Vec<u8> {
    let bpr = k / 32;
    let kgroups = bpr.div_ceil(4);
    let mut out = vec![0u8; kgroups * ntok * 144];
    for token in 0..m {
        for b in 0..bpr {
            let src = (token * bpr + b) * 36;
            let rec = ((b / 4) * ntok + token) * 144;
            let sub = b % 4;
            let sum: i32 = act[src + 4..src + 36]
                .iter()
                .map(|&byte| i32::from(byte as i8))
                .sum();
            let slot = rec + sub * 4;
            out[slot..slot + 2].copy_from_slice(&act[src..src + 2]);
            out[slot + 2..slot + 4].copy_from_slice(&(sum as i16).to_le_bytes());
            out[rec + 16 + sub * 32..rec + 16 + sub * 32 + 32]
                .copy_from_slice(&act[src + 4..src + 36]);
        }
    }
    out
}

/// Magnitude-relative bound both lowbit paths are held to, against each other
/// and against the f64 reference.
///
/// The two kernels do NOT share a float order, so a bitwise check is the
/// wrong claim. The GEMV (`legacy_ntok_body.cuh`) gives each lane 8 elements
/// of a 32-element chunk, forms `dw * da * (float)dp4a` per lane, sums those
/// per-lane partials over K in registers, and reduces the lanes by shuffle
/// and the warps through shared memory. The MMA kernel (`mmqf_vec_dot_d`)
/// forms the whole 32-element dot as one exact int32 and adds
/// `(float)D * da * dw` per chunk in K order. Different partial products,
/// different multiplication order, different summation tree: the same
/// value, not the same bits. Every block term is exact before the float
/// rounding, so the error of either order is a few ulps of the magnitude
/// sum; `1e-5` is the bound `examples/mmq_kernel_compare.rs` holds every
/// kernel to against the same reference.
const LOWBIT_RTOL: f64 = 1e-5;

/// Token tile of the feature-major variant launched below. 64 tokens takes
/// the two-half cadence, one token tile, and dynamic shared memory below the
/// opt-in threshold.
const LOWBIT_MMQ_X: u32 = 64;

/// `false` when the lowbit tests cannot run here: no CUDA, or a GPU before
/// sm_80, which `mma.sync.aligned.m16n8k32` requires (`caps.bf16` marks
/// that floor, so a pre-Ampere device skips instead of failing to load the
/// module). Prints the loud skip either way.
fn lowbit_device_ready(name: &str) -> bool {
    if !numr::runtime::cuda::is_cuda_available() {
        println!("!! {name} SKIPPED: CUDA is not available on this machine. NOTHING WAS VERIFIED.");
        eprintln!(
            "!! {name} SKIPPED: CUDA is not available on this machine. NOTHING WAS VERIFIED."
        );
        return false;
    }
    if !CudaDevice::new(0).profile().caps.bf16 {
        println!(
            "!! {name} SKIPPED: this GPU predates sm_80, which `mma.sync.aligned.m16n8k32` \
             requires. NOTHING WAS VERIFIED."
        );
        eprintln!(
            "!! {name} SKIPPED: this GPU predates sm_80, which `mma.sync.aligned.m16n8k32` \
             requires. NOTHING WAS VERIFIED."
        );
        return false;
    }
    true
}

/// Launches the feature-major kernel `mma_kernel` at token tile
/// [`LOWBIT_MMQ_X`] on the repacked activation `packed_ptr`: grid (token
/// tiles, feature tiles), 256 threads, dynamic shared memory of one 76-int
/// weight row per feature plus one 36-int activation record per token
/// (`smem_bytes` in `tiling/`). Does not synchronize.
#[allow(clippy::too_many_arguments)]
fn launch_lowbit_mma(
    client: &CudaClient,
    device_index: usize,
    mma_kernel: &str,
    packed_ptr: u64,
    weight_ptr: u64,
    out_ptr: u64,
    m: usize,
    k: usize,
    n: usize,
) {
    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;
    let ntok = LOWBIT_MMQ_X;
    let mma_module =
        kernels::get_or_load_module(client.context(), device_index, QUANT_MMQ_MMA_MODULE).unwrap();
    let mma_func = kernels::get_kernel_function(&mma_module, mma_kernel).unwrap();
    let smem = 4 * (128 * 76 + LOWBIT_MMQ_X * 36);
    mma_func
        .set_attribute(
            cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem as i32,
        )
        .unwrap();
    let cfg_mma = LaunchConfig {
        grid_dim: (m_u32.div_ceil(LOWBIT_MMQ_X), n_u32.div_ceil(128), 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: smem,
    };
    unsafe {
        let mut builder = client.stream().launch_builder(&mma_func);
        builder.arg(&packed_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&out_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.arg(&n_u32);
        builder.arg(&ntok);
        builder.launch(cfg_mma).unwrap();
    }
}

/// Launches the feature-major MMA kernel and the `_n4` token-batched GEMV on
/// the same Q8_1 activation, checks both against the f64 reference and
/// against each other within [`LOWBIT_RTOL`]. `k` is chosen by the caller so
/// the last 256-k staging group is partial and the ragged tail runs.
fn lowbit_mma_matches_gemv(name: &str, case: &LowbitCase, weight_bytes: &[u8], k: usize) {
    if !lowbit_device_ready(name) {
        return;
    }
    let _lock = cuda_lock();

    // N does not divide the 128-feature tile, so the row clamp runs.
    let m: usize = LOWBIT_MMQ_X as usize;
    let n: usize = 96;
    assert!(
        k.is_multiple_of(case.block_elems),
        "{name}: K must be whole blocks"
    );
    assert!(
        !k.is_multiple_of(256),
        "{name}: K is chosen to leave a ragged 256-k tail"
    );

    let act_bytes = build_q8_1_activation(m, k);
    let packed_bytes = repack_q8_1_mmq(&act_bytes, m, k, m);

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    client.synchronize();
    let device_index = device.id();

    let weight =
        Tensor::<CudaRuntime>::from_slice(weight_bytes, &[weight_bytes.len()], &device).unwrap();
    let act = Tensor::<CudaRuntime>::from_slice(&act_bytes, &[act_bytes.len()], &device).unwrap();
    let packed =
        Tensor::<CudaRuntime>::from_slice(&packed_bytes, &[packed_bytes.len()], &device).unwrap();
    let out_gemv = Tensor::<CudaRuntime>::from_slice(&vec![0f32; m * n], &[m, n], &device).unwrap();
    let out_mma = Tensor::<CudaRuntime>::from_slice(&vec![0f32; m * n], &[m, n], &device).unwrap();

    let weight_ptr = weight.ptr();
    let act_ptr = act.ptr();
    let out_gemv_ptr = out_gemv.ptr();
    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;

    // `_n4`: one output column and four token columns per block, 128 threads
    // (`mwr_nwarps_ntok(4)` warps), as `dispatch_gemv` launches it.
    let gemv_module =
        kernels::get_or_load_module(client.context(), device_index, case.gemv_module).unwrap();
    let gemv_func = kernels::get_kernel_function(&gemv_module, case.gemv_kernel).unwrap();
    let cfg_gemv = LaunchConfig {
        grid_dim: (n_u32, m_u32.div_ceil(4), 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let mut builder = client.stream().launch_builder(&gemv_func);
        builder.arg(&act_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&out_gemv_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.arg(&n_u32);
        builder.launch(cfg_gemv).unwrap();
    }

    launch_lowbit_mma(
        &client,
        device_index,
        case.mma_kernel,
        packed.ptr(),
        weight_ptr,
        out_mma.ptr(),
        m,
        k,
        n,
    );

    client.synchronize();

    let gemv_host = out_gemv.to_vec::<f32>();
    let mma_host = out_mma.to_vec::<f32>();

    let mut worst = 0.0f64;
    for token in 0..m {
        for feat in 0..n {
            let idx = token * n + feat;
            let (want, magnitude) =
                lowbit_reference(case, weight_bytes, &act_bytes, token, feat, k);
            let scale = magnitude.max(f64::MIN_POSITIVE);
            let g = f64::from(gemv_host[idx]);
            let a = f64::from(mma_host[idx]);
            let err_gemv = (g - want).abs() / scale;
            let err_mma = (a - want).abs() / scale;
            let err_pair = (a - g).abs() / scale;
            assert!(
                err_gemv <= LOWBIT_RTOL,
                "{name}: GEMV disagrees with the f64 reference at (token={token}, feat={feat}): \
                 got {g}, want {want}, magnitude-relative error {err_gemv:.3e}"
            );
            assert!(
                err_mma <= LOWBIT_RTOL,
                "{name}: MMA disagrees with the f64 reference at (token={token}, feat={feat}): \
                 got {a}, want {want}, magnitude-relative error {err_mma:.3e}"
            );
            assert!(
                err_pair <= LOWBIT_RTOL,
                "{name}: MMA and GEMV disagree at (token={token}, feat={feat}): mma={a}, \
                 gemv={g}, magnitude-relative error {err_pair:.3e}"
            );
            worst = worst.max(err_pair);
        }
    }
    println!(
        "{name}: {} outputs within {LOWBIT_RTOL:.0e} (worst {worst:.2e})",
        m * n
    );
}

/// K = 640: five 128-element blocks, so two whole 256-k groups and a
/// four-chunk tail.
#[test]
fn pq2_0_mma_kernel_matches_gemv_kernel() {
    let k = 640;
    lowbit_mma_matches_gemv(
        "pq2_0_mma_kernel_matches_gemv_kernel",
        &LowbitCase {
            gemv_kernel: "quant_gemv_pq2_0_q8_1_mwr_n4",
            gemv_module: GEMV_PQ2_0_MODULE,
            mma_kernel: "quant_mmq_pq2_0_q8_1_mma_x64",
            block_elems: 128,
            block_bytes: 34,
            decode: lowbit_code2,
        },
        &build_pq2_0_weight(96, k),
        k,
    );
}

/// K = 576: nine 64-element blocks, so two whole 256-k groups and a
/// two-chunk tail, which only this format's block size can leave.
#[test]
fn q2_0_mma_kernel_matches_gemv_kernel() {
    let k = 576;
    lowbit_mma_matches_gemv(
        "q2_0_mma_kernel_matches_gemv_kernel",
        &LowbitCase {
            gemv_kernel: "quant_gemv_q2_0_q8_1_mwr_n4",
            gemv_module: GEMV_Q2_0_MODULE,
            mma_kernel: "quant_mmq_q2_0_q8_1_mma_x64",
            block_elems: 64,
            block_bytes: 18,
            decode: lowbit_code2,
        },
        &build_q2_0_weight(96, k),
        k,
    );
}

/// K = 640: five 128-element blocks, so two whole 256-k groups and a
/// four-chunk tail.
#[test]
fn q1_0_mma_kernel_matches_gemv_kernel() {
    let k = 640;
    lowbit_mma_matches_gemv(
        "q1_0_mma_kernel_matches_gemv_kernel",
        &LowbitCase {
            gemv_kernel: "quant_gemv_q1_0_q8_1_mwr_n4",
            gemv_module: GEMV_Q1_0_MODULE,
            mma_kernel: "quant_mmq_q1_0_q8_1_mma_x64",
            block_elems: 128,
            block_bytes: 18,
            decode: lowbit_sign,
        },
        &build_q1_0_weight(96, k),
        k,
    );
}

// ── PTQ1_0 ─────────────────────────────────────────────────────────────

/// Packs five trits the way llama.cpp's `quantize_row_ptq1_0_ref` does, and
/// as `pack5` in the test module of `src/quant/cpu/kernels/dequant_lowbit.rs`:
/// base 3 with the FIRST trit most significant, then a ceiling scale by
/// 256/243. `gguf_base3_trit` recovers trit `level` by a wrapping 8-bit
/// multiply with `pow3[level]` followed by `(q * 3) >> 8`, so level 0 is the
/// first, most significant digit and level 4 the last.
fn ptq1_0_pack5(trits: [i32; 5]) -> u8 {
    let q = trits.iter().fold(0u16, |q, &t| q * 3 + (t + 1) as u16);
    (q * 256).div_ceil(243) as u8
}

/// Packs four trits for `qh`: the first trit lands in the most significant
/// position of a five-trit byte, the fifth slot is zero (trit `-1`).
fn ptq1_0_pack4(trits: [i32; 4]) -> u8 {
    ptq1_0_pack5([trits[0], trits[1], trits[2], trits[3], -1])
}

/// Deterministic trit {-1, 0, 1} of element `elem` of block `block`: a
/// hashed LCG step per (block, element), so no two blocks share a pattern
/// and every one of the packer's 243 `qs` byte values and 81 `qh` byte
/// values appears in the weight below (asserted by `build_ptq1_0_weight`).
fn ptq1_0_trit_at(block: usize, elem: usize) -> i32 {
    let mut x = (block as u32)
        .wrapping_mul(2_654_435_761)
        .wrapping_add((elem as u32).wrapping_mul(40_503).wrapping_add(1));
    x ^= x >> 13;
    x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    x ^= x >> 16;
    (x % 3) as i32 - 1
}

/// Builds a PTQ1_0 weight buffer: `n * (k / 128)` blocks of 28 bytes —
/// `qs[0..24]`, `qh[24..26]`, f16 `d` at byte 26, the END. Element to
/// (byte, level) follows `gguf_ptq1_0_trit`: `qs[m]` for `m < 16` holds
/// elements `m + 16 * level`, `qs[16 + m]` for `m < 8` holds
/// `80 + m + 8 * level`, and `qh[h]` holds `120 + h + 2 * level`, level 0
/// most significant in each byte.
///
/// Asserts that the packed `qs` bytes cover all 243 values the packer can
/// emit and the `qh` bytes all 81, so every base-3 byte the kernel can meet
/// is decoded at least once.
fn build_ptq1_0_weight(n: usize, k: usize) -> Vec<u8> {
    let bpr = k / 128;
    let mut out = vec![0u8; n * bpr * 28];
    let mut seen_qs = [false; 256];
    let mut seen_qh = [false; 256];
    for row in 0..n {
        for b in 0..bpr {
            let block = row * bpr + b;
            let base = block * 28;
            let trit = |e: usize| ptq1_0_trit_at(block, e);
            for m in 0..16 {
                let byte = ptq1_0_pack5(std::array::from_fn(|l| trit(m + 16 * l)));
                out[base + m] = byte;
                seen_qs[usize::from(byte)] = true;
            }
            for m in 0..8 {
                let byte = ptq1_0_pack5(std::array::from_fn(|l| trit(80 + m + 8 * l)));
                out[base + 16 + m] = byte;
                seen_qs[usize::from(byte)] = true;
            }
            for h in 0..2 {
                let byte = ptq1_0_pack4(std::array::from_fn(|l| trit(120 + h + 2 * l)));
                out[base + 24 + h] = byte;
                seen_qh[usize::from(byte)] = true;
            }
            out[base + 26..base + 28].copy_from_slice(&block_scale(block).to_le_bytes());
        }
    }
    let qs_values = seen_qs.iter().filter(|&&s| s).count();
    let qh_values = seen_qh.iter().filter(|&&s| s).count();
    assert_eq!(
        qs_values, 243,
        "PTQ1_0 fixture must cover every packed qs byte"
    );
    assert_eq!(
        qh_values, 81,
        "PTQ1_0 fixture must cover every packed qh byte"
    );
    out
}

/// Trit {-1, 0, 1} of element `elem` of a PTQ1_0 block, as `gguf_base3_trit`
/// and `gguf_ptq1_0_trit` read it: the wrapping 8-bit multiply is the same
/// in both languages.
fn ptq1_0_trit(block: &[u8], elem: usize) -> i64 {
    const POW3: [u8; 5] = [1, 3, 9, 27, 81];
    let (byte, level) = if elem < 80 {
        (block[elem % 16], elem / 16)
    } else if elem < 120 {
        let r = elem - 80;
        (block[16 + r % 8], r / 8)
    } else {
        let r = elem - 120;
        (block[24 + r % 2], r / 2)
    };
    let q = byte.wrapping_mul(POW3[level]);
    i64::from((u16::from(q) * 3) >> 8) - 1
}

/// f64 reference for one PTQ1_0 output element over the Q8_1 activation,
/// and the sum of the magnitudes of its 32-element block terms, as
/// [`lowbit_reference`] computes them. Each term is exact — an integer trit
/// dot times two f16 scales — so the magnitude bounds the float error of
/// any accumulation order.
fn ptq1_0_reference(weight: &[u8], act: &[u8], token: usize, feat: usize, k: usize) -> (f64, f64) {
    let bpr = k / 32;
    let wpr = k / 128;
    let mut sum = 0.0f64;
    let mut magnitude = 0.0f64;
    for b in 0..bpr {
        let wb = (feat * wpr + b / 4) * 28;
        let block = &weight[wb..wb + 28];
        let dw = f64::from(half::f16::from_le_bytes([block[26], block[27]]).to_f32());
        let ab = (token * bpr + b) * 36;
        let da = f64::from(half::f16::from_le_bytes([act[ab], act[ab + 1]]).to_f32());
        let mut dot = 0i64;
        for pos in 0..32 {
            let wq = ptq1_0_trit(block, (b % 4) * 32 + pos);
            let aq = i64::from(act[ab + 4 + pos] as i8);
            dot += wq * aq;
        }
        let term = dw * da * dot as f64;
        sum += term;
        magnitude += term.abs();
    }
    (sum, magnitude)
}

/// The Q8_1 activation dequantized back to f32, `d * q` per element. `d` is
/// f16 and `q` an int8, so the product is exact in f32 and the F32 GEMV sees
/// the same activation values the MMA kernel sees through its Q8_1 record.
fn dequant_q8_1_activation(act: &[u8], m: usize, k: usize) -> Vec<f32> {
    let bpr = k / 32;
    let mut out = vec![0f32; m * k];
    for token in 0..m {
        for b in 0..bpr {
            let base = (token * bpr + b) * 36;
            let d = half::f16::from_le_bytes([act[base], act[base + 1]]).to_f32();
            for pos in 0..32 {
                out[token * k + b * 32 + pos] = d * f32::from(act[base + 4 + pos] as i8);
            }
        }
    }
    out
}

/// Launches `quant_mmq_ptq1_0_q8_1_mma_x64` on the repacked Q8_1 activation
/// and `quant_gemv_ptq1_0_f32` on that activation dequantized to f32, and
/// checks both against the f64 reference and against each other within
/// [`LOWBIT_RTOL`].
///
/// PTQ1_0 has no dp4a GEMV, so the F32 GEMV is the reference kernel. The MMA
/// path quantizes its activation to Q8_1 and the F32 GEMV does not, so the
/// two paths are given the SAME activation by dequantizing the Q8_1 record
/// ([`dequant_q8_1_activation`]): every term of the f64 reference is then
/// exact for both, and [`LOWBIT_RTOL`] — the bound the lowbit tests above
/// and `examples/mmq_kernel_compare.rs` already use — holds the F32 GEMV's
/// f32 sum and the MMA kernel's per-chunk f32 folds alike. `k` is chosen
/// so the last 256-k staging group is partial and the ragged tail runs.
fn ptq1_0_mma_matches_f32_gemv(name: &str, weight_bytes: &[u8], k: usize) {
    if !lowbit_device_ready(name) {
        return;
    }
    let _lock = cuda_lock();

    // N does not divide the 128-feature tile, so the row clamp runs.
    let m: usize = LOWBIT_MMQ_X as usize;
    let n: usize = 96;
    assert!(k.is_multiple_of(128), "{name}: K must be whole blocks");
    assert!(
        !k.is_multiple_of(256),
        "{name}: K is chosen to leave a ragged 256-k tail"
    );

    let act_bytes = build_q8_1_activation(m, k);
    let packed_bytes = repack_q8_1_mmq(&act_bytes, m, k, m);
    let act_f32 = dequant_q8_1_activation(&act_bytes, m, k);

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    client.synchronize();
    let device_index = device.id();

    let weight =
        Tensor::<CudaRuntime>::from_slice(weight_bytes, &[weight_bytes.len()], &device).unwrap();
    let act = Tensor::<CudaRuntime>::from_slice(&act_f32, &[m, k], &device).unwrap();
    let packed =
        Tensor::<CudaRuntime>::from_slice(&packed_bytes, &[packed_bytes.len()], &device).unwrap();
    let out_gemv = Tensor::<CudaRuntime>::from_slice(&vec![0f32; m * n], &[m, n], &device).unwrap();
    let out_mma = Tensor::<CudaRuntime>::from_slice(&vec![0f32; m * n], &[m, n], &device).unwrap();

    let weight_ptr = weight.ptr();
    let act_ptr = act.ptr();
    let out_gemv_ptr = out_gemv.ptr();
    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;

    // F32 GEMV: 8 warps per block, one output column per warp, one token
    // per grid row, as `dispatch_gemv`'s F32 branch launches it.
    let gemv_module =
        kernels::get_or_load_module(client.context(), device_index, GEMV_PTQ1_0_MODULE).unwrap();
    let gemv_func = kernels::get_kernel_function(&gemv_module, "quant_gemv_ptq1_0_f32").unwrap();
    let cfg_gemv = LaunchConfig {
        grid_dim: (n_u32.div_ceil(8), m_u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let mut builder = client.stream().launch_builder(&gemv_func);
        builder.arg(&act_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&out_gemv_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.arg(&n_u32);
        builder.launch(cfg_gemv).unwrap();
    }

    launch_lowbit_mma(
        &client,
        device_index,
        "quant_mmq_ptq1_0_q8_1_mma_x64",
        packed.ptr(),
        weight_ptr,
        out_mma.ptr(),
        m,
        k,
        n,
    );

    client.synchronize();

    let gemv_host = out_gemv.to_vec::<f32>();
    let mma_host = out_mma.to_vec::<f32>();

    let mut worst = 0.0f64;
    for token in 0..m {
        for feat in 0..n {
            let idx = token * n + feat;
            let (want, magnitude) = ptq1_0_reference(weight_bytes, &act_bytes, token, feat, k);
            let scale = magnitude.max(f64::MIN_POSITIVE);
            let g = f64::from(gemv_host[idx]);
            let a = f64::from(mma_host[idx]);
            let err_gemv = (g - want).abs() / scale;
            let err_mma = (a - want).abs() / scale;
            let err_pair = (a - g).abs() / scale;
            assert!(
                err_gemv <= LOWBIT_RTOL,
                "{name}: F32 GEMV disagrees with the f64 reference at (token={token}, \
                 feat={feat}): got {g}, want {want}, magnitude-relative error {err_gemv:.3e}"
            );
            assert!(
                err_mma <= LOWBIT_RTOL,
                "{name}: MMA disagrees with the f64 reference at (token={token}, feat={feat}): \
                 got {a}, want {want}, magnitude-relative error {err_mma:.3e}"
            );
            assert!(
                err_pair <= LOWBIT_RTOL,
                "{name}: MMA and F32 GEMV disagree at (token={token}, feat={feat}): mma={a}, \
                 gemv={g}, magnitude-relative error {err_pair:.3e}"
            );
            worst = worst.max(err_pair);
        }
    }
    println!(
        "{name}: {} outputs within {LOWBIT_RTOL:.0e} (worst {worst:.2e})",
        m * n
    );
}

/// K = 640: five 128-element blocks, so two whole 256-k groups and a
/// four-chunk tail. Every lane group of the block — the two `qs` runs at
/// every level and the `qh` tail — is staged in every block.
#[test]
fn ptq1_0_mma_kernel_matches_gemv_kernel() {
    let k = 640;
    ptq1_0_mma_matches_f32_gemv(
        "ptq1_0_mma_kernel_matches_gemv_kernel",
        &build_ptq1_0_weight(96, k),
        k,
    );
}

// ── Single-token kernel vs the tensor-core kernel ───────────────────────

/// K walks the single-token check covers: a five-block walk with a ragged
/// tail, and two deep walks. With `n = 96` (one feature tile) the deep walks
/// cut K into several ranges on a device with more SMs than feature tiles
/// (`split_count`), and 17408 is 68 groups, which no count from 16 down
/// divides, so the ranges are ragged. At `n = 5120` the walk is one range.
const GEMV1_DEPTHS: [usize; 3] = [640, 5120, 17408];

/// Output widths: one partial feature tile, and a wide projection.
const GEMV1_WIDTHS: [usize; 2] = [96, 5120];

/// A K per format whose row byte stride is 2 (mod 4), so a lane's chunk
/// words straddle the staged 32-bit words at a shift the kernel cannot
/// fold: an odd block count of a 34- or 18-byte block. 640 (5 blocks) does
/// this for PQ2_0 and Q1_0 already; Q2_0 needs 9 blocks, Q8_0 21. PTQ1_0's
/// 28-byte block keeps every stride a multiple of 4.
fn gemv1_odd_stride_depth(format: QuantFormat) -> Option<usize> {
    match format {
        QuantFormat::Q2_0 => Some(576),
        QuantFormat::Q8_0 => Some(672),
        _ => None,
    }
}

/// An activation row with a spread of magnitudes, so the per-block scales
/// vary and every chunk's term is a different float.
fn gemv1_activation(k: usize) -> Vec<f32> {
    (0..k)
        .map(|i| {
            let x = i as f32;
            ((x * 0.013).sin() * 0.4 + (x * 0.31).cos() * 0.05) * (1.0 + (i % 7) as f32)
        })
        .collect()
}

/// Runs `[1, k] x weight^T` through the public `quant_matmul`, which takes
/// `quant_mmq_<fmt>_q8_1_gemv1` at one token, and through the tensor-core
/// kernel forced by `Schedule::TileParallel` (the `_x8` variant at one
/// token, or `_ms_x8` when the split count is above one), and asserts the
/// two are the same bits at every output.
fn gemv1_matches_mma(name: &str, format: QuantFormat, weight_bytes: &[u8], n: usize, k: usize) {
    if !lowbit_device_ready(name) {
        return;
    }
    let _lock = cuda_lock();

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let weight = QuantTensor::from_bytes(weight_bytes, format, &[n, k], &device).unwrap();
    let act = Tensor::<CudaRuntime>::from_slice(&gemv1_activation(k), &[1, k], &device).unwrap();

    let mma = quant_matmul_forced_schedule(&client, &act, &weight, Schedule::TileParallel)
        .unwrap()
        .to_vec::<f32>();
    let gemv1 = client.quant_matmul(&act, &weight).unwrap().to_vec::<f32>();

    assert_eq!(mma.len(), n);
    assert_eq!(gemv1.len(), n);
    for f in 0..n {
        assert!(
            mma[f].to_bits() == gemv1[f].to_bits(),
            "{name} N={n} K={k}: feature {f} is {:e} ({:#010x}) on the tensor-core kernel and \
             {:e} ({:#010x}) on the single-token kernel; the two must agree bitwise: same \
             int8 lanes, same scales, same `acc += (float)D * da * dw` per chunk in ascending \
             chunk order, same K ranges summed in range order",
            mma[f],
            mma[f].to_bits(),
            gemv1[f],
            gemv1[f].to_bits()
        );
    }
}

fn gemv1_check_format(name: &str, format: QuantFormat, build: fn(usize, usize) -> Vec<u8>) {
    let depths = GEMV1_DEPTHS
        .iter()
        .copied()
        .chain(gemv1_odd_stride_depth(format));
    for k in depths {
        for &n in &GEMV1_WIDTHS {
            gemv1_matches_mma(name, format, &build(n, k), n, k);
        }
    }
}

#[test]
fn q8_0_gemv1_matches_mma_at_one_token() {
    gemv1_check_format(
        "q8_0_gemv1_matches_mma_at_one_token",
        QuantFormat::Q8_0,
        build_q8_0_weight,
    );
}

#[test]
fn pq2_0_gemv1_matches_mma_at_one_token() {
    gemv1_check_format(
        "pq2_0_gemv1_matches_mma_at_one_token",
        QuantFormat::PQ2_0,
        build_pq2_0_weight,
    );
}

#[test]
fn q2_0_gemv1_matches_mma_at_one_token() {
    gemv1_check_format(
        "q2_0_gemv1_matches_mma_at_one_token",
        QuantFormat::Q2_0,
        build_q2_0_weight,
    );
}

#[test]
fn q1_0_gemv1_matches_mma_at_one_token() {
    gemv1_check_format(
        "q1_0_gemv1_matches_mma_at_one_token",
        QuantFormat::Q1_0,
        build_q1_0_weight,
    );
}

#[test]
fn ptq1_0_gemv1_matches_mma_at_one_token() {
    gemv1_check_format(
        "ptq1_0_gemv1_matches_mma_at_one_token",
        QuantFormat::PTQ1_0,
        build_ptq1_0_weight,
    );
}
