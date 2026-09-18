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
//! magnitude-relative tolerance; see [`prism_mma_matches_gemv`].
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test quant_mmq_mma_parity

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::quant::cuda::kernels::{
    self, GEMV_PQ2_0_MODULE, GEMV_Q1_0_MODULE, GEMV_Q2_0_MODULE, QUANT_GEMV_MODULE,
    QUANT_MMQ_MMA_MODULE,
};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaDevice, CudaRuntime};
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

/// Builds a PrismML-fork weight buffer: `n * (k / block_elems)` blocks of
/// `block_bytes`, f16 `d` at byte 0 then `block_bytes - 2` bytes of packed
/// codes. Every bit pattern of the code run is a valid block, so the quant
/// stream is used as raw bytes.
fn build_prism_weight(n: usize, k: usize, block_elems: usize, block_bytes: usize) -> Vec<u8> {
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
    build_prism_weight(n, k, 128, 34)
}

/// Q2_0: 18-byte blocks of 64 elements, 16 bytes of 2-bit codes at byte 2.
fn build_q2_0_weight(n: usize, k: usize) -> Vec<u8> {
    build_prism_weight(n, k, 64, 18)
}

/// Q1_0: 18-byte blocks of 128 elements, 16 bytes of sign bits at byte 2.
fn build_q1_0_weight(n: usize, k: usize) -> Vec<u8> {
    build_prism_weight(n, k, 128, 18)
}

/// The prism value maps, on the code run of one block. `code2` is
/// `(code - 1)`, low bits first; `sign` is `+1` for a set bit, `-1` clear.
fn prism_code2(qs: &[u8], elem: usize) -> i32 {
    i32::from((qs[elem >> 2] >> ((elem & 3) * 2)) & 0x03) - 1
}

fn prism_sign(qs: &[u8], elem: usize) -> i32 {
    if (qs[elem >> 3] >> (elem & 7)) & 1 == 1 {
        1
    } else {
        -1
    }
}

/// The geometry of one prism format, as the tests below need it.
struct PrismCase {
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
fn prism_reference(
    case: &PrismCase,
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

/// Magnitude-relative bound both prism paths are held to, against each other
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
const PRISM_RTOL: f64 = 1e-5;

/// Token tile of the feature-major variant launched below. 64 tokens takes
/// the two-half cadence, one token tile, and dynamic shared memory below the
/// opt-in threshold.
const PRISM_MMQ_X: u32 = 64;

/// Launches the feature-major MMA kernel and the `_n4` token-batched GEMV on
/// the same Q8_1 activation, checks both against the f64 reference and
/// against each other within [`PRISM_RTOL`]. `k` is chosen by the caller so
/// the last 256-k staging group is partial and the ragged tail runs.
fn prism_mma_matches_gemv(name: &str, case: &PrismCase, weight_bytes: &[u8], k: usize) {
    if !numr::runtime::cuda::is_cuda_available() {
        println!("!! {name} SKIPPED: CUDA is not available on this machine. NOTHING WAS VERIFIED.");
        eprintln!(
            "!! {name} SKIPPED: CUDA is not available on this machine. NOTHING WAS VERIFIED."
        );
        return;
    }
    let _lock = cuda_lock();

    // `m16n8k32` needs sm_80. `caps.bf16` marks that floor, so a pre-Ampere
    // device skips here instead of failing to load the module.
    if !CudaDevice::new(0).profile().caps.bf16 {
        println!(
            "!! {name} SKIPPED: this GPU predates sm_80, which `mma.sync.aligned.m16n8k32` \
             requires. NOTHING WAS VERIFIED."
        );
        eprintln!(
            "!! {name} SKIPPED: this GPU predates sm_80, which `mma.sync.aligned.m16n8k32` \
             requires. NOTHING WAS VERIFIED."
        );
        return;
    }

    // N does not divide the 128-feature tile, so the row clamp runs.
    let m: usize = PRISM_MMQ_X as usize;
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
    let packed_ptr = packed.ptr();
    let out_gemv_ptr = out_gemv.ptr();
    let out_mma_ptr = out_mma.ptr();
    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;
    let ntok = PRISM_MMQ_X;

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

    // Feature-major: grid (token tiles, feature tiles), 256 threads, dynamic
    // shared memory of one 76-int weight row per feature plus one 36-int
    // activation record per token (`smem_bytes` in `tiling/`).
    let mma_module =
        kernels::get_or_load_module(client.context(), device_index, QUANT_MMQ_MMA_MODULE).unwrap();
    let mma_func = kernels::get_kernel_function(&mma_module, case.mma_kernel).unwrap();
    let smem = 4 * (128 * 76 + PRISM_MMQ_X * 36);
    mma_func
        .set_attribute(
            cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem as i32,
        )
        .unwrap();
    let cfg_mma = LaunchConfig {
        grid_dim: (m_u32.div_ceil(PRISM_MMQ_X), n_u32.div_ceil(128), 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: smem,
    };
    unsafe {
        let mut builder = client.stream().launch_builder(&mma_func);
        builder.arg(&packed_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&out_mma_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.arg(&n_u32);
        builder.arg(&ntok);
        builder.launch(cfg_mma).unwrap();
    }

    client.synchronize();

    let gemv_host = out_gemv.to_vec::<f32>();
    let mma_host = out_mma.to_vec::<f32>();

    let mut worst = 0.0f64;
    for token in 0..m {
        for feat in 0..n {
            let idx = token * n + feat;
            let (want, magnitude) = prism_reference(case, weight_bytes, &act_bytes, token, feat, k);
            let scale = magnitude.max(f64::MIN_POSITIVE);
            let g = f64::from(gemv_host[idx]);
            let a = f64::from(mma_host[idx]);
            let err_gemv = (g - want).abs() / scale;
            let err_mma = (a - want).abs() / scale;
            let err_pair = (a - g).abs() / scale;
            assert!(
                err_gemv <= PRISM_RTOL,
                "{name}: GEMV disagrees with the f64 reference at (token={token}, feat={feat}): \
                 got {g}, want {want}, magnitude-relative error {err_gemv:.3e}"
            );
            assert!(
                err_mma <= PRISM_RTOL,
                "{name}: MMA disagrees with the f64 reference at (token={token}, feat={feat}): \
                 got {a}, want {want}, magnitude-relative error {err_mma:.3e}"
            );
            assert!(
                err_pair <= PRISM_RTOL,
                "{name}: MMA and GEMV disagree at (token={token}, feat={feat}): mma={a}, \
                 gemv={g}, magnitude-relative error {err_pair:.3e}"
            );
            worst = worst.max(err_pair);
        }
    }
    println!(
        "{name}: {} outputs within {PRISM_RTOL:.0e} (worst {worst:.2e})",
        m * n
    );
}

/// K = 640: five 128-element blocks, so two whole 256-k groups and a
/// four-chunk tail.
#[test]
fn pq2_0_mma_kernel_matches_gemv_kernel() {
    let k = 640;
    prism_mma_matches_gemv(
        "pq2_0_mma_kernel_matches_gemv_kernel",
        &PrismCase {
            gemv_kernel: "quant_gemv_pq2_0_q8_1_mwr_n4",
            gemv_module: GEMV_PQ2_0_MODULE,
            mma_kernel: "quant_mmq_pq2_0_q8_1_mma_x64",
            block_elems: 128,
            block_bytes: 34,
            decode: prism_code2,
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
    prism_mma_matches_gemv(
        "q2_0_mma_kernel_matches_gemv_kernel",
        &PrismCase {
            gemv_kernel: "quant_gemv_q2_0_q8_1_mwr_n4",
            gemv_module: GEMV_Q2_0_MODULE,
            mma_kernel: "quant_mmq_q2_0_q8_1_mma_x64",
            block_elems: 64,
            block_bytes: 18,
            decode: prism_code2,
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
    prism_mma_matches_gemv(
        "q1_0_mma_kernel_matches_gemv_kernel",
        &PrismCase {
            gemv_kernel: "quant_gemv_q1_0_q8_1_mwr_n4",
            gemv_module: GEMV_Q1_0_MODULE,
            mma_kernel: "quant_mmq_q1_0_q8_1_mma_x64",
            block_elems: 128,
            block_bytes: 18,
            decode: prism_sign,
        },
        &build_q1_0_weight(96, k),
        k,
    );
}
