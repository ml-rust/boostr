//! Proves `quant_mmq_q8_0_q8_1_mma` is bit-identical to `quant_mmq_q8_0_q8_1`.
//!
//! Both kernels stage the same Q8_0/Q8_1 bytes into shared memory, accumulate
//! the same int8 products in int32, and apply the same two f16 scales once
//! per 32-element block in the same block order. Nothing about that changes
//! between dp4a and `mma.sync.aligned.m16n8k32` — only the instruction that
//! does the multiply-accumulate. So the outputs must match down to the bit,
//! not just within a float tolerance.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test quant_mmq_mma_parity

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::quant::cuda::kernels::{self, QUANT_GEMV_MODULE, QUANT_MMQ_MMA_MODULE};
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
