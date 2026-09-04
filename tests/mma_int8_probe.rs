//! Proves the `mma_int8.cuh` fragment index map against a scalar reference.
//!
//! `mma_m16n8k32_s8` computes `D[i][j] = sum over k of A[i][k] * B[j][k]`
//! entirely inside a tensor-core instruction: no intermediate value is
//! observable from Rust. The only way to check the fragment layout is to
//! round-trip through `mma_int8_probe`, which gathers `A`/`B` and scatters
//! `D` using the same index helpers the header exports, and compare the
//! result against a CPU dot product computed straight from the packed bytes.
//!
//! A wrong index in `mma_a_i`/`mma_a_j`/`mma_b_i`/`mma_b_j` reads or writes
//! the wrong lane's data and produces a wrong int32, not a crash — this test
//! is the only thing standing between that defect and every future kernel
//! built on this wrapper.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test mma_int8_probe

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::quant::cuda::kernels::{self, MMA_INT8_PROBE_MODULE};
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

/// Byte `byte` of int `word` in row `row`, chosen so no two bytes in the
/// whole tile are equal and both `word` and `byte` vary.
fn packed_byte(row: usize, word: usize, byte: usize) -> i8 {
    (((row * 7 + word * 3 + byte * 5) % 251) as i32 - 125) as i8
}

/// Packs one 32-bit word from four `i8` values, byte 0 in the low 8 bits.
fn pack_word(row: usize, word: usize) -> i32 {
    let mut w: u32 = 0;
    for byte in 0..4 {
        let v = packed_byte(row, word, byte) as u8;
        w |= (v as u32) << (byte * 8);
    }
    w as i32
}

/// Byte `k % 4` of word `k / 4` of row `i`, i.e. the signed int8 element
/// `a_i8[i][k]` packed into the tile.
fn element(row: usize, k: usize) -> i32 {
    packed_byte(row, k / 4, k % 4) as i32
}

#[test]
fn mma_int8_matches_scalar_reference() {
    if !numr::runtime::cuda::is_cuda_available() {
        println!(
            "!! mma_int8_matches_scalar_reference SKIPPED: CUDA is not available on this \
             machine. NOTHING WAS VERIFIED."
        );
        eprintln!(
            "!! mma_int8_matches_scalar_reference SKIPPED: CUDA is not available on this \
             machine. NOTHING WAS VERIFIED."
        );
        return;
    }
    let _lock = cuda_lock();

    // `m16n8k32` needs sm_80. `caps.bf16` marks that floor, so a pre-Ampere
    // device skips here instead of failing to load the module.
    if !CudaDevice::new(0).profile().caps.bf16 {
        println!(
            "!! mma_int8_matches_scalar_reference SKIPPED: this GPU predates sm_80, which \
             `mma.sync.aligned.m16n8k32` requires. NOTHING WAS VERIFIED."
        );
        eprintln!(
            "!! mma_int8_matches_scalar_reference SKIPPED: this GPU predates sm_80, which \
             `mma.sync.aligned.m16n8k32` requires. NOTHING WAS VERIFIED."
        );
        return;
    }

    let a_host: Vec<i32> = (0..16)
        .flat_map(|row| (0..8).map(move |word| pack_word(row, word)))
        .collect();
    let b_host: Vec<i32> = (0..8)
        .flat_map(|row| (0..8).map(move |word| pack_word(row, word)))
        .collect();

    let mut reference = [[0i32; 8]; 16];
    for i in 0..16 {
        for j in 0..8 {
            let mut sum = 0i32;
            for k in 0..32 {
                sum += element(i, k) * element(j, k);
            }
            reference[i][j] = sum;
        }
    }

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    client.synchronize();

    let device_index = device.id();
    let module =
        kernels::get_or_load_module(client.context(), device_index, MMA_INT8_PROBE_MODULE).unwrap();
    let func = kernels::get_kernel_function(&module, "mma_int8_probe").unwrap();

    let d_host = {
        let a = Tensor::<CudaRuntime>::from_slice(&a_host, &[16, 8], &device).unwrap();
        let b = Tensor::<CudaRuntime>::from_slice(&b_host, &[8, 8], &device).unwrap();
        let d = Tensor::<CudaRuntime>::from_slice(&[0i32; 128], &[16, 8], &device).unwrap();

        let a_ptr = a.ptr();
        let b_ptr = b.ptr();
        let d_ptr = d.ptr();

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let mut builder = client.stream().launch_builder(&func);
            builder.arg(&a_ptr);
            builder.arg(&b_ptr);
            builder.arg(&d_ptr);
            builder.launch(cfg).unwrap();
        }

        d.to_vec::<i32>()
    };
    client.synchronize();

    for i in 0..16 {
        for j in 0..8 {
            let got = d_host[i * 8 + j];
            let want = reference[i][j];
            assert_eq!(
                got, want,
                "mismatch at (i={i}, j={j}): got {got}, want {want}. A wrong value here means \
                 the fragment index map in mma_int8.cuh is wrong, not the arithmetic."
            );
        }
    }
}

#[test]
fn mma_int8_k16_matches_scalar_reference() {
    if !numr::runtime::cuda::is_cuda_available() {
        println!(
            "!! mma_int8_k16_matches_scalar_reference SKIPPED: CUDA is not available on this \
             machine. NOTHING WAS VERIFIED."
        );
        eprintln!(
            "!! mma_int8_k16_matches_scalar_reference SKIPPED: CUDA is not available on this \
             machine. NOTHING WAS VERIFIED."
        );
        return;
    }
    let _lock = cuda_lock();

    // `m16n8k16` needs sm_80. `caps.bf16` marks that floor, so a pre-Ampere
    // device skips here instead of failing to load the module.
    if !CudaDevice::new(0).profile().caps.bf16 {
        println!(
            "!! mma_int8_k16_matches_scalar_reference SKIPPED: this GPU predates sm_80, which \
             `mma.sync.aligned.m16n8k16` requires. NOTHING WAS VERIFIED."
        );
        eprintln!(
            "!! mma_int8_k16_matches_scalar_reference SKIPPED: this GPU predates sm_80, which \
             `mma.sync.aligned.m16n8k16` requires. NOTHING WAS VERIFIED."
        );
        return;
    }

    let a_host: Vec<i32> = (0..16)
        .flat_map(|row| (0..4).map(move |word| pack_word(row, word)))
        .collect();
    let b_host: Vec<i32> = (0..8)
        .flat_map(|row| (0..4).map(move |word| pack_word(row, word)))
        .collect();

    let mut reference = [[0i32; 8]; 16];
    for i in 0..16 {
        for j in 0..8 {
            let mut sum = 0i32;
            for k in 0..16 {
                sum += element(i, k) * element(j, k);
            }
            reference[i][j] = sum;
        }
    }

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    client.synchronize();

    let device_index = device.id();
    let module =
        kernels::get_or_load_module(client.context(), device_index, MMA_INT8_PROBE_MODULE).unwrap();
    let func = kernels::get_kernel_function(&module, "mma_int8_k16_probe").unwrap();

    let d_host = {
        let a = Tensor::<CudaRuntime>::from_slice(&a_host, &[16, 4], &device).unwrap();
        let b = Tensor::<CudaRuntime>::from_slice(&b_host, &[8, 4], &device).unwrap();
        let d = Tensor::<CudaRuntime>::from_slice(&[0i32; 128], &[16, 8], &device).unwrap();

        let a_ptr = a.ptr();
        let b_ptr = b.ptr();
        let d_ptr = d.ptr();

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let mut builder = client.stream().launch_builder(&func);
            builder.arg(&a_ptr);
            builder.arg(&b_ptr);
            builder.arg(&d_ptr);
            builder.launch(cfg).unwrap();
        }

        d.to_vec::<i32>()
    };
    client.synchronize();

    for i in 0..16 {
        for j in 0..8 {
            let got = d_host[i * 8 + j];
            let want = reference[i][j];
            assert_eq!(
                got, want,
                "mismatch at (i={i}, j={j}): got {got}, want {want}. A wrong value here means \
                 the k16 fragment index map in mma_int8.cuh is wrong, not the arithmetic."
            );
        }
    }
}
