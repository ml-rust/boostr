//! Launches the dp4a and tensor-core Q8_0 MMQ kernels back to back in one
//! process, on identical inputs, so a profiler attributes instruction counts
//! to each kernel without an A/B rebuild.
//!
//! ```text
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --n 4096 --k 14336 --m 512
//! ```

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("mmq_kernel_compare needs --features cuda");
}

#[cfg(feature = "cuda")]
use boostr::quant::cuda::kernels::{self, QUANT_GEMV_MODULE, QUANT_MMQ_MMA_MODULE};
#[cfg(feature = "cuda")]
use cudarc::driver::PushKernelArg;
#[cfg(feature = "cuda")]
use cudarc::driver::safe::LaunchConfig;
#[cfg(feature = "cuda")]
use numr::runtime::Device;
#[cfg(feature = "cuda")]
use numr::runtime::cuda::{CudaDevice, CudaRuntime};
#[cfg(feature = "cuda")]
use numr::runtime::{Runtime, RuntimeClient};
#[cfg(feature = "cuda")]
use numr::tensor::Tensor;

/// Calls timed per kernel, after warmup.
#[cfg(feature = "cuda")]
const ITERS: usize = 100;
/// Calls made before timing starts, to cover module load and any autotune.
#[cfg(feature = "cuda")]
const WARMUP: usize = 20;

/// Deterministic pseudo-random `i8` quant, varied by block index and
/// position so no permutation of the payload coincides with another.
#[cfg(feature = "cuda")]
fn quant_byte(block: usize, pos: usize) -> i8 {
    (((block * 131 + pos * 17) % 251) as i32 - 125) as i8
}

/// Plausible per-block f16 scale, varied by block index.
#[cfg(feature = "cuda")]
fn block_scale(block: usize) -> half::f16 {
    half::f16::from_f32(0.01 + (block as f32 * 0.003) % 0.5)
}

/// Builds a Q8_0 weight buffer: `n * (k / 32)` blocks of 34 bytes, half scale
/// at byte 0, 32 `i8` quants at byte 2.
#[cfg(feature = "cuda")]
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
#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
fn main() {
    if !numr::runtime::cuda::is_cuda_available() {
        println!("mmq_kernel_compare SKIPPED: CUDA is not available on this machine.");
        return;
    }

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (mut n, mut k, mut m) = (4096usize, 14336usize, 512usize);

    let mut i = 0;
    while i < argv.len() {
        let value = || argv.get(i + 1).expect("flag needs a value").clone();
        match argv[i].as_str() {
            "--n" => n = value().parse().expect("--n must be a usize"),
            "--k" => k = value().parse().expect("--k must be a usize"),
            "--m" => m = value().parse().expect("--m must be a usize"),
            other => panic!("unknown flag {other}, expected --n, --k, or --m"),
        }
        i += 2;
    }

    if k % 32 != 0 {
        eprintln!("--k must be a multiple of 32, got {k}");
        std::process::exit(1);
    }

    let device = CudaDevice::new(0);

    // `m16n8k32` needs sm_80. `caps.bf16` marks that floor, so a pre-Ampere
    // device skips here instead of failing to load the module.
    if !device.profile().caps.bf16 {
        println!(
            "mmq_kernel_compare SKIPPED: this GPU predates sm_80, which \
             `mma.sync.aligned.m16n8k32` requires."
        );
        return;
    }

    let client = CudaRuntime::default_client(&device);
    client.synchronize();
    let device_index = device.id();

    let weight_bytes = build_q8_0_weight(n, k);
    let act_bytes = build_q8_1_activation(m, k);

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
        kernels::get_or_load_module(client.context(), device_index, QUANT_GEMV_MODULE)
            .expect("load dp4a module");
    let dp4a_func = kernels::get_kernel_function(&dp4a_module, "quant_mmq_q8_0_q8_1")
        .expect("resolve quant_mmq_q8_0_q8_1");

    let mma_module =
        kernels::get_or_load_module(client.context(), device_index, QUANT_MMQ_MMA_MODULE)
            .expect("load mma module");
    let mma_func = kernels::get_kernel_function(&mma_module, "quant_mmq_q8_0_q8_1_mma")
        .expect("resolve quant_mmq_q8_0_q8_1_mma");

    let launch_dp4a = || unsafe {
        let mut builder = client.stream().launch_builder(&dp4a_func);
        builder.arg(&act_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&out_dp4a_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.arg(&n_u32);
        builder.launch(cfg).expect("launch dp4a kernel");
    };
    let launch_mma = || unsafe {
        let mut builder = client.stream().launch_builder(&mma_func);
        builder.arg(&act_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&out_mma_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.arg(&n_u32);
        builder.launch(cfg).expect("launch mma kernel");
    };

    for _ in 0..WARMUP {
        launch_dp4a();
    }
    client.synchronize();
    let started = std::time::Instant::now();
    for _ in 0..ITERS {
        launch_dp4a();
    }
    client.synchronize();
    let dp4a_us = started.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    for _ in 0..WARMUP {
        launch_mma();
    }
    client.synchronize();
    let started = std::time::Instant::now();
    for _ in 0..ITERS {
        launch_mma();
    }
    client.synchronize();
    let mma_us = started.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

    let dp4a_host = out_dp4a.to_vec::<f32>();
    let mma_host = out_mma.to_vec::<f32>();

    let mut mismatch = None;
    'outer: for row in 0..m {
        for col in 0..n {
            let idx = row * n + col;
            if dp4a_host[idx].to_bits() != mma_host[idx].to_bits() {
                mismatch = Some((idx, dp4a_host[idx], mma_host[idx]));
                break 'outer;
            }
        }
    }

    match mismatch {
        None => println!(
            "outputs match: dp4a and mma agree bit-for-bit at all {} elements",
            m * n
        ),
        Some((idx, a, b)) => {
            eprintln!("outputs MISMATCH at index {idx}: dp4a={a}, mma={b}");
            std::process::exit(1);
        }
    }

    println!("quant_mmq_q8_0_q8_1 (dp4a) {dp4a_us:9.2} us/call");
    println!("quant_mmq_q8_0_q8_1_mma    {mma_us:9.2} us/call");
    println!("ratio dp4a/mma: {:.3}", dp4a_us / mma_us);
}
