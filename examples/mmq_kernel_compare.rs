//! Launches the dp4a and tensor-core MMQ kernels back to back in one process,
//! on identical inputs, so a profiler attributes instruction counts to each
//! kernel without an A/B rebuild. Covers Q8_0, Q4_K and Q6_K via `--format`.
//!
//! ```text
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format q8_0 --n 4096 --k 14336 --m 512
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format q4_k --n 4096 --k 14336 --m 512
//! cargo run --release --features cuda --example mmq_kernel_compare -- \
//!     --format q6_k --n 4096 --k 14336 --m 512
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

/// The MMQ formats this tool can compare, selected by `--format`.
///
/// Kernel names and module constants come from
/// `src/quant/cuda/quant_matmul/format_dispatch.rs`'s `dispatch_matmul`,
/// which is the authoritative dispatch this tool mirrors.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
enum MmqFormat {
    Q8_0,
    Q4K,
    Q6K,
}

#[cfg(feature = "cuda")]
impl MmqFormat {
    fn parse(s: &str) -> Self {
        match s {
            "q8_0" => MmqFormat::Q8_0,
            "q4_k" => MmqFormat::Q4K,
            "q6_k" => MmqFormat::Q6K,
            other => panic!("unknown --format {other}, expected one of: q8_0, q4_k, q6_k"),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            MmqFormat::Q8_0 => "q8_0",
            MmqFormat::Q4K => "q4_k",
            MmqFormat::Q6K => "q6_k",
        }
    }

    /// Q8_0 blocks are 32 elements; Q4_K and Q6_K are 256-element
    /// super-blocks, so `k` must divide evenly by that instead. This mirrors
    /// the `k.is_multiple_of(...)` guards in `dispatch_matmul`.
    fn k_multiple(&self) -> usize {
        match self {
            MmqFormat::Q8_0 => 32,
            MmqFormat::Q4K | MmqFormat::Q6K => 256,
        }
    }

    fn dp4a_kernel(&self) -> &'static str {
        match self {
            MmqFormat::Q8_0 => "quant_mmq_q8_0_q8_1",
            MmqFormat::Q4K => "quant_mmq_q4_k_q8_1",
            MmqFormat::Q6K => "quant_mmq_q6_k_q8_1",
        }
    }

    fn mma_kernel(&self) -> &'static str {
        match self {
            MmqFormat::Q8_0 => "quant_mmq_q8_0_q8_1_mma",
            MmqFormat::Q4K => "quant_mmq_q4_k_q8_1_mma",
            MmqFormat::Q6K => "quant_mmq_q6_k_q8_1_mma",
        }
    }

    fn build_weight(&self, n: usize, k: usize) -> Vec<u8> {
        match self {
            MmqFormat::Q8_0 => build_q8_0_weight(n, k),
            MmqFormat::Q4K => build_q4_k_weight(n, k),
            MmqFormat::Q6K => build_q6_k_weight(n, k),
        }
    }
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

/// Builds a Q4_K weight buffer: `n * (k / 256)` super-blocks of 144 bytes.
///
/// Layout (authoritative source: `src/quant/cpu/kernels/quantize/q4k_q5k.rs`
/// doc comment on `quantize_q4k`, and the byte offsets its `fit_super_block`
/// writes at lines 195-215): f16 `d`@0, f16 `dmin`@2, a 12-byte packed 6-bit
/// scale/min array@4, 128-byte nibble-packed quants@16.
///
/// The 12-byte scale/min packing below is copied from `fit_super_block`
/// (lines 196-211 of that file), which is the inverse of the unpacker
/// `q4k_scale_min` in `src/quant/cuda/kernels/decode.cuh` (lines 97-107) that
/// both the dp4a and MMA MMQ kernels call.
///
/// The nibble layout for sub-block `j` (0..8) is copied from
/// `quant_mmq_q4_k_q8_1` in `src/quant/cuda/kernels/quant_gemv.cu`
/// (`blk + 16 + (j / 2) * 32`, low nibble for even `j`, high for odd) and
/// from the identical byte indexing `mmq_q4_k_stage_load` uses in
/// `src/quant/cuda/kernels/quant_mmq_mma.cu`.
#[cfg(feature = "cuda")]
fn build_q4_k_weight(n: usize, k: usize) -> Vec<u8> {
    const SUPER: usize = 256;
    const BYTES: usize = 144;
    let bpr = k / SUPER;
    let mut out = vec![0u8; n * bpr * BYTES];
    for row in 0..n {
        for sup in 0..bpr {
            let block = row * bpr + sup;
            let base = block * BYTES;
            out[base..base + 2].copy_from_slice(&block_scale(block).to_le_bytes());
            out[base + 2..base + 4].copy_from_slice(&block_scale(block * 7 + 1).to_le_bytes());

            // Eight 6-bit scales and eight 6-bit minimums, one pair per
            // 32-element sub-block.
            let mut scale = [0u8; 8];
            let mut min = [0u8; 8];
            for j in 0..8 {
                scale[j] = ((block * 11 + j * 5) % 64) as u8;
                min[j] = ((block * 13 + j * 3 + 1) % 64) as u8;
            }
            // Pack per `fit_super_block`: low four pairs are plain 6-bit
            // values in sc[0..4]/sc[4..8]; high four pairs split their low
            // nibble into sc[8..12] and their top two bits into the spare
            // high bits of sc[0..4] (scales) and sc[4..8] (minimums).
            let mut sc = [0u8; 12];
            for idx in 0..4 {
                sc[idx] = scale[idx] & 0x3F;
                sc[4 + idx] = min[idx] & 0x3F;
            }
            for idx in 0..4 {
                let j = 4 + idx;
                sc[8 + idx] = (scale[j] & 0x0F) | ((min[j] & 0x0F) << 4);
                sc[idx] |= (scale[j] >> 4) << 6;
                sc[4 + idx] |= (min[j] >> 4) << 6;
            }
            out[base + 4..base + 16].copy_from_slice(&sc);

            for j in 0..8 {
                let group = base + 16 + (j / 2) * 32;
                for pos in 0..32 {
                    let nib = quant_byte(block * 8 + j, pos) as u8 & 0x0F;
                    let byte_idx = group + pos;
                    out[byte_idx] = if j % 2 == 0 {
                        (out[byte_idx] & 0xF0) | nib
                    } else {
                        (out[byte_idx] & 0x0F) | (nib << 4)
                    };
                }
            }
        }
    }
    out
}

/// Builds a Q6_K weight buffer: `n * (k / 256)` super-blocks of 210 bytes.
///
/// Layout (authoritative source: `src/quant/cpu/kernels/quantize/q6k.rs`
/// module doc, "Field order — the trap", and its `quantize_q6k_with` byte
/// offsets at lines 70-73): 128-byte `ql` (low nibbles)@0, 64-byte `qh`
/// (high bit pairs)@128, 16 signed 8-bit sub-block scales@192, f16 `d`@208.
/// `d` comes LAST, not first — this is the one GGML block whose scale isn't
/// at byte 0, per that file's warning.
///
/// The `ql`/`qh` bit-packing below is copied verbatim from that file's
/// `pack_q6k` function (lines 99-114), which both the dp4a
/// (`quant_mmq_q6_k_q8_1` in `quant_gemv.cu`) and MMA
/// (`mmq_q6_k_stage_load` in `quant_mmq_mma.cu`) kernels unpack identically.
#[cfg(feature = "cuda")]
fn build_q6_k_weight(n: usize, k: usize) -> Vec<u8> {
    const SUPER: usize = 256;
    const BYTES: usize = 210;
    let bpr = k / SUPER;
    let mut out = vec![0u8; n * bpr * BYTES];
    for row in 0..n {
        for sup in 0..bpr {
            let block = row * bpr + sup;
            let base = block * BYTES;

            // 256 biased 6-bit levels (0..63, i.e. a signed [-32, 31] value
            // biased by +32, matching the format's `d * scale * (q - 32)`).
            let mut levels = [0u8; SUPER];
            for (pos, level) in levels.iter_mut().enumerate() {
                *level = ((quant_byte(block, pos) as i32 + 128) % 64) as u8;
            }
            // 16 signed 8-bit sub-block scales, one per 16 elements.
            for ib in 0..16 {
                let scale = (((block * 17 + ib * 9) % 121) as i32 - 60) as i8;
                out[base + 192 + ib] = scale as u8;
            }
            out[base + 208..base + 210].copy_from_slice(&block_scale(block).to_le_bytes());

            for half in 0..2 {
                let hbase = half * 128;
                let ql = base + half * 64;
                let qh = base + 128 + half * 32;
                for l in 0..32 {
                    let q1 = levels[hbase + l];
                    let q2 = levels[hbase + l + 32];
                    let q3 = levels[hbase + l + 64];
                    let q4 = levels[hbase + l + 96];
                    out[ql + l] = (q1 & 0x0F) | ((q3 & 0x0F) << 4);
                    out[ql + l + 32] = (q2 & 0x0F) | ((q4 & 0x0F) << 4);
                    out[qh + l] =
                        (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
                }
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
    let mut format = MmqFormat::Q8_0;

    let mut i = 0;
    while i < argv.len() {
        let value = || argv.get(i + 1).expect("flag needs a value").clone();
        match argv[i].as_str() {
            "--n" => n = value().parse().expect("--n must be a usize"),
            "--k" => k = value().parse().expect("--k must be a usize"),
            "--m" => m = value().parse().expect("--m must be a usize"),
            "--format" => format = MmqFormat::parse(&value()),
            other => panic!("unknown flag {other}, expected --n, --k, --m, or --format"),
        }
        i += 2;
    }

    let k_multiple = format.k_multiple();
    if k % k_multiple != 0 {
        eprintln!(
            "--k must be a multiple of {k_multiple} for --format {}, got {k}",
            format.label()
        );
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

    println!(
        "mmq_kernel_compare: format={} n={n} k={k} m={m}",
        format.label()
    );

    let client = CudaRuntime::default_client(&device);
    client.synchronize();
    let device_index = device.id();

    let weight_bytes = format.build_weight(n, k);
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

    let dp4a_kernel = format.dp4a_kernel();
    let mma_kernel = format.mma_kernel();

    let dp4a_module =
        kernels::get_or_load_module(client.context(), device_index, QUANT_GEMV_MODULE)
            .expect("load dp4a module");
    let dp4a_func = kernels::get_kernel_function(&dp4a_module, dp4a_kernel)
        .unwrap_or_else(|_| panic!("resolve {dp4a_kernel}"));

    let mma_module =
        kernels::get_or_load_module(client.context(), device_index, QUANT_MMQ_MMA_MODULE)
            .expect("load mma module");
    let mma_func = kernels::get_kernel_function(&mma_module, mma_kernel)
        .unwrap_or_else(|_| panic!("resolve {mma_kernel}"));

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

    println!("{dp4a_kernel} (dp4a) {dp4a_us:9.2} us/call");
    println!("{mma_kernel}    {mma_us:9.2} us/call");
    println!("ratio dp4a/mma: {:.3}", dp4a_us / mma_us);
}
