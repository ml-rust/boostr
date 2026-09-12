//! Time `quant_matmul` at explicit `[N, K]` x M shapes, for cross-runtime
//! comparison against another implementation's own numbers.
//!
//! ```text
//! cargo run --release --features cuda --example quant_shape_bench -- \
//!     --format q8_0 --n 4096 --k 14336 --m 1,2,4,8,512
//! ```
//!
//! `--format` takes a GGUF block name (`q8_0`, `q6_k`, `q4_k`), so one
//! invocation shape can be compared against an external runtime.
//!
//! Reports microseconds per call, the same unit `test-backend-ops perf` prints,
//! so the two can be read side by side at MATCHED shapes. A comparison at
//! different shapes measures the shapes, not the kernels.
//!
//! The weight payload is deterministic pseudo-random bytes rather than a real
//! tensor: a matmul's cost does not depend on the VALUES, only on the layout,
//! and this keeps the benchmark free of any checkpoint.

/// Without the CUDA backend there is no client to time against, so the example
/// reports that rather than failing to build.
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("quant_shape_bench needs --features cuda");
}

#[cfg(feature = "cuda")]
use boostr::QuantMatmulOps;
#[cfg(feature = "cuda")]
use boostr::quant::{QuantFormat, QuantTensor};
#[cfg(feature = "cuda")]
use numr::runtime::RuntimeClient;
#[cfg(feature = "cuda")]
use numr::runtime::cuda::{CudaClient, CudaDevice};
#[cfg(feature = "cuda")]
use numr::tensor::Tensor;

/// Calls timed per measurement, after warmup.
#[cfg(feature = "cuda")]
const ITERS: usize = 100;
/// Calls made before timing starts, to cover module load and any autotune.
#[cfg(feature = "cuda")]
const WARMUP: usize = 20;

#[cfg(feature = "cuda")]
fn parse_format(name: &str) -> QuantFormat {
    match name {
        "q8_0" => QuantFormat::Q8_0,
        "q6_k" => QuantFormat::Q6K,
        "q4_k" => QuantFormat::Q4K,
        other => panic!("unknown --format {other}, expected q8_0, q6_k, or q4_k"),
    }
}

/// Packed weight bytes for `format` at `[n, k]`, through the CPU quantizer,
/// so the kernel is measured on the bytes the real encoder produces.
#[cfg(feature = "cuda")]
fn packed_weight(format: QuantFormat, n: usize, k: usize) -> Result<Vec<u8>, String> {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    // Deterministic source values; a matmul's cost is set by the layout, not
    // the bits, but a real encode keeps the block statistics realistic.
    let values: Vec<f32> = (0..n * k)
        .map(|i| ((i % 977) as f32 * 0.031).sin())
        .collect();
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let input = Tensor::<CpuRuntime>::from_slice(&values, &[n, k], &device)
        .map_err(|e| format!("weight tensor: {e}"))?;
    boostr::quant::QuantizeOps::quantize(&client, &input, format)
        .map_err(|e| format!("quantize {}: {e}", format.name()))?
        .to_bytes()
        .map_err(|e| format!("read back {}: {e}", format.name()))
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut format = QuantFormat::Q8_0;
    let (mut n, mut k) = (4096usize, 14336usize);
    let mut ms = vec![1usize, 2, 4, 8, 512];

    let mut i = 0;
    while i < argv.len() {
        let value = || argv.get(i + 1).expect("flag needs a value").clone();
        match argv[i].as_str() {
            "--format" => format = parse_format(&value()),
            "--n" => n = value().parse()?,
            "--k" => k = value().parse()?,
            "--m" => ms = value().split(',').map(|s| s.parse().unwrap()).collect(),
            other => panic!("unknown flag {other}"),
        }
        i += 2;
    }

    let device = CudaDevice::new(0);
    let client = CudaClient::new(device.clone())?;

    let weight_bytes = packed_weight(format, n, k)?;
    let weight = QuantTensor::from_bytes(&weight_bytes, format, &[n, k], &device)?;

    println!("{} N={n} K={k}", format.name());
    for &m in &ms {
        let act_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).sin() * 0.4).collect();
        let act = Tensor::from_slice(&act_data, &[m, k], &device)?;

        for _ in 0..WARMUP {
            let _ = client.quant_matmul(&act, &weight)?;
        }
        client.synchronize();

        let started = std::time::Instant::now();
        for _ in 0..ITERS {
            let _ = client.quant_matmul(&act, &weight)?;
        }
        client.synchronize();
        let per_call = started.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
        println!("  M={m:<4} {per_call:9.2} us");
    }
    Ok(())
}
