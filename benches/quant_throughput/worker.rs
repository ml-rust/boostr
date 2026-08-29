//! Execution of exactly one case, in its own process.
//!
//! Setup — payload load, tensor upload, one warm-up call — happens identically
//! whether `iters` is 0 or `N`. That is what makes the parent's two-run
//! subtraction an exact isolation of the measured loop: lazy CUDA module loads
//! and WebGPU pipeline compilations land in the warm-up, in both runs.

use std::hint::black_box;
use std::time::Instant;

use boostr::quant::QuantTensor;
use boostr::{DType, DequantOps, QuantMatmulOps, Runtime, RuntimeClient, Tensor};

use crate::alloc_counter;
use crate::cases::{Backend, Case, Op};
use crate::payload;

/// What one worker process reports back.
pub struct Sample {
    pub alloc_count: u64,
    pub alloc_bytes: u64,
    /// Minimum elapsed nanoseconds over the measured iterations. Advisory: this
    /// is wall-clock and moves with machine load.
    pub min_ns: u64,
    /// Minimum reference cycles over the measured iterations, 0 where the
    /// architecture exposes no cheap counter. Advisory for the same reason.
    pub min_cycles: u64,
}

impl Sample {
    /// The zero sample, reported when `iters` is 0.
    const fn empty() -> Self {
        Self {
            alloc_count: 0,
            alloc_bytes: 0,
            min_ns: 0,
            min_cycles: 0,
        }
    }
}

/// Run `case` on its own backend. Returns the backend's own message when the
/// backend is compiled out or the device is unavailable.
pub fn run_on_backend(case: &Case, payload: &[u8], iters: usize) -> Result<Sample, String> {
    match case.backend {
        Backend::Cpu => {
            use boostr::{CpuClient, CpuDevice, CpuRuntime};
            let device = CpuDevice::new();
            let client = CpuClient::new(device.clone());
            run::<CpuRuntime, CpuClient>(&client, &device, case, payload, iters)
        }
        #[cfg(feature = "cuda")]
        Backend::Cuda => {
            use boostr::{CudaClient, CudaDevice, CudaRuntime};
            if !numr::runtime::cuda::is_cuda_available() {
                return Err("CUDA runtime unavailable".into());
            }
            let device = CudaDevice::new(0);
            let client = CudaRuntime::default_client(&device);
            run::<CudaRuntime, CudaClient>(&client, &device, case, payload, iters)
        }
        #[cfg(not(feature = "cuda"))]
        Backend::Cuda => Err("built without the cuda feature".into()),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => {
            use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};
            let device = WgpuDevice::new(0);
            let client =
                WgpuClient::new(device.clone()).map_err(|e| format!("no WebGPU adapter: {e:?}"))?;
            run::<WgpuRuntime, WgpuClient>(&client, &device, case, payload, iters)
        }
        #[cfg(not(feature = "wgpu"))]
        Backend::Wgpu => Err("built without the wgpu feature".into()),
    }
}

/// The measured loop, once per backend by monomorphization.
fn run<R, C>(
    client: &C,
    device: &R::Device,
    case: &Case,
    bytes: &[u8],
    iters: usize,
) -> Result<Sample, String>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + DequantOps<R> + QuantMatmulOps<R>,
{
    let (n, k) = (case.n(), case.k());
    let weight = QuantTensor::<R>::from_bytes(bytes, case.scheme(), &[n, k], device)
        .map_err(|e| format!("weight upload: {e}"))?;

    let activation = match case.op {
        Op::Dequant => None,
        Op::Matmul { m } => {
            let values = payload::activation_values(m, k);
            Some(
                Tensor::<R>::from_slice(&values, &[m, k], device)
                    .map_err(|e| format!("activation upload: {e}"))?,
            )
        }
    };

    // One warm-up, always, in both the zero-iteration and the N-iteration run.
    once(client, case, &weight, activation.as_ref())?;
    client.synchronize();

    if iters == 0 {
        return Ok(Sample::empty());
    }

    let (count_before, bytes_before) = alloc_counter::snapshot();
    let mut min_ns = u64::MAX;
    let mut min_cycles = u64::MAX;
    for _ in 0..iters {
        let cycles_start = read_cycles();
        let start = Instant::now();
        let out = once(client, case, &weight, activation.as_ref())?;
        client.synchronize();
        let elapsed = start.elapsed().as_nanos() as u64;
        let cycles = read_cycles().saturating_sub(cycles_start);
        drop(black_box(out));
        min_ns = min_ns.min(elapsed);
        min_cycles = min_cycles.min(cycles);
    }
    let (count_after, bytes_after) = alloc_counter::snapshot();

    Ok(Sample {
        alloc_count: count_after.saturating_sub(count_before),
        alloc_bytes: bytes_after.saturating_sub(bytes_before),
        min_ns,
        min_cycles,
    })
}

/// One invocation of the operation under test.
fn once<R, C>(
    client: &C,
    case: &Case,
    weight: &QuantTensor<R>,
    activation: Option<&Tensor<R>>,
) -> Result<Tensor<R>, String>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + DequantOps<R> + QuantMatmulOps<R>,
{
    match case.op {
        Op::Dequant => client
            .dequantize(black_box(weight), DType::F32)
            .map_err(|e| format!("dequantize: {e}")),
        Op::Matmul { .. } => {
            let activation = activation.ok_or("matmul case built without an activation")?;
            client
                .quant_matmul(black_box(activation), black_box(weight))
                .map_err(|e| format!("quant_matmul: {e}"))
        }
    }
}

/// A cheap monotonic cycle counter, or 0 where the architecture has none.
///
/// `rdtsc` counts REFERENCE cycles, not core cycles, so it does not track
/// frequency scaling. It is reported for continuity with `fluxbench`'s cycle
/// column and is advisory, like the nanosecond column beside it.
#[inline]
fn read_cycles() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `_rdtsc` is unconditionally available on x86_64 and reads a
        // counter register with no memory effects.
        unsafe { core::arch::x86_64::_rdtsc() }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0
    }
}
