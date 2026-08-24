//! On-demand GPU benchmark: is the dedicated MQA/GQA CUDA kernel
//! (`boostr::ops::cuda::attention::mqa_gqa::{mqa_gqa_fwd, mqa_gqa_bwd}`) faster
//! than the general `FlashAttentionOps::flash_attention_fwd/bwd` path, at the
//! shapes the kernel's own selection heuristic (`should_use_mqa_gqa`) targets?
//!
//! This is NOT a correctness test — numerical parity against the CPU
//! reference is already covered by `tests/backend_parity/mqa_gqa_attention.rs`.
//! This file only measures relative speed, so it is `#[ignore]`d: it never
//! runs under plain `cargo test` or `cargo test --features cuda`, only when
//! explicitly requested.
//!
//! Run:
//!   cargo test --release --features cuda --test mqa_gqa_vs_flash_bench \
//!       -- --ignored --nocapture
//!
//! # Timing method
//!
//! `numr`'s `CudaClient::record_event_on_compute` creates its events with
//! `CU_EVENT_DISABLE_TIMING` (see `numr/src/runtime/cuda/client.rs`) — it
//! exists only to order the copy stream against the compute stream, not to
//! measure elapsed device time, and no other CUDA-event timing API is exposed
//! through `boostr`/`numr`. So this benchmark falls back to HOST wall-clock:
//! `Instant::now()` around each call, bracketed by `client.synchronize()`
//! calls so the timed window covers actual device execution, not just kernel
//! launch/enqueue latency.
//!
//! These are therefore wall-clock figures, trustworthy only on an otherwise
//! quiet machine (no other process contending for the GPU or scheduling the
//! host thread). Report MEDIAN and MIN over an ADAPTIVE number of timed
//! iterations (see `BUDGET_PER_MEASUREMENT`) after `WARMUP` untimed ones
//! (module load / JIT), per the project's standing guidance to prefer
//! median/min over mean so a single scheduling hiccup does not dominate the
//! number.
//!
//! # Fairness notes
//!
//! - Both APIs allocate their own output tensors on every call (neither
//!   accepts a caller-provided output buffer), so allocation cost is included
//!   in both measurements identically.
//! - `flash_attention_fwd`/`_bwd` take two extra parameters that
//!   `mqa_gqa_fwd`/`_bwd` do not: `window_size` and `kv_seq_len` (fwd only).
//!   Both are passed as "off" (`0` and `None`) here, which makes flash_v2
//!   iterate over the full, un-windowed K/V range — the same range the
//!   dedicated kernel always uses. This is an apples-to-apples comparison,
//!   not a handicap in either direction.
//! - Dtype is F32 for both paths. The dedicated kernel and flash_v2 both also
//!   support F16/BF16; this benchmark does not sweep dtype, since the
//!   question asked is about the two CUDA kernels' relative cost at a fixed
//!   dtype, not about dtype's own effect on either.

#![cfg(feature = "cuda")]

use std::time::{Duration, Instant};

use numr::runtime::Runtime;
use numr::runtime::RuntimeClient;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

use boostr::ops::cuda::attention::mqa_gqa::{mqa_gqa_bwd, mqa_gqa_fwd, should_use_mqa_gqa};
use boostr::ops::traits::attention::flash::FlashAttentionOps;

/// One untimed call before measuring, to cover module load and JIT.
const WARMUP: usize = 1;

/// Lower and upper bounds on timed iterations per measurement.
const MIN_ITERS: usize = 3;
const MAX_ITERS: usize = 32;

/// Wall-clock budget per measurement. Iteration count adapts to fit it.
///
/// A FIXED iteration count is wrong here: attention backward is O(batch * seq^2),
/// so across this sweep's shapes the per-call cost spans roughly three orders of
/// magnitude (~120 ms at `b=1, seq=512` to ~60 s at `b=8, seq=4096`). A count
/// tuned for the small end runs for hours at the large end. Budgeting time
/// instead gives many samples where they are cheap and few where they are not,
/// which is where the extra samples were buying nothing anyway.
const BUDGET_PER_MEASUREMENT: Duration = Duration::from_secs(4);

/// Upper bound on a shape's cost proxy `batch * seq^2 * head_dim`.
///
/// Calibrated against measurement: `b=1, seq=512, hd=64` (proxy 1.7e7) runs a
/// backward in ~120 ms, and `b=1, seq=4096, hd=64` (proxy 1.1e9) takes ~7.7 s.
/// This ceiling keeps the longest-sequence cell — the one the kernel's whole
/// memory-amortization argument is about — while dropping the cells that cost
/// minutes per row. Raise it deliberately, and expect the sweep to get slow.
const MAX_SHAPE_COST: usize = 1_100_000_000;

/// One shape point in the sweep.
struct Shape {
    label: &'static str,
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    seq: usize,
    head_dim: usize,
}

/// Deterministic, closed-form fixture data (same shape as the parity test's
/// `det_data`) — the actual values do not matter for timing, only that they
/// are finite and cheap to generate.
fn det_data(shape: &[usize], phase: f32) -> Vec<f32> {
    let n: usize = shape.iter().product();
    (0..n)
        .map(|i| ((i as f32) * 0.1 + phase).sin() * 0.5)
        .collect()
}

/// Median and min, in milliseconds, over timed calls to `f`, after `WARMUP`
/// untimed calls. `client.synchronize()` brackets every timed call so the window
/// covers device execution, not just launch latency.
///
/// Iteration count adapts: always at least `MIN_ITERS`, then more only while the
/// measurement stays inside `BUDGET_PER_MEASUREMENT`, capped at `MAX_ITERS`.
/// Returns `(median_ms, min_ms)`.
fn time_calls_ms<F: FnMut()>(client: &CudaClient, mut f: F) -> (f64, f64) {
    for _ in 0..WARMUP {
        f();
    }
    client.synchronize();

    let mut samples_ms = Vec::with_capacity(MAX_ITERS);
    let sweep_start = Instant::now();
    while samples_ms.len() < MAX_ITERS {
        let start = Instant::now();
        f();
        client.synchronize();
        samples_ms.push(start.elapsed().as_secs_f64() * 1000.0);

        if samples_ms.len() >= MIN_ITERS && sweep_start.elapsed() >= BUDGET_PER_MEASUREMENT {
            break;
        }
    }
    samples_ms.sort_by(|a, b| a.partial_cmp(b).expect("timing sample is NaN"));

    let n = samples_ms.len();
    let median = if n % 2 == 0 {
        (samples_ms[n / 2 - 1] + samples_ms[n / 2]) / 2.0
    } else {
        samples_ms[n / 2]
    };
    (median, samples_ms[0])
}

fn print_row(op: &str, shape: &Shape, mqa: (f64, f64), flash: (f64, f64)) {
    let (mqa_med, mqa_min) = mqa;
    let (flash_med, flash_min) = flash;
    // >1.0 means the dedicated kernel is faster (flash takes longer).
    let ratio_median = flash_med / mqa_med;
    println!(
        "{op:<4} | {label:<28} | b={batch:<2} h={h:<2}/kv={kv:<2} seq={seq:<5} hd={hd:<3} | \
         mqa_gqa med={mqa_med:>8.4}ms min={mqa_min:>8.4}ms | \
         flash_v2 med={flash_med:>8.4}ms min={flash_min:>8.4}ms | \
         speedup(median)={ratio_median:>5.2}x",
        label = shape.label,
        batch = shape.batch,
        h = shape.num_heads,
        kv = shape.num_kv_heads,
        seq = shape.seq,
        hd = shape.head_dim,
    );
}

#[test]
#[ignore = "GPU benchmark: run on demand with --ignored on a quiet machine"]
fn mqa_gqa_vs_flash_bench() {
    if !numr::runtime::cuda::is_cuda_available() {
        eprintln!(
            "SKIPPED: mqa_gqa_vs_flash_bench — the `cuda` feature is on but no CUDA device \
             is available at runtime."
        );
        return;
    }

    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);

    println!(
        "\n=== mqa_gqa dedicated kernel vs flash_v2 (FlashAttentionOps) ===\n\
         WALL-CLOCK / GPU-TIME figures (host Instant + client.synchronize()) — \
         trustworthy only on an otherwise-quiet machine.\n\
         causal=true for every case. warmup={WARMUP} untimed, then {MIN_ITERS}..={MAX_ITERS} \
         timed iterations within a {budget:?} budget per measurement (attention backward is \
         O(batch * seq^2), so per-call cost spans ~3 orders of magnitude here).\n\
         Reporting MEDIAN and MIN, with the iteration count each row actually used.\n",
        budget = BUDGET_PER_MEASUREMENT
    );

    // num_kv_heads=1 => true MQA (ratio 32). num_kv_heads=8 => the ratio-4
    // floor of the kernel's selection heuristic. Both use num_heads=32.
    let kv_configs: [(&str, usize, usize); 2] =
        [("mqa(kv=1,ratio=32)", 32, 1), ("gqa(kv=8,ratio=4)", 32, 8)];
    let head_dims = [64usize, 128usize];
    // Short sequence vs long: the amortization argument for this kernel is
    // K/V memory traffic, which should matter more as seq grows.
    let seqs = [512usize, 4096usize];
    let batches = [1usize, 8usize];

    for &(kv_label, num_heads, num_kv_heads) in &kv_configs {
        for &head_dim in &head_dims {
            for &seq in &seqs {
                for &batch in &batches {
                    // Bound the sweep by predicted cost. Attention backward is
                    // O(batch * seq^2 * head_dim), and across this matrix that spans
                    // ~3 orders of magnitude: ~120 ms at the small end, ~60 s at
                    // b=8/seq=4096. The expensive cells cost minutes per row even at
                    // MIN_ITERS and add no signal the cheaper ones do not already
                    // carry, so skip them LOUDLY rather than pretending the matrix
                    // was fully swept.
                    let cost = batch * seq * seq * head_dim;
                    if cost > MAX_SHAPE_COST {
                        println!(
                            "SKIPPED (cost, not capability): {kv_label} b={batch} seq={seq} \
                             hd={head_dim} — cost proxy {cost} over budget {MAX_SHAPE_COST}."
                        );
                        continue;
                    }

                    assert!(
                        should_use_mqa_gqa(num_heads, num_kv_heads, head_dim),
                        "shape matrix entry ({num_heads}/{num_kv_heads}, hd={head_dim}) is \
                         outside the kernel's own selection heuristic — fix the matrix, not \
                         the assertion"
                    );

                    let shape = Shape {
                        label: kv_label,
                        batch,
                        num_heads,
                        num_kv_heads,
                        seq,
                        head_dim,
                    };
                    run_shape(&client, &device, &shape);
                }
            }
        }
    }
}

fn run_shape(client: &CudaClient, device: &CudaDevice, shape: &Shape) {
    let q_shape = [shape.batch, shape.num_heads, shape.seq, shape.head_dim];
    let kv_shape = [shape.batch, shape.num_kv_heads, shape.seq, shape.head_dim];

    let q_data = det_data(&q_shape, 0.0);
    let k_data = det_data(&kv_shape, 1.7);
    let v_data = det_data(&kv_shape, 3.1);
    let dout_data = det_data(&q_shape, 5.3);

    let q = Tensor::<CudaRuntime>::from_slice(&q_data, &q_shape, device).unwrap();
    let k = Tensor::<CudaRuntime>::from_slice(&k_data, &kv_shape, device).unwrap();
    let v = Tensor::<CudaRuntime>::from_slice(&v_data, &kv_shape, device).unwrap();
    let dout = Tensor::<CudaRuntime>::from_slice(&dout_data, &q_shape, device).unwrap();

    // ---- Forward ----
    let mqa_fwd = time_calls_ms(client, || {
        let _ = mqa_gqa_fwd(
            client,
            &q,
            &k,
            &v,
            shape.num_heads,
            shape.num_kv_heads,
            shape.head_dim,
            true,
        )
        .expect("mqa_gqa_fwd failed during benchmark");
    });
    let flash_fwd = time_calls_ms(client, || {
        let _ = client
            .flash_attention_fwd(
                &q,
                &k,
                &v,
                shape.num_heads,
                shape.num_kv_heads,
                shape.head_dim,
                true,
                0,
                None,
            )
            .expect("flash_attention_fwd failed during benchmark");
    });
    print_row("fwd", shape, mqa_fwd, flash_fwd);

    // ---- Backward ----
    // Each path consumes its OWN forward output/logsumexp, exactly as a real
    // caller would use it — not a shared/foreign (out, lse) pair.
    let (mqa_out, mqa_lse) = mqa_gqa_fwd(
        client,
        &q,
        &k,
        &v,
        shape.num_heads,
        shape.num_kv_heads,
        shape.head_dim,
        true,
    )
    .expect("mqa_gqa_fwd (feeding backward) failed during benchmark");
    let mqa_bwd = time_calls_ms(client, || {
        let _ = mqa_gqa_bwd(
            client,
            &dout,
            &q,
            &k,
            &v,
            &mqa_out,
            &mqa_lse,
            shape.num_heads,
            shape.num_kv_heads,
            shape.head_dim,
            true,
        )
        .expect("mqa_gqa_bwd failed during benchmark");
    });

    let (flash_out, flash_lse) = client
        .flash_attention_fwd(
            &q,
            &k,
            &v,
            shape.num_heads,
            shape.num_kv_heads,
            shape.head_dim,
            true,
            0,
            None,
        )
        .expect("flash_attention_fwd (feeding backward) failed during benchmark");
    let flash_bwd = time_calls_ms(client, || {
        let _ = client
            .flash_attention_bwd(
                &dout,
                &q,
                &k,
                &v,
                &flash_out,
                &flash_lse,
                shape.num_heads,
                shape.num_kv_heads,
                shape.head_dim,
                true,
                0,
            )
            .expect("flash_attention_bwd failed during benchmark");
    });
    print_row("bwd", shape, mqa_bwd, flash_bwd);
}
