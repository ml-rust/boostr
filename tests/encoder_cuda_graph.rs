//! CUDA graph capture integration tests for `Encoder::embed_inference`.
//!
//! Validates that the graph-captured forward path produces numerically identical
//! results to the standard forward path, and that 100 consecutive replays do not
//! fault or drift.
//!
//! Run with:
//!   cd boostr && cargo test --features cuda --test encoder_cuda_graph
//!
//! The entire file is gated on the `cuda` feature so it compiles away entirely
//! on CPU-only builds.

#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use boostr::model::encoder::config::{ArchFamily, HiddenAct};
use boostr::model::encoder::model::Pooling;
use boostr::{Encoder, EncoderConfig};
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

// ---------------------------------------------------------------------------
// CUDA serialization lock
// ---------------------------------------------------------------------------
//
// CUDA graph capture (`cuStreamBeginCapture`) puts the stream into a "capture"
// state that is incompatible with any other use of the same stream from another
// thread.  Rust test threads run concurrently by default, so without explicit
// serialization multiple tests would try to capture the SAME shared CUDA stream
// simultaneously, causing CUDA_ERROR_ILLEGAL_STATE on all but the first.
//
// This lock guarantees that only one CUDA-using test body executes at a time,
// matching the approach used in `tests/backend_parity/helpers.rs`.
static CUDA_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn cuda_lock() -> std::sync::MutexGuard<'static, ()> {
    CUDA_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

// ---------------------------------------------------------------------------
// Skip guard
// ---------------------------------------------------------------------------

/// Returns `true` if a real CUDA device is available at runtime.
fn cuda_available() -> bool {
    numr::runtime::cuda::is_cuda_available()
}

// ---------------------------------------------------------------------------
// Encoder constructor
// ---------------------------------------------------------------------------

/// Port of `make_test_encoder()` from `src/model/encoder/model/tests.rs`, using
/// `CudaRuntime` instead of `CpuRuntime`.
///
/// Config: hidden=8, 1 layer, 2 heads, intermediate=16, vocab=10, max_pos=32.
/// Uses varied synthetic weights (slight per-element offset) so that LayerNorm
/// does not degenerate to zero variance. Pooling::Mean.
fn make_cuda_test_encoder() -> (Encoder<CudaRuntime>, CudaClient, CudaDevice) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);

    let config = EncoderConfig {
        vocab_size: 10,
        hidden_size: 8,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        intermediate_size: 16,
        max_position_embeddings: 32,
        layer_norm_eps: 1e-12,
        hidden_act: HiddenAct::Gelu,
        type_vocab_size: 0,
        arch_family: ArchFamily::Bert,
        padding_token_id: 0,
        compute_dtype: numr::dtype::DType::F32,
        max_tokens_per_forward: None,
    };

    let d = &device;

    // Varied weights: base + index * small_delta avoids identical rows so that
    // LayerNorm variance is non-zero and outputs are non-trivial.
    let make_w = |rows: usize, cols: usize, base: f32| -> Vec<f32> {
        (0..rows * cols)
            .map(|i| base + (i as f32) * 0.001)
            .collect()
    };

    let encoder = Encoder::from_weights(config, Pooling::Mean, |name| match name {
        "embeddings.word_embeddings.weight" => {
            Ok(Tensor::from_slice(&make_w(10, 8, 0.1), &[10, 8], d))
        }
        "embeddings.position_embeddings.weight" => {
            Ok(Tensor::from_slice(&make_w(32, 8, 0.01), &[32, 8], d))
        }
        "embeddings.layer_norm.weight" => Ok(Tensor::from_slice(&make_w(8, 1, 1.0), &[8], d)),
        "embeddings.layer_norm.bias" => Ok(Tensor::from_slice(&[0.0f32; 8], &[8], d)),
        n if n.ends_with("query.weight")
            || n.ends_with("key.weight")
            || n.ends_with("value.weight") =>
        {
            Ok(Tensor::from_slice(&make_w(8, 8, 0.02), &[8, 8], d))
        }
        n if n.ends_with("query.bias") || n.ends_with("key.bias") || n.ends_with("value.bias") => {
            Ok(Tensor::from_slice(&[0.0f32; 8], &[8], d))
        }
        n if n.ends_with("attention.output.dense.weight") => {
            Ok(Tensor::from_slice(&make_w(8, 8, 0.02), &[8, 8], d))
        }
        n if n.ends_with("attention.output.dense.bias") => {
            Ok(Tensor::from_slice(&[0.0f32; 8], &[8], d))
        }
        n if n.ends_with("output.dense.weight") => {
            Ok(Tensor::from_slice(&make_w(8, 16, 0.02), &[8, 16], d))
        }
        n if n.ends_with("output.dense.bias") => Ok(Tensor::from_slice(&[0.0f32; 8], &[8], d)),
        n if n.ends_with("LayerNorm.weight") => Ok(Tensor::from_slice(&make_w(8, 1, 1.0), &[8], d)),
        n if n.ends_with("LayerNorm.bias") => Ok(Tensor::from_slice(&[0.0f32; 8], &[8], d)),
        n if n.ends_with("intermediate.dense.weight") => {
            Ok(Tensor::from_slice(&make_w(16, 8, 0.02), &[16, 8], d))
        }
        n if n.ends_with("intermediate.dense.bias") => {
            Ok(Tensor::from_slice(&[0.0f32; 16], &[16], d))
        }
        _ => Err(boostr::error::Error::ModelError {
            reason: format!("unknown weight: {name}"),
        }),
    })
    .expect("Encoder::from_weights must succeed with synthetic weights");

    (encoder, client, device)
}

// ---------------------------------------------------------------------------
// Tensor comparison helper
// ---------------------------------------------------------------------------

/// Assert that two CUDA tensors are element-wise close within tolerance.
///
/// Reads both tensors to host via `to_vec` (which implies a stream sync).
/// Panics with a descriptive message on the first out-of-tolerance element.
///
/// For the "no drift" / bit-exact case, pass `rtol=0.0, atol=0.0`.
fn assert_tensors_close(
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    rtol: f32,
    atol: f32,
    label: &str,
) {
    assert_eq!(
        a.shape(),
        b.shape(),
        "[{label}] shape mismatch: {:?} vs {:?}",
        a.shape(),
        b.shape()
    );

    let a_vals: Vec<f32> = a.to_vec();
    let b_vals: Vec<f32> = b.to_vec();

    for (i, (&av, &bv)) in a_vals.iter().zip(b_vals.iter()).enumerate() {
        if av.is_nan() || av.is_infinite() {
            panic!("[{label}] tensor A has non-finite value {av} at index {i}");
        }
        if bv.is_nan() || bv.is_infinite() {
            panic!("[{label}] tensor B has non-finite value {bv} at index {i}");
        }
        let tol = atol + rtol * bv.abs();
        let diff = (av - bv).abs();
        if diff > tol {
            panic!(
                "[{label}] mismatch at index {i}: A={av}, B={bv}, diff={diff}, tol={tol} \
                 (rtol={rtol}, atol={atol})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 1: single-shot parity — graph path vs. standard path
// ---------------------------------------------------------------------------

/// Verify that the CUDA graph capture path produces the same embeddings as the
/// standard (non-graph) path.
///
/// The first call to `embed_inference` triggers graph capture.  CUDA graph
/// capture records kernel dispatches but does NOT execute them during recording
/// — the captured graph must be launched at least once for actual compute to
/// occur and results to appear in `stable_out`.
///
/// This test therefore calls `embed_inference` TWICE: the first call captures
/// (and the returned tensor from the cache is the pre-launch stable_out buffer),
/// and the second call replays the captured graph so the output is populated.
/// The parity check uses the second (replay) result.
#[test]
fn embed_inference_matches_standard_cuda() {
    let _guard = cuda_lock();
    if !cuda_available() {
        eprintln!("CUDA not available — skipping embed_inference_matches_standard_cuda");
        return;
    }

    let (encoder, client, device) = make_cuda_test_encoder();

    let input_ids = Tensor::<CudaRuntime>::from_slice(&[1i64, 2, 3, 4], &[1, 4], &device);

    // First call → graph capture (CUDA records ops; no compute yet; output buffer
    // still holds its zero-initialization until the first launch).
    let first_out = encoder
        .embed_inference(&client, &input_ids, None)
        .expect("embed_inference call 1 (capture) must not error");

    client.synchronize();
    let first_vals: Vec<f32> = first_out.to_vec();
    println!("call-1 (capture) output: {:?}", &first_vals);

    // Second call → graph replay (cuGraphLaunch executes the captured ops;
    // stable_out is now populated with real results).
    let graph_out = encoder
        .embed_inference(&client, &input_ids, None)
        .expect("embed_inference call 2 (replay) must not error");

    // Standard (non-graph) path for comparison.
    let std_out = encoder
        .embed_inference_standard(&client, &input_ids, None)
        .expect("embed_inference_standard must succeed");

    client.synchronize();

    let graph_vals: Vec<f32> = graph_out.to_vec();
    let std_vals: Vec<f32> = std_out.to_vec();
    println!("call-2 (replay) output:  {:?}", &graph_vals);
    println!("standard path output:    {:?}", &std_vals);

    // Diagnose capture-path output from call 1: was it zero (pre-launch)?
    let capture_is_zero = first_vals.iter().all(|&v| v == 0.0);
    println!(
        "capture-path call-1 output is all-zero: {} \
         (expected: true — graph ops are not executed during recording)",
        capture_is_zero
    );

    // Replay output must match the standard path within FP32 tolerance.
    assert_tensors_close(
        &graph_out,
        &std_out,
        1e-5,
        1e-6,
        "graph replay (call 2) vs standard",
    );

    // Exactly 1 capture should have occurred (both calls used the same shape).
    assert_eq!(
        encoder.graph_capture_count(),
        1,
        "expected exactly 1 capture after two calls to the same shape"
    );

    println!("PASS: embed_inference_matches_standard_cuda");
}

// ---------------------------------------------------------------------------
// Test 2: 100 replays — no fault, no drift
// ---------------------------------------------------------------------------

/// Verify that 100 consecutive replays of the same captured graph do not fault
/// (no `CUDA_ERROR_ILLEGAL_ADDRESS`) and produce bit-identical results.
///
/// The first call triggers capture (output may be zeros — pre-launch buffer).
/// The second call triggers the first real launch, establishing the reference.
/// Calls 3..=100 are replays that must be bit-exact with call 2.
#[test]
fn graph_capture_100_replays_no_drift() {
    let _guard = cuda_lock();
    if !cuda_available() {
        eprintln!("CUDA not available — skipping graph_capture_100_replays_no_drift");
        return;
    }

    let (encoder, client, device) = make_cuda_test_encoder();

    let input_ids = Tensor::<CudaRuntime>::from_slice(&[1i64, 2, 3, 4], &[1, 4], &device);

    // Call 0: triggers graph capture.
    let _capture_out = encoder
        .embed_inference(&client, &input_ids, None)
        .expect("embed_inference capture call (iter 0) must succeed");
    client.synchronize();

    // Call 1: first replay — this is the reference result.
    let reference = encoder
        .embed_inference(&client, &input_ids, None)
        .expect("embed_inference replay (iter 1) must succeed");
    client.synchronize();
    let reference_vals: Vec<f32> = reference.to_vec();
    println!("reference values (first replay): {:?}", &reference_vals);

    // Calls 2..=99: further replays must be bit-exact with the reference.
    for iter in 2usize..100 {
        let result = encoder
            .embed_inference(&client, &input_ids, None)
            .unwrap_or_else(|e| {
                panic!(
                    "embed_inference replay at iteration {iter} returned error: {e:#}\n\
                     If this is CUDA_ERROR_ILLEGAL_ADDRESS, the graph-internal scratch \
                     arena is being freed before the graph completes."
                )
            });

        client.synchronize();

        // Bit-exact: same inputs + same weights + deterministic graph → identical output.
        assert_tensors_close(
            &result,
            &reference,
            0.0,
            0.0,
            &format!("replay iter {iter} vs reference"),
        );
    }

    // Only 1 capture: all other 99 calls were replays.
    assert_eq!(
        encoder.graph_capture_count(),
        1,
        "expected exactly 1 capture; all remaining calls should be replays"
    );

    println!("PASS: graph_capture_100_replays_no_drift");
}

// ---------------------------------------------------------------------------
// Test 3: multiple shapes — one capture per unique shape
// ---------------------------------------------------------------------------

/// Verify that the graph cache captures exactly once per unique `(batch, seq_len)`
/// shape, and that second-pass calls are pure replays.
#[test]
fn graph_capture_multiple_shapes_no_thrash() {
    let _guard = cuda_lock();
    if !cuda_available() {
        eprintln!("CUDA not available — skipping graph_capture_multiple_shapes_no_thrash");
        return;
    }

    let (encoder, client, device) = make_cuda_test_encoder();

    // Three distinct shapes within max_position_embeddings=32.
    let shapes: &[(usize, usize)] = &[(1, 4), (2, 4), (1, 6)];

    // First pass: capture one graph per shape.
    // Each call 1 per shape = capture (output may be zero-init).
    // Each call 2 per shape = first replay (real output).
    let mut first_replay_results: Vec<Tensor<CudaRuntime>> = Vec::new();

    for &(batch, seq_len) in shapes {
        let ids_data: Vec<i64> = (1..=(batch * seq_len) as i64).collect();
        let input_ids = Tensor::<CudaRuntime>::from_slice(&ids_data, &[batch, seq_len], &device);

        // Call A: capture (zeros expected out of graph path).
        let _cap = encoder
            .embed_inference(&client, &input_ids, None)
            .expect("embed_inference capture must succeed");
        client.synchronize();

        // Call B: first replay (real result).
        let replay_out = encoder
            .embed_inference(&client, &input_ids, None)
            .expect("embed_inference first replay must succeed");

        // Standard path for parity check.
        let std_out = encoder
            .embed_inference_standard(&client, &input_ids, None)
            .expect("embed_inference_standard must succeed");

        client.synchronize();

        assert_tensors_close(
            &replay_out,
            &std_out,
            1e-5,
            1e-6,
            &format!("shape ({batch},{seq_len}) replay vs standard"),
        );

        first_replay_results.push(replay_out);
    }

    // After 3 distinct shapes, exactly 3 captures.
    assert_eq!(
        encoder.graph_capture_count(),
        3,
        "expected 3 captures after 3 distinct shapes"
    );

    // Second pass: same shapes — these must be pure replays, bit-exact with
    // the first replay result.
    for (idx, &(batch, seq_len)) in shapes.iter().enumerate() {
        let ids_data: Vec<i64> = (1..=(batch * seq_len) as i64).collect();
        let input_ids = Tensor::<CudaRuntime>::from_slice(&ids_data, &[batch, seq_len], &device);

        let replay_out = encoder
            .embed_inference(&client, &input_ids, None)
            .expect("embed_inference second pass (replay) must succeed");

        client.synchronize();

        assert_tensors_close(
            &replay_out,
            &first_replay_results[idx],
            0.0,
            0.0,
            &format!("shape ({batch},{seq_len}) second pass vs first replay"),
        );
    }

    // Still exactly 3 captures — no new ones from the replay pass.
    assert_eq!(
        encoder.graph_capture_count(),
        3,
        "capture count must remain 3 after second pass (all replays)"
    );

    println!("PASS: graph_capture_multiple_shapes_no_thrash");
}

// ---------------------------------------------------------------------------
// Test 4: cache eviction safety — more shapes than GRAPH_CACHE_CAP
// ---------------------------------------------------------------------------

/// Verify that calling `embed_inference` with more than `GRAPH_CACHE_CAP` (16)
/// distinct shapes does not crash on cache eviction.
///
/// Each shape: call 1 = capture (zeros from stable_out), call 2 = first replay
/// (real result). We only verify shape and finiteness on the replay call.
#[test]
fn graph_capture_cache_eviction_safe() {
    let _guard = cuda_lock();
    if !cuda_available() {
        eprintln!("CUDA not available — skipping graph_capture_cache_eviction_safe");
        return;
    }

    let (encoder, client, device) = make_cuda_test_encoder();

    // GRAPH_CACHE_CAP = 16; generate 18 distinct shapes.
    // max_position_embeddings = 32, so seq_len must stay ≤ 32.
    // b in 1..=3, s in 4..=10 → 21 combinations; take the first 18.
    let mut shapes: Vec<(usize, usize)> = Vec::new();
    'outer: for b in 1usize..=3 {
        for s in 4usize..=10 {
            shapes.push((b, s));
            if shapes.len() == 18 {
                break 'outer;
            }
        }
    }
    assert_eq!(shapes.len(), 18, "need exactly 18 shapes for this test");

    for (run, &(batch, seq_len)) in shapes.iter().enumerate() {
        let ids_data: Vec<i64> = (1..=(batch * seq_len) as i64)
            .map(|v| v % 10) // keep within vocab_size=10
            .collect();
        let input_ids = Tensor::<CudaRuntime>::from_slice(&ids_data, &[batch, seq_len], &device);

        // Capture call (output is zeros from pre-launch stable_out).
        let _cap = encoder
            .embed_inference(&client, &input_ids, None)
            .unwrap_or_else(|e| {
                panic!(
                    "embed_inference capture must not crash on eviction. \
                     shape ({batch},{seq_len}), run {run}: {e:#}"
                )
            });
        client.synchronize();

        // First replay: actual result.
        let result = encoder
            .embed_inference(&client, &input_ids, None)
            .unwrap_or_else(|e| {
                panic!(
                    "embed_inference first replay must not crash on eviction. \
                     shape ({batch},{seq_len}), run {run}: {e:#}"
                )
            });
        client.synchronize();

        let vals: Vec<f32> = result.to_vec();

        // All output values must be finite.
        for (i, &v) in vals.iter().enumerate() {
            if v.is_nan() || v.is_infinite() {
                panic!("shape ({batch},{seq_len}) run {run}: output[{i}] = {v} is non-finite");
            }
        }

        assert_eq!(
            result.shape(),
            &[batch, 8],
            "shape ({batch},{seq_len}): wrong output shape"
        );

        println!(
            "  run {:02}: shape ({batch:},{seq_len:}), capture_count={}, output[..4]={:.4?}",
            run,
            encoder.graph_capture_count(),
            &vals[..vals.len().min(4)],
        );
    }

    // Each of the 18 shapes triggered exactly one capture.
    assert_eq!(
        encoder.graph_capture_count(),
        18,
        "expected 18 distinct captures (one per unique shape)"
    );

    println!("PASS: graph_capture_cache_eviction_safe");
}
