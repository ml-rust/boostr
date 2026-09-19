//! A captured `qwen35` decode graph stays valid after `reset()` and a new
//! eager prefill into the same `LayeredKvCache` and `LayeredGdnState`.
//!
//! Run with:
//!   cd boostr && cargo test --release --features cuda --test qwen35_graph_reset
//!
//! Prefill prompt A, capture one decode step against the caches, then
//! reset both caches, prefill prompt B eagerly into the same buffers, and
//! replay the graph. Argmax ids must agree with an eager decode of prompt B
//! at every step, logits within `1e-4`, the GDN buffers after the last step
//! within `1e-5`, and every KV and GDN buffer address must be unchanged.

#![cfg(feature = "cuda")]

mod common;

use boostr::inference::decode_graph::{
    DeviceScalars, MropeScalars, argmax_to_buf, copy_into_stable,
};
use common::qwen35_cuda::{
    ARENA_BYTES, caches, cuda_available, cuda_setup, eager_decode, gdn_flat, max_abs_diff, prefill,
    state_ptrs,
};
use common::qwen35_tiny::{MAX_POS, VOCAB, tiny_model};
use numr::dtype::DType;
use numr::ops::BinaryOps;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

const PREFILL: usize = 5;
const DECODE_STEPS: usize = 6;

/// Both captures in this binary live in this one function, so they never
/// overlap on the device.
#[test]
fn graph_survives_reset_and_reprefill() {
    if !cuda_available() {
        println!("skip: no CUDA device");
        return;
    }
    let (client, device) = cuda_setup();
    let model = tiny_model::<CudaRuntime>(&device, 0x3535_0901);
    let vocab = model.config().vocab_size;
    let prompt_a: Vec<i64> = (0..PREFILL).map(|i| ((i * 5 + 3) % VOCAB) as i64).collect();
    let prompt_b: Vec<i64> = (0..PREFILL).map(|i| ((i * 7 + 1) % VOCAB) as i64).collect();
    assert_ne!(prompt_a, prompt_b);

    // Prompt A into the buffers the graph will read.
    let (mut kv, mut gdn) = caches(&device, &model, MAX_POS);
    let first_a = prefill(&client, &device, &model, &prompt_a, &mut kv, &mut gdn);
    let seq_len = kv.seq_len();

    // Stable buffers, all allocated before capture.
    let token_buf = Tensor::<CudaRuntime>::from_slice(&[first_a], &[1, 1], &device).unwrap();
    let next_token_buf = Tensor::<CudaRuntime>::zeros(&[1], DType::I64, &device).unwrap();
    let logits_buf = Tensor::<CudaRuntime>::zeros(&[1, 1, vocab], DType::F32, &device).unwrap();
    let scalars = DeviceScalars::new(seq_len, &device).unwrap();
    let mrope = MropeScalars::new(seq_len, &device).unwrap();

    // Warm-up outside capture writes the KV cache and the GDN buffers.
    // Reset both in place and prefill again: the addresses must hold.
    scalars.update(&client, seq_len).unwrap();
    mrope.update(&client, seq_len).unwrap();
    let warm = model
        .forward_qwen35_graph_mode(&client, &token_buf, &kv, &gdn, &scalars, &mrope)
        .unwrap();
    argmax_to_buf(&client, &warm, &next_token_buf).unwrap();
    let _ = next_token_buf.to_vec::<i64>();
    let ptrs_before = state_ptrs(&kv, &gdn);
    kv.reset();
    gdn.reset(&client).unwrap();
    assert_eq!(
        prefill(&client, &device, &model, &prompt_a, &mut kv, &mut gdn),
        first_a
    );
    assert_eq!(state_ptrs(&kv, &gdn), ptrs_before);

    let graph = CudaRuntime::capture_graph_into_with_arena(
        &client,
        &[&token_buf],
        &[&next_token_buf],
        ARENA_BYTES,
        |c| {
            let logits = model
                .forward_qwen35_graph_mode(c, &token_buf, &kv, &gdn, &scalars, &mrope)
                .map_err(|e| numr::error::Error::Backend(format!("capture forward: {e}")))?;
            copy_into_stable(c, &logits, &logits_buf)?;
            argmax_to_buf(c, &logits, &next_token_buf)
        },
    )
    .unwrap();

    // Eager reference for prompt B on fresh caches.
    let (mut kv_ref, mut gdn_ref) = caches(&device, &model, MAX_POS);
    let first_b = prefill(
        &client,
        &device,
        &model,
        &prompt_b,
        &mut kv_ref,
        &mut gdn_ref,
    );
    let (rows_ref, ids_ref) = eager_decode(
        &client,
        &device,
        &model,
        first_b,
        DECODE_STEPS,
        &mut kv_ref,
        &mut gdn_ref,
    );

    // Reset, prefill prompt B into the captured buffers, replay.
    kv.reset();
    gdn.reset(&client).unwrap();
    assert_eq!(
        prefill(&client, &device, &model, &prompt_b, &mut kv, &mut gdn),
        first_b
    );
    assert_eq!(state_ptrs(&kv, &gdn), ptrs_before);
    assert_eq!(kv.seq_len(), seq_len);

    let first_b_tensor = Tensor::<CudaRuntime>::from_slice(&[first_b], &[1, 1], &device).unwrap();
    client.copy_into(&token_buf, &first_b_tensor).unwrap();

    let mut rows = Vec::with_capacity(DECODE_STEPS);
    let mut ids = Vec::with_capacity(DECODE_STEPS);
    for (step, position) in (seq_len..seq_len + DECODE_STEPS).enumerate() {
        if step > 0 {
            copy_into_stable(&client, &next_token_buf, &token_buf).unwrap();
        }
        scalars.update(&client, position).unwrap();
        mrope.update(&client, position).unwrap();
        graph.launch().unwrap();
        rows.push(logits_buf.to_vec::<f32>());
        ids.push(next_token_buf.to_vec::<i64>()[0]);
    }

    assert_eq!(
        ids_ref, ids,
        "argmax ids diverge between eager and replayed graph"
    );
    for (step, (a, b)) in rows_ref.iter().zip(&rows).enumerate() {
        assert!(a.iter().all(|v| v.is_finite()));
        let diff = max_abs_diff(a, b);
        assert!(diff < 1e-4, "step {step}: logits diff {diff}");
    }

    let (conv_ref, ssm_ref) = gdn_flat(&gdn_ref);
    let (conv, ssm) = gdn_flat(&gdn);
    let conv_diff = max_abs_diff(&conv_ref, &conv);
    assert!(conv_diff < 1e-5, "gdn conv window diff {conv_diff}");
    let ssm_diff = max_abs_diff(&ssm_ref, &ssm);
    assert!(ssm_diff < 1e-5, "gdn ssm state diff {ssm_diff}");
    assert_eq!(state_ptrs(&kv, &gdn), ptrs_before);
}
