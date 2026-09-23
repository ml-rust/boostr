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

use boostr::inference::LaneTable;
use boostr::inference::decode_graph::Qwen35DecodeGraph;
use common::qwen35_cuda::{
    argmax, caches, cuda_available, cuda_setup, eager_decode, gdn_flat, max_abs_diff, prefill,
    state_ptrs,
};
use common::qwen35_graph::{GraphBufs, capture};
use common::qwen35_tiny::{MAX_POS, VOCAB, tiny_model};
use numr::runtime::cuda::CudaRuntime;

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

    // Stable buffers and lane state, all set before capture.
    let mut bufs = GraphBufs::new(&device, 1, vocab);
    let mut lanes = LaneTable::<CudaRuntime>::new(1, MAX_POS, &device).unwrap();
    lanes.claim(&client, 0, 0, seq_len, seq_len).unwrap();

    // Warm-up outside capture writes the KV cache and the GDN buffers.
    // Reset both in place and prefill again: the addresses must hold.
    bufs.warmup(&client, &model, &kv, &gdn, &lanes, MAX_POS);
    let ptrs_before = state_ptrs(&kv, &gdn);
    kv.reset();
    gdn.reset(&client).unwrap();
    assert_eq!(
        prefill(&client, &device, &model, &prompt_a, &mut kv, &mut gdn),
        first_a
    );
    assert_eq!(state_ptrs(&kv, &gdn), ptrs_before);

    let mut captures = 0usize;
    let (graph, _) = capture(
        &client,
        &device,
        &model,
        &kv,
        &gdn,
        &lanes,
        &bufs,
        &mut captures,
    );
    assert_eq!(captures, 1, "one decode step, one capture");
    let GraphBufs {
        token_buf,
        logits_buf,
        scalars,
    } = bufs;
    let mut decode =
        Qwen35DecodeGraph::new(graph, scalars, MAX_POS, lanes, token_buf, logits_buf).unwrap();

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

    // Prompt B sits at the same length, so the lane is claimed again at the
    // position the second prefill left.
    decode.lanes.claim(&client, 0, 0, seq_len, seq_len).unwrap();
    decode.set_next_tokens(&client, &[first_b]).unwrap();

    let mut rows = Vec::with_capacity(DECODE_STEPS);
    let mut ids = Vec::with_capacity(DECODE_STEPS);
    for _ in 0..DECODE_STEPS {
        decode.pre_replay_and_launch(&client).unwrap();
        let row = decode.logits().to_vec::<f32>();
        let token = argmax(&row);
        decode.set_next_tokens(&client, &[token]).unwrap();
        rows.push(row);
        ids.push(token);
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
