//! Unit tests for the MiniCPM4 incremental-decode path.
//!
//! In a sibling file rather than inline so `decode.rs` stays inside the
//! architecture-file size limit — the same split
//! `llama/model/blocks/attention.rs` uses.

use super::*;
use crate::model::audio::voxcpm::minicpm4::model::tests::{
    HIDDEN, filled, tiny_model, tiny_nope_model,
};
use crate::test_utils::cpu_setup;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

fn values(v: &Var<CpuRuntime>) -> Vec<f32> {
    v.tensor().contiguous().expect("contiguous").to_vec::<f32>()
}

/// Non-degenerate, deterministic embeddings for `seq` positions, batch 1.
fn embeds(seq: usize, device: &numr::runtime::cpu::CpuDevice) -> Var<CpuRuntime> {
    let data: Vec<f32> = (0..seq * HIDDEN)
        .map(|i| ((i % 11) as f32 - 5.0) / 8.0)
        .collect();
    Var::new(
        Tensor::<CpuRuntime>::from_slice(&data, &[1, seq, HIDDEN], device).expect("embeds"),
        false,
    )
}

/// One position of `[1, seq, HIDDEN]` as the `[1, HIDDEN]` shape
/// `decode_step` takes.
fn row_at(x: &Var<CpuRuntime>, position: usize) -> Var<CpuRuntime> {
    Var::new(
        x.tensor()
            .narrow(1, position, 1)
            .expect("narrow")
            .contiguous()
            .expect("contiguous")
            .reshape(&[1, HIDDEN])
            .expect("reshape"),
        false,
    )
}

/// The load-bearing test: stepping one position at a time must reproduce
/// the full-sequence forward. Different key-axis reduction order, so a
/// tight tolerance rather than bit-equality.
#[test]
fn step_wise_matches_full_sequence() {
    let (client, device) = cpu_setup();
    let model = tiny_model(&device);
    let seq = 5;

    let x = embeds(seq, &device);
    let full = values(&model.forward(&client, &x).expect("forward"));

    let mut cache = model.new_kv_cache(1, 8).expect("cache");
    for position in 0..seq {
        let row = row_at(&x, position);
        let step = values(
            &model
                .decode_step(&client, &row, &mut cache, position)
                .expect("decode_step"),
        );
        assert_eq!(step.len(), HIDDEN);
        let expected = &full[position * HIDDEN..(position + 1) * HIDDEN];
        for (got, want) in step.iter().zip(expected) {
            assert!(
                (got - want).abs() < 1e-5,
                "position {position}: step {got} vs full {want}"
            );
        }
        // A model whose output is constant across positions would pass the
        // comparison above vacuously.
        assert!(step.iter().any(|v| v.abs() > 1e-6), "degenerate output");
    }
    assert_eq!(cache.seq_len(), seq);
}

/// Prefill a prefix, then continue stepping — the real generate sequence.
#[test]
fn prefill_then_step_matches_full_sequence() {
    let (client, device) = cpu_setup();
    let model = tiny_model(&device);
    let seq = 5;
    let prefix = 3;

    let x = embeds(seq, &device);
    let full = values(&model.forward(&client, &x).expect("forward"));

    let mut cache = model.new_kv_cache(1, seq).expect("cache");
    let prefix_x = Var::new(
        x.tensor()
            .narrow(1, 0, prefix)
            .expect("narrow")
            .contiguous()
            .expect("contiguous"),
        false,
    );
    let primed = values(
        &model
            .prefill(&client, &prefix_x, &mut cache)
            .expect("prefill"),
    );
    assert_eq!(cache.seq_len(), prefix);
    for (got, want) in primed.iter().zip(&full[..prefix * HIDDEN]) {
        assert!((got - want).abs() < 1e-5, "prefill {got} vs full {want}");
    }

    for position in prefix..seq {
        let row = row_at(&x, position);
        let step = values(
            &model
                .decode_step(&client, &row, &mut cache, position)
                .expect("decode_step"),
        );
        let expected = &full[position * HIDDEN..(position + 1) * HIDDEN];
        for (got, want) in step.iter().zip(expected) {
            assert!(
                (got - want).abs() < 1e-5,
                "position {position}: step {got} vs full {want}"
            );
        }
    }
}

/// The reference raises at `current_length >= max_length`. So do we — and
/// without panicking or writing past the cache.
#[test]
fn stepping_past_max_length_errors() {
    let (client, device) = cpu_setup();
    let model = tiny_model(&device);
    let max_length = 2;
    let mut cache = model.new_kv_cache(1, max_length).expect("cache");

    let row = Var::new(filled(&[1, HIDDEN], 3, &device), false);
    for position in 0..max_length {
        model
            .decode_step(&client, &row, &mut cache, position)
            .expect("in-range step");
    }
    let err = model
        .decode_step(&client, &row, &mut cache, max_length)
        .unwrap_err();
    assert!(err.to_string().contains("max_length"), "got {err}");
    assert_eq!(
        cache.seq_len(),
        max_length,
        "cache advanced past max_length"
    );
}

#[test]
fn decode_step_rejects_out_of_order_position() {
    let (client, device) = cpu_setup();
    let model = tiny_model(&device);
    let mut cache = model.new_kv_cache(1, 8).expect("cache");
    let row = Var::new(filled(&[1, HIDDEN], 3, &device), false);

    // Cache is empty, so only position 0 is writable.
    let err = model.decode_step(&client, &row, &mut cache, 3).unwrap_err();
    assert!(
        err.to_string().contains("next free cache slot"),
        "got {err}"
    );
}

#[test]
fn decode_step_rejects_3d_input() {
    let (client, device) = cpu_setup();
    let model = tiny_model(&device);
    let mut cache = model.new_kv_cache(1, 8).expect("cache");
    let x = Var::new(filled(&[1, 1, HIDDEN], 3, &device), false);
    let err = model.decode_step(&client, &x, &mut cache, 0).unwrap_err();
    assert!(err.to_string().contains("2D"), "got {err}");
}

#[test]
fn new_kv_cache_rejects_degenerate_sizes() {
    let (_client, device) = cpu_setup();
    let model = tiny_model(&device);
    assert!(model.new_kv_cache(0, 4).is_err(), "zero batch accepted");
    assert!(
        model.new_kv_cache(1, 0).is_err(),
        "zero max_length accepted"
    );
    // The tiny model's RoPE table is 16 positions long.
    assert!(
        model.new_kv_cache(1, 17).is_err(),
        "max_length beyond the RoPE table accepted"
    );
    assert!(model.new_kv_cache(1, 16).is_ok());
}

#[test]
fn prefill_rejects_batch_mismatch() {
    let (client, device) = cpu_setup();
    let model = tiny_model(&device);
    let mut cache = model.new_kv_cache(2, 8).expect("cache");
    let x = embeds(3, &device);
    let err = model.prefill(&client, &x, &mut cache).unwrap_err();
    assert!(err.to_string().contains("batch"), "got {err}");
}

/// The NoPE (`residual_lm`) stack has TWO RoPE call sites to skip — the
/// full-sequence one inside `attention_core_masked` and the direct one in
/// `forward_cached`. Honouring only one leaves the paths computing different
/// models, which is exactly what this comparison catches.
#[test]
fn nope_step_wise_matches_full_sequence() {
    let (client, device) = cpu_setup();
    let model = tiny_nope_model(&device);
    assert!(!model.uses_rope());
    let seq = 5;

    let x = embeds(seq, &device);
    let full = values(&model.forward(&client, &x).expect("forward"));

    let mut cache = model.new_kv_cache(1, 8).expect("cache");
    for position in 0..seq {
        let row = row_at(&x, position);
        let step = values(
            &model
                .decode_step(&client, &row, &mut cache, position)
                .expect("decode_step"),
        );
        let expected = &full[position * HIDDEN..(position + 1) * HIDDEN];
        for (got, want) in step.iter().zip(expected) {
            assert!(
                (got - want).abs() < 1e-5,
                "position {position}: step {got} vs full {want}"
            );
        }
        assert!(step.iter().any(|v| v.abs() > 1e-6), "degenerate output");
    }
    assert_eq!(cache.seq_len(), seq);
}

/// A NoPE stack owns no RoPE table, so the cache length it can serve is
/// bounded by the cache alone.
#[test]
fn nope_kv_cache_is_not_bounded_by_a_rope_table() {
    let (_client, device) = cpu_setup();
    let model = tiny_nope_model(&device);
    // 17 exceeds the 16-position table the rotary tiny model carries; this one
    // has no table to exceed.
    assert!(model.new_kv_cache(1, 17).is_ok());
    assert!(
        model.new_kv_cache(1, 0).is_err(),
        "zero max_length accepted"
    );
}
