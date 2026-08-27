//! Tests for the ggml-conventional VoxCPM2 name map, the squeeze rule, and
//! the sentinel probe.
//!
//! The probe tests build a MINIMAL in-memory GGUF (a header plus one
//! zero-length-data tensor entry) rather than opening a real 1.2 GB
//! checkpoint: what is under test is which name is present, not any byte of
//! tensor data.

use super::names::{
    GGML_SENTINEL, GgufNaming, VERBATIM_SENTINEL, hf_to_ggml_name, probe_naming,
    restore_leading_unit_dims,
};
use crate::format::gguf::Gguf;

/// One representative tensor from each of the four per-layer stacks, plus
/// the shared suffix table exercised across them.
#[test]
fn maps_every_per_layer_stack() {
    let cases = [
        (
            "base_lm.layers.0.self_attn.q_proj.weight",
            "tslm.blk.0.attn_q.weight",
        ),
        (
            "residual_lm.layers.7.mlp.down_proj.weight",
            "ralm.blk.7.ffn_down.weight",
        ),
        (
            "feat_encoder.encoder.layers.11.input_layernorm.weight",
            "locenc.blk.11.attn_norm.weight",
        ),
        (
            "feat_decoder.estimator.decoder.layers.3.post_attention_layernorm.weight",
            "locdit.blk.3.ffn_norm.weight",
        ),
    ];
    for (ours, theirs) in cases {
        assert_eq!(hf_to_ggml_name(ours).as_deref(), Some(theirs), "for {ours}");
    }
}

/// All nine within-layer suffixes, on one stack, so a typo in the table is
/// caught per row rather than only where a stack test happens to touch it.
#[test]
fn maps_every_within_layer_suffix() {
    let cases = [
        ("self_attn.q_proj.weight", "attn_q.weight"),
        ("self_attn.k_proj.weight", "attn_k.weight"),
        ("self_attn.v_proj.weight", "attn_v.weight"),
        ("self_attn.o_proj.weight", "attn_output.weight"),
        ("input_layernorm.weight", "attn_norm.weight"),
        ("post_attention_layernorm.weight", "ffn_norm.weight"),
        ("mlp.gate_proj.weight", "ffn_gate.weight"),
        ("mlp.up_proj.weight", "ffn_up.weight"),
        ("mlp.down_proj.weight", "ffn_down.weight"),
    ];
    for (ours, theirs) in cases {
        let mapped = hf_to_ggml_name(&format!("base_lm.layers.5.{ours}"));
        assert_eq!(
            mapped.as_deref(),
            Some(format!("tslm.blk.5.{theirs}").as_str()),
            "for {ours}"
        );
    }
}

/// Every flat-tensor family, including both members of the biased ones and
/// the `linear_1`/`linear_2` -> `0`/`1` index shift on the two timestep MLPs.
#[test]
fn maps_every_flat_tensor_family() {
    let cases = [
        ("base_lm.embed_tokens.weight", "tslm.token_embd.weight"),
        ("base_lm.norm.weight", "tslm.output_norm.weight"),
        ("residual_lm.norm.weight", "ralm.output_norm.weight"),
        (
            "feat_encoder.encoder.norm.weight",
            "locenc.output_norm.weight",
        ),
        ("feat_encoder.in_proj.weight", "locenc.in_proj.weight"),
        ("feat_encoder.in_proj.bias", "locenc.in_proj.bias"),
        ("feat_encoder.special_token", "locenc.cls_token"),
        (
            "feat_decoder.estimator.decoder.norm.weight",
            "locdit.output_norm.weight",
        ),
        (
            "feat_decoder.estimator.in_proj.weight",
            "locdit.in_proj.weight",
        ),
        (
            "feat_decoder.estimator.out_proj.bias",
            "locdit.out_proj.bias",
        ),
        (
            "feat_decoder.estimator.cond_proj.weight",
            "locdit.cond_proj.weight",
        ),
        (
            "feat_decoder.estimator.time_mlp.linear_1.weight",
            "locdit.time_mlp.0.weight",
        ),
        (
            "feat_decoder.estimator.time_mlp.linear_2.bias",
            "locdit.time_mlp.1.bias",
        ),
        (
            "feat_decoder.estimator.delta_time_mlp.linear_1.bias",
            "locdit.dt_mlp.0.bias",
        ),
        (
            "feat_decoder.estimator.delta_time_mlp.linear_2.weight",
            "locdit.dt_mlp.1.weight",
        ),
        ("fsq_layer.in_proj.weight", "fsq.in_proj.weight"),
        ("fsq_layer.out_proj.bias", "fsq.out_proj.bias"),
        ("enc_to_lm_proj.weight", "proj.enc_to_lm.weight"),
        ("fusion_concat_proj.bias", "proj.fusion.bias"),
        ("lm_to_dit_proj.weight", "proj.lm_to_dit.weight"),
        ("res_to_dit_proj.bias", "proj.res_to_dit.bias"),
        ("stop_head.weight", "stop.head.weight"),
        ("stop_proj.bias", "stop.proj.bias"),
    ];
    for (ours, theirs) in cases {
        assert_eq!(hf_to_ggml_name(ours).as_deref(), Some(theirs), "for {ours}");
    }
}

/// An unknown name maps to nothing. The caller turns that into an error
/// naming the key, rather than passing a HuggingFace name to a file that has
/// none.
#[test]
fn unknown_names_map_to_nothing() {
    for name in [
        "vae.enc.conv0.weight",
        "base_lm.layers.0.self_attn.q_proj.bias",
        "base_lm.layers.foo.mlp.up_proj.weight",
        "base_lm.layers.",
        "lm_head.weight",
    ] {
        assert!(hf_to_ggml_name(name).is_none(), "{name} mapped");
    }
}

/// The one squeezed tensor: `[1024]` in their file, `[1, 1, 1, 1024]` here.
#[test]
fn restores_the_squeezed_cls_token() {
    assert_eq!(
        restore_leading_unit_dims("feat_encoder.special_token", &[1024]),
        Some(vec![1, 1, 1, 1024])
    );
    // Partially squeezed is restored the same way.
    assert_eq!(
        restore_leading_unit_dims("feat_encoder.special_token", &[1, 1024]),
        Some(vec![1, 1, 1, 1024])
    );
}

/// A shape that already has the rank we demand is left alone, whatever it
/// holds — this rule restores leading `1`s, it does not re-lay-out tensors.
#[test]
fn leaves_a_full_rank_shape_alone() {
    for shape in [vec![1, 1, 1, 1024], vec![2, 2, 2, 128]] {
        assert_eq!(
            restore_leading_unit_dims("feat_encoder.special_token", &shape),
            None,
            "for {shape:?}"
        );
    }
}

/// The bright line: same element count, genuinely different layout, still
/// REJECTED. `[2, 512]` has 1024 elements like `[1, 1, 1, 1024]` does, and
/// a general same-numel reshape would happily accept it and hide the
/// disagreement.
#[test]
fn rejects_a_same_numel_but_different_layout() {
    assert_eq!(
        restore_leading_unit_dims("feat_encoder.special_token", &[2, 512]),
        None
    );
}

/// No other tensor is in the squeeze table, so no other tensor is reshaped.
#[test]
fn reshapes_nothing_outside_the_table() {
    assert_eq!(
        restore_leading_unit_dims("base_lm.norm.weight", &[2048]),
        None
    );
}

/// Build a valid GGUF v3 byte image holding exactly the named tensors, each
/// an empty F32 tensor. Enough for the probe, which reads only the index.
fn gguf_with_tensors(names: &[&str]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes()); // version
    out.extend_from_slice(&(names.len() as u64).to_le_bytes()); // tensor count
    out.extend_from_slice(&0u64.to_le_bytes()); // metadata kv count
    for name in names {
        out.extend_from_slice(&(name.len() as u64).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        out.extend_from_slice(&4u64.to_le_bytes()); // dim 0
        out.extend_from_slice(&0u32.to_le_bytes()); // GGML type F32
        out.extend_from_slice(&0u64.to_le_bytes()); // data offset
    }
    out
}

fn open_with(names: &[&str]) -> Gguf {
    Gguf::from_bytes(gguf_with_tensors(names)).expect("parse synthetic GGUF")
}

/// cstr's sentinel picks the ggml-conventional path; ours picks the verbatim
/// one. `general.architecture` is absent from both images on purpose: it is
/// `voxcpm2` on both real files and therefore cannot decide this.
#[test]
fn probe_picks_the_convention_by_sentinel() {
    let ggml = open_with(&[GGML_SENTINEL, "tslm.blk.0.attn_q.weight"]);
    assert!(matches!(
        probe_naming(&ggml).expect("probe"),
        GgufNaming::Ggml
    ));

    let verbatim = open_with(&[
        VERBATIM_SENTINEL,
        "base_lm.layers.0.self_attn.q_proj.weight",
    ]);
    assert!(matches!(
        probe_naming(&verbatim).expect("probe"),
        GgufNaming::Verbatim
    ));
}

/// Neither sentinel: the error names both, and shows what the file actually
/// holds, so the operator can tell "wrong model" from "unknown convention".
#[test]
fn probe_errors_naming_both_sentinels() {
    let other = open_with(&["blk.0.attn_q.weight", "token_embd.weight"]);
    let err = probe_naming(&other).expect_err("neither sentinel present");
    let message = err.to_string();
    assert!(message.contains(GGML_SENTINEL), "got {message}");
    assert!(message.contains(VERBATIM_SENTINEL), "got {message}");
    assert!(message.contains("token_embd.weight"), "got {message}");
}
