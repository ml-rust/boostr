//! Tests for the hybrid attention wiring of ALiBi and sliding-window attention.

use super::blocks::AttentionBlock;
use super::model::{HybridBlock, HybridModel};
use crate::model::config::{AttentionConfig, HybridConfig, SsmConfig, UniversalConfig};
use crate::nn::{Linear, RmsNorm, RoPE, VarBuilder, VarMap};
use crate::test_utils::cpu_setup;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

const NEG: f32 = f32::MIN;

// ── Mask shape and contents ─────────────────────────────────────────

/// Reference mask: `0.0` where query `i` may attend key `j`, `f32::MIN` else.
///
/// Query row `i` sits at absolute position `sk - sq + i`; `window_size == 0`
/// means unlimited, matching the kernel contract.
fn expected_mask(sq: usize, sk: usize, window_size: usize) -> Vec<f32> {
    let offset = sk - sq;
    let mut out = Vec::with_capacity(sq * sk);
    for i in 0..sq {
        let pos = offset + i;
        for j in 0..sk {
            let future = j > pos;
            let too_old = window_size > 0 && j + window_size <= pos;
            out.push(if future || too_old { NEG } else { 0.0 });
        }
    }
    out
}

/// An attention block carrying only the two flags under test. The projections
/// are never touched by `attention_mask`, so they are left minimal.
fn flag_block(use_alibi: bool, sliding_window: usize) -> AttentionBlock<CpuRuntime> {
    let (_, device) = cpu_setup();
    let w = || Tensor::<CpuRuntime>::zeros(&[4, 4], DType::F32, &device).unwrap();
    let n = || Tensor::<CpuRuntime>::zeros(&[4], DType::F32, &device).unwrap();
    AttentionBlock {
        input_layernorm: RmsNorm::new(n(), 1e-5, false),
        q_proj: Linear::new(w(), None, false),
        k_proj: Linear::new(w(), None, false),
        v_proj: Linear::new(w(), None, false),
        o_proj: Linear::new(w(), None, false),
        post_attention_layernorm: RmsNorm::new(n(), 1e-5, false),
        gate_proj: Linear::new(w(), None, false),
        up_proj: Linear::new(w(), None, false),
        down_proj: Linear::new(w(), None, false),
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 2,
        use_alibi,
        sliding_window,
    }
}

fn mask_values(window_size: usize, sq: usize, sk: usize, position: usize) -> Vec<f32> {
    let (client, device) = cpu_setup();
    let block = flag_block(false, window_size);
    let mask = block
        .attention_mask(&client, 1, sq, sk, position, &device)
        .unwrap()
        .expect("a positive window must produce a mask");
    assert_eq!(mask.shape(), &[1, 1, sq, sk]);
    mask.tensor().to_vec::<f32>()
}

#[test]
fn windowed_prefill_mask_has_the_exact_expected_pattern() {
    let actual = mask_values(3, 6, 6, 0);
    assert_eq!(actual, expected_mask(6, 6, 3));

    // Spelled out for the record: row `i` keeps exactly `j ∈ (i - 3, i]`.
    #[rustfmt::skip]
    let literal = vec![
        0.0, NEG, NEG, NEG, NEG, NEG,
        0.0, 0.0, NEG, NEG, NEG, NEG,
        0.0, 0.0, 0.0, NEG, NEG, NEG,
        NEG, 0.0, 0.0, 0.0, NEG, NEG,
        NEG, NEG, 0.0, 0.0, 0.0, NEG,
        NEG, NEG, NEG, 0.0, 0.0, 0.0,
    ];
    assert_eq!(actual, literal);
}

#[test]
fn windowed_mask_never_overflows_to_negative_infinity() {
    // `triu` and `tril` regions must stay disjoint: an overlap would sum
    // `f32::MIN + f32::MIN` and overflow to `-inf`.
    for window_size in 1..=5 {
        for value in mask_values(window_size, 6, 6, 0) {
            assert!(value.is_finite(), "window {window_size} produced {value}");
        }
    }
}

#[test]
fn window_wider_than_the_sequence_is_pure_causal() {
    assert_eq!(mask_values(10, 4, 4, 0), expected_mask(4, 4, 0));
    assert_eq!(mask_values(4, 4, 4, 0), expected_mask(4, 4, 0));
}

#[test]
fn decode_mask_rows_are_offset_by_the_cached_key_count() {
    // One new query against four cached keys: the query is at absolute
    // position 3, so a window of 2 keeps keys 2 and 3 only.
    assert_eq!(mask_values(2, 1, 4, 3), vec![NEG, NEG, 0.0, 0.0]);
    assert_eq!(mask_values(2, 1, 4, 3), expected_mask(1, 4, 2));

    // Two new queries against five cached keys, window 3.
    assert_eq!(mask_values(3, 2, 5, 3), expected_mask(2, 5, 3));
}

#[test]
fn alibi_mask_covers_every_batch_and_head() {
    let (client, device) = cpu_setup();
    let block = flag_block(true, 0);
    let mask = block
        .attention_mask(&client, 2, 3, 3, 0, &device)
        .unwrap()
        .expect("ALiBi always produces a bias mask");
    assert_eq!(mask.shape(), &[2, 2, 3, 3]);
}

#[test]
fn alibi_ignores_the_sliding_window() {
    // ALiBi's kernel writes causality together with the distance bias; the two
    // mechanisms do not compose, so the window must not change the result.
    let (client, device) = cpu_setup();
    let unwindowed = flag_block(true, 0)
        .attention_mask(&client, 1, 4, 4, 0, &device)
        .unwrap()
        .expect("ALiBi always produces a bias mask");
    let windowed = flag_block(true, 2)
        .attention_mask(&client, 1, 4, 4, 0, &device)
        .unwrap()
        .expect("ALiBi always produces a bias mask");
    assert_eq!(
        unwindowed.tensor().to_vec::<f32>(),
        windowed.tensor().to_vec::<f32>()
    );
}

#[test]
fn plain_attention_is_still_causal() {
    // Regression guard for a REAL bug this replaced: the unwindowed, non-ALiBi
    // branch used to return `None`, so hybrid prefill attended to FUTURE
    // tokens. Shape checks cannot see it and the model still emits fluent
    // text. Prefill must be causal; decode must see every cached key.
    let (client, device) = cpu_setup();
    let block = flag_block(false, 0);

    // Prefill: strictly-upper triangle masked, diagonal and below visible.
    let mask = block
        .attention_mask(&client, 1, 4, 4, 0, &device)
        .unwrap()
        .expect("prefill must be masked, or attention sees the future");
    let values = mask.tensor().to_vec::<f32>();
    for i in 0..4 {
        for j in 0..4 {
            let got = values[i * 4 + j];
            if j > i {
                assert_eq!(got, f32::MIN, "future key ({i},{j}) must be masked");
            } else {
                assert_eq!(got, 0.0, "past/self key ({i},{j}) must be visible");
            }
        }
    }

    // Decode: one query against five cached keys — all are in the past.
    let mask = block
        .attention_mask(&client, 1, 1, 6, 5, &device)
        .unwrap()
        .expect("decode is masked too");
    assert!(
        mask.tensor().to_vec::<f32>().iter().all(|v| *v == 0.0),
        "every cached key precedes the query, so none may be masked"
    );
}

// ── RoPE is skipped for ALiBi blocks ────────────────────────────────

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32 % 7.0) * 0.25 - 0.75).collect()
}

/// A block with deterministic non-zero weights, so a change in the rotary
/// frequencies actually moves the output.
fn rope_probe_block(use_alibi: bool) -> AttentionBlock<CpuRuntime> {
    let (_, device) = cpu_setup();
    let w = || Tensor::<CpuRuntime>::from_slice(&ramp(64), &[8, 8], &device).unwrap();
    let n = || Tensor::<CpuRuntime>::from_slice(&[1.0f32; 8], &[8], &device).unwrap();
    AttentionBlock {
        input_layernorm: RmsNorm::new(n(), 1e-5, false),
        q_proj: Linear::new(w(), None, false),
        k_proj: Linear::new(w(), None, false),
        v_proj: Linear::new(w(), None, false),
        o_proj: Linear::new(w(), None, false),
        post_attention_layernorm: RmsNorm::new(n(), 1e-5, false),
        gate_proj: Linear::new(w(), None, false),
        up_proj: Linear::new(w(), None, false),
        down_proj: Linear::new(w(), None, false),
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 4,
        use_alibi,
        sliding_window: 0,
    }
}

/// Run one prefill through a fresh KV cache with a RoPE cache built at `base`.
fn forward_with_rope_base(use_alibi: bool, base: f32) -> Vec<f32> {
    let (client, device) = cpu_setup();
    let block = rope_probe_block(use_alibi);
    let rope = RoPE::<CpuRuntime>::precompute_freqs(8, 4, base, None, &device).unwrap();
    let mut cache =
        crate::inference::KvCache::<CpuRuntime>::new(1, 2, 8, 8, 4, DType::F32, &device).unwrap();
    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&ramp(24), &[1, 3, 8], &device).unwrap(),
        false,
    );
    let out = block
        .forward_with_kv_cache(&client, &x, &rope, &mut cache, 0)
        .unwrap();
    out.tensor().to_vec::<f32>()
}

#[test]
fn alibi_blocks_skip_rope() {
    // The rotary frequencies depend on `base`, so an ALiBi block that still
    // applied RoPE would produce different outputs for different bases.
    assert_eq!(
        forward_with_rope_base(true, 10_000.0),
        forward_with_rope_base(true, 100.0)
    );
}

#[test]
fn non_alibi_blocks_still_apply_rope() {
    // Guards the test above against passing for the wrong reason.
    assert_ne!(
        forward_with_rope_base(false, 10_000.0),
        forward_with_rope_base(false, 100.0)
    );
}

// ── Builder wiring ──────────────────────────────────────────────────

fn hybrid_config(sliding_window: Option<usize>, use_alibi: bool) -> UniversalConfig {
    UniversalConfig {
        model_type: "hybrid".into(),
        vocab_size: 16,
        hidden_size: 8,
        num_layers: 2,
        max_seq_len: 16,
        intermediate_size: Some(16),
        rms_norm_eps: 1e-5,
        attention: Some(AttentionConfig {
            num_heads: 2,
            num_kv_heads: None,
            head_dim: None,
            kv_latent_dim: None,
            q_latent_dim: None,
            d_rope: None,
            rope_theta: 10000.0,
            rope_scaling: None,
            sliding_window,
            use_alibi,
        }),
        ssm: Some(SsmConfig {
            variant: "mamba2".into(),
            state_size: 4,
            num_heads: 2,
            head_dim: 8,
            expand: 2,
            conv_kernel: 4,
            chunk_size: 4,
            n_groups: 1,
            complex_rope: None,
            mimo_rank: None,
            use_conv: None,
        }),
        moe: None,
        hybrid_layers: Some(HybridConfig {
            ssm_layers: vec![0],
            attention_layers: vec![1],
        }),
        tie_word_embeddings: true,
        grow_vocab: false,
        vision: None,
        audio: None,
    }
}

/// Weight names and shapes matching [`hybrid_config`]: layer 0 is Mamba2,
/// layer 1 is attention.
fn weight_shapes() -> Vec<(&'static str, Vec<usize>)> {
    vec![
        ("model.embed_tokens.weight", vec![16, 8]),
        ("model.layers.0.input_layernorm.weight", vec![8]),
        ("model.layers.0.mixer.in_proj.weight", vec![42, 8]),
        ("model.layers.0.mixer.conv1d.weight", vec![24, 1, 4]),
        ("model.layers.0.mixer.out_proj.weight", vec![8, 16]),
        ("model.layers.0.mixer.A_log", vec![2]),
        ("model.layers.0.mixer.dt_bias", vec![2]),
        ("model.layers.0.mixer.D", vec![2]),
        ("model.layers.1.input_layernorm.weight", vec![8]),
        ("model.layers.1.self_attn.q_proj.weight", vec![8, 8]),
        ("model.layers.1.self_attn.k_proj.weight", vec![8, 8]),
        ("model.layers.1.self_attn.v_proj.weight", vec![8, 8]),
        ("model.layers.1.self_attn.o_proj.weight", vec![8, 8]),
        ("model.layers.1.post_attention_layernorm.weight", vec![8]),
        ("model.layers.1.mlp.gate_proj.weight", vec![16, 8]),
        ("model.layers.1.mlp.up_proj.weight", vec![16, 8]),
        ("model.layers.1.mlp.down_proj.weight", vec![8, 16]),
        ("model.norm.weight", vec![8]),
    ]
}

/// `(sliding_window, use_alibi)` as carried by the model's attention layers.
fn attention_flags(sliding_window: Option<usize>, use_alibi: bool) -> (usize, bool) {
    let (_, device) = cpu_setup();
    let config = hybrid_config(sliding_window, use_alibi);
    let mut varmap = VarMap::<CpuRuntime>::new();
    for (name, shape) in weight_shapes() {
        varmap.insert(
            name.into(),
            Tensor::<CpuRuntime>::zeros(&shape, DType::F32, &device).unwrap(),
        );
    }
    let mut vb = VarBuilder::new(&mut varmap, &device);
    let model = HybridModel::<CpuRuntime>::from_varbuilder(&mut vb, &config).unwrap();
    let attn = model
        .blocks
        .iter()
        .find_map(|b| match b {
            HybridBlock::Attention(a) => Some(a),
            HybridBlock::Ssm(_) => None,
        })
        .expect("the config places an attention layer at index 1");
    (attn.sliding_window, attn.use_alibi)
}

#[test]
fn builder_reads_sliding_window_from_config() {
    assert_eq!(attention_flags(Some(64), false).0, 64);
    assert_eq!(attention_flags(Some(1), false).0, 1);
}

#[test]
fn absent_sliding_window_is_disabled() {
    assert_eq!(attention_flags(None, false).0, 0);
}

#[test]
fn explicit_zero_sliding_window_is_disabled() {
    // `Some(0)` is not a zero-width window — it maps to the disabled sentinel.
    assert_eq!(attention_flags(Some(0), false).0, 0);
}

#[test]
fn builder_reads_use_alibi_from_config() {
    assert!(attention_flags(None, true).1);
    assert!(!attention_flags(None, false).1);
}

#[test]
fn default_config_leaves_both_features_off() {
    assert_eq!(attention_flags(None, false), (0, false));
}
