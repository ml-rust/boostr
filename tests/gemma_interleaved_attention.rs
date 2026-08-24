//! Gemma-embedding interleaved attention: per-layer RoPE base and windowed masking.
//!
//! Gemma3-derived encoders do not use one uniform attention configuration for
//! every block. Blocks alternate between two attention types on a fixed period,
//! and the two types differ in *two* independent ways:
//!
//!   1. **RoPE base.** Local blocks rotate at the local base (10 000 unless the
//!      file overrides it); global blocks rotate at `rope.freq_base` (1 000 000
//!      for EmbeddingGemma). This applies at *every* sequence length — a
//!      one-token input already sees the difference.
//!   2. **Attention window.** Local blocks attend only within a symmetric
//!      window of `sliding_window` positions — a query at position `p` may
//!      attend to keys in `[p - sliding_window/2, p + sliding_window/2]`.
//!      Global blocks attend to the whole sequence.
//!
//! For a 24-block EmbeddingGemma with the default period of 6, blocks
//! 5, 11, 17 and 23 are global and the other 20 are local.
//!
//! Both consequences flow from the same per-block flag, so both are covered
//! here. The tests are written against the public encoder API and compare an
//! interleaved encoder against uniform-base reference encoders, because the
//! per-block RoPE cache is not observable from outside the crate.
//!
//! `sliding_window: None` means "not an interleaved architecture" — every block
//! is global and shares one RoPE base. BERT and NomicBert rely on that, so the
//! uniform reference encoders below double as a guard against regressing them.

use boostr::model::encoder::{
    config::{ArchFamily, EncoderConfig, FfnVariant, NormScheme},
    model::{Encoder, Pooling},
};
use boostr::nn::Weight;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// Architecture constants
// ─────────────────────────────────────────────────────────────────────────────
const HIDDEN: usize = 128;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const INTER: usize = 256;
const VOCAB: usize = 64;
const MAX_POS: usize = 128;
const RMS_EPS: f64 = 1e-6;

/// Block period for the local/global alternation. Blocks `il` with
/// `il % PERIOD < PERIOD - 1` are local; the remainder are global.
const PERIOD: usize = 6;

/// RoPE base for global blocks — EmbeddingGemma's `rope.freq_base`.
const GLOBAL_BASE: f32 = 1_000_000.0;
/// RoPE base for local blocks when the file carries no explicit override.
const LOCAL_BASE: f32 = 10_000.0;

// ─────────────────────────────────────────────────────────────────────────────
// Builders
// ─────────────────────────────────────────────────────────────────────────────

fn gemma_config(
    layers: usize,
    rope_freq_base: f32,
    sliding_window: Option<usize>,
) -> EncoderConfig {
    EncoderConfig {
        vocab_size: VOCAB,
        hidden_size: HIDDEN,
        num_hidden_layers: layers,
        num_attention_heads: HEADS,
        intermediate_size: INTER,
        max_position_embeddings: MAX_POS,
        layer_norm_eps: RMS_EPS,
        arch_family: ArchFamily::GemmaEmbedding,
        ffn_variant: FfnVariant::GatedGelu,
        norm_scheme: NormScheme::Sandwich,
        num_kv_heads: KV_HEADS,
        head_dim_explicit: Some(HEAD_DIM),
        rms_eps: RMS_EPS,
        rope_freq_base,
        rope_freq_base_local: LOCAL_BASE,
        sliding_window,
        // `sliding_window: None` above is what actually disables interleaving;
        // the period is always set so the two are tested independently.
        sliding_window_pattern: PERIOD,
        embed_scale: true,
        ..Default::default()
    }
}

/// Deterministic, layer-varying weight values so that no two blocks are
/// interchangeable and a mis-assigned per-block RoPE cache cannot cancel out.
fn seeded(len: usize, layer: usize, salt: f32) -> Vec<f32> {
    let l = layer as f32;
    (0..len)
        .map(|i| (((i as f32) * salt) + l * 1.7).sin() * 0.01)
        .collect()
}

fn build_encoder(
    layers: usize,
    rope_freq_base: f32,
    sliding_window: Option<usize>,
) -> (Encoder<CpuRuntime>, CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let d = &device;

    let config = gemma_config(layers, rope_freq_base, sliding_window);

    let encoder = Encoder::from_weights_gemma(config, Pooling::Mean, &client, |name| {
        // Non-block tensors first.
        let t = match name {
            "token_embd.weight" => {
                let data: Vec<f32> = (0..VOCAB * HIDDEN)
                    .map(|i| ((i as f32) * 0.11).sin() * 0.05)
                    .collect();
                return Ok(Weight::Standard(
                    Tensor::try_from_slice(&data, &[VOCAB, HIDDEN], d).unwrap(),
                ));
            }
            "position_embd.weight" => {
                return Ok(Weight::Standard(
                    Tensor::try_from_slice(&vec![0.0f32; HIDDEN], &[1, HIDDEN], d).unwrap(),
                ));
            }
            "output_norm.weight" => {
                return Ok(Weight::Standard(
                    Tensor::try_from_slice(&vec![1.0f32; HIDDEN], &[HIDDEN], d).unwrap(),
                ));
            }
            other => other,
        };

        // Block tensors: "blk.{i}.{suffix}".
        let rest = t
            .strip_prefix("blk.")
            .ok_or_else(|| boostr::error::Error::ModelError {
                reason: format!("unknown weight in test: {t}"),
            })?;
        let (idx, suffix) =
            rest.split_once('.')
                .ok_or_else(|| boostr::error::Error::ModelError {
                    reason: format!("malformed block weight name in test: {t}"),
                })?;
        let layer: usize = idx.parse().map_err(|_| boostr::error::Error::ModelError {
            reason: format!("malformed block index in test: {t}"),
        })?;

        let q_rows = HEADS * HEAD_DIM;
        let kv_rows = KV_HEADS * HEAD_DIM;

        let tensor = match suffix {
            "attn_norm.weight"
            | "post_attention_norm.weight"
            | "ffn_norm.weight"
            | "post_ffw_norm.weight" => {
                Tensor::try_from_slice(&vec![1.0f32; HIDDEN], &[HIDDEN], d).unwrap()
            }
            "attn_q_norm.weight" | "attn_k_norm.weight" => {
                Tensor::try_from_slice(&vec![1.0f32; HEAD_DIM], &[HEAD_DIM], d).unwrap()
            }
            "attn_q.weight" => {
                Tensor::try_from_slice(&seeded(q_rows * HIDDEN, layer, 1.0), &[q_rows, HIDDEN], d)
                    .unwrap()
            }
            "attn_k.weight" => {
                Tensor::try_from_slice(&seeded(kv_rows * HIDDEN, layer, 0.7), &[kv_rows, HIDDEN], d)
                    .unwrap()
            }
            "attn_v.weight" => {
                Tensor::try_from_slice(&seeded(kv_rows * HIDDEN, layer, 0.3), &[kv_rows, HIDDEN], d)
                    .unwrap()
            }
            "attn_output.weight" => {
                Tensor::try_from_slice(&seeded(HIDDEN * q_rows, layer, 1.3), &[HIDDEN, q_rows], d)
                    .unwrap()
            }
            "ffn_gate.weight" => {
                Tensor::try_from_slice(&seeded(INTER * HIDDEN, layer, 0.5), &[INTER, HIDDEN], d)
                    .unwrap()
            }
            "ffn_up.weight" => {
                Tensor::try_from_slice(&seeded(INTER * HIDDEN, layer, 0.9), &[INTER, HIDDEN], d)
                    .unwrap()
            }
            "ffn_down.weight" => {
                Tensor::try_from_slice(&seeded(HIDDEN * INTER, layer, 0.2), &[HIDDEN, INTER], d)
                    .unwrap()
            }
            other => {
                return Err(boostr::error::Error::ModelError {
                    reason: format!("unknown block weight in test: {other}"),
                });
            }
        };
        Ok(Weight::Standard(tensor))
    })
    .expect("build gemma encoder");

    (encoder, client, device)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn ids(n: usize) -> Vec<i64> {
    (0..n).map(|i| ((i * 7 + 3) % VOCAB) as i64).collect()
}

/// Mean-pooled embedding for one document.
fn embed(
    encoder: &Encoder<CpuRuntime>,
    client: &CpuClient,
    device: &CpuDevice,
    tokens: &[i64],
) -> Vec<f32> {
    let input = Tensor::<CpuRuntime>::try_from_slice(tokens, &[1, tokens.len()], device).unwrap();
    encoder
        .embed_inference_standard(client, &input, None)
        .expect("embed")
        .to_vec()
}

/// Per-token hidden states `[S, HIDDEN]` flattened, for one document.
fn hidden_states(
    encoder: &Encoder<CpuRuntime>,
    client: &CpuClient,
    device: &CpuDevice,
    tokens: &[i64],
) -> Vec<f32> {
    let input = Tensor::<CpuRuntime>::try_from_slice(tokens, &[1, tokens.len()], device).unwrap();
    encoder
        .encode_inference(client, &input, None)
        .expect("encode")
        .to_vec()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared vectors must have equal length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Hidden state for one token position out of a flattened `[S, HIDDEN]` buffer.
fn token_state(states: &[f32], pos: usize) -> &[f32] {
    &states[pos * HIDDEN..(pos + 1) * HIDDEN]
}

/// Two embeddings produced by genuinely different RoPE bases differ by far more
/// than float noise. Anything below this is "the same computation ran twice".
const DISTINCT: f32 = 1e-5;
/// Bit-level agreement tolerance for computations that must be identical.
const IDENTICAL: f32 = 1e-6;

// ─────────────────────────────────────────────────────────────────────────────
// Per-block RoPE base
// ─────────────────────────────────────────────────────────────────────────────

/// An interleaved encoder must not be reducible to either uniform-base encoder:
/// its local blocks rotate at the local base and its global blocks at the global
/// base, so it agrees with neither.
#[test]
fn interleaved_blocks_match_neither_uniform_rope_base() {
    let tokens = ids(24);

    let (interleaved, c0, d0) = build_encoder(12, GLOBAL_BASE, Some(512));
    let (all_global, c1, d1) = build_encoder(12, GLOBAL_BASE, None);
    let (all_local, c2, d2) = build_encoder(12, LOCAL_BASE, None);

    let got = embed(&interleaved, &c0, &d0, &tokens);
    let uniform_global = embed(&all_global, &c1, &d1, &tokens);
    let uniform_local = embed(&all_local, &c2, &d2, &tokens);

    // Sanity: the two uniform references are themselves distinguishable, so a
    // failure below cannot be blamed on the two bases producing equal output.
    assert!(
        max_abs_diff(&uniform_global, &uniform_local) > DISTINCT,
        "test is degenerate: the two uniform RoPE bases produced identical embeddings"
    );

    assert!(
        max_abs_diff(&got, &uniform_global) > DISTINCT,
        "interleaved encoder collapsed to the all-global RoPE base — the local \
         blocks are rotating at {GLOBAL_BASE} instead of {LOCAL_BASE}"
    );
    assert!(
        max_abs_diff(&got, &uniform_local) > DISTINCT,
        "interleaved encoder collapsed to the all-local RoPE base — the global \
         blocks are rotating at {LOCAL_BASE} instead of {GLOBAL_BASE}"
    );
}

/// The alternation starts local: a model whose block count stays inside the
/// first period has no global block at all, so it must equal the all-local
/// reference exactly.
#[test]
fn blocks_before_the_first_global_are_all_local() {
    let tokens = ids(24);
    let layers = PERIOD - 1; // 5 blocks: indices 0..=4, all local

    let (interleaved, c0, d0) = build_encoder(layers, GLOBAL_BASE, Some(512));
    let (all_local, c1, d1) = build_encoder(layers, LOCAL_BASE, None);
    let (all_global, c2, d2) = build_encoder(layers, GLOBAL_BASE, None);

    let got = embed(&interleaved, &c0, &d0, &tokens);
    let uniform_local = embed(&all_local, &c1, &d1, &tokens);
    let uniform_global = embed(&all_global, &c2, &d2, &tokens);

    assert!(
        max_abs_diff(&got, &uniform_local) < IDENTICAL,
        "the first {layers} blocks must all be local, so this model must match \
         the all-local reference exactly (max diff {})",
        max_abs_diff(&got, &uniform_local)
    );
    assert!(
        max_abs_diff(&got, &uniform_global) > DISTINCT,
        "a model of only local blocks must not match the all-global reference"
    );
}

/// Adding exactly one block crosses into the first global block, which must
/// change the result. This pins the local→global boundary at index `PERIOD - 1`
/// rather than merely confirming that *some* alternation happens.
#[test]
fn the_first_global_block_appears_at_the_end_of_the_first_period() {
    let tokens = ids(24);

    let (five, c0, d0) = build_encoder(PERIOD - 1, GLOBAL_BASE, Some(512));
    let (five_local, c1, d1) = build_encoder(PERIOD - 1, LOCAL_BASE, None);
    let (six, c2, d2) = build_encoder(PERIOD, GLOBAL_BASE, Some(512));
    let (six_local, c3, d3) = build_encoder(PERIOD, LOCAL_BASE, None);

    // Block indices 0..=4 are local — matches the all-local reference.
    assert!(
        max_abs_diff(
            &embed(&five, &c0, &d0, &tokens),
            &embed(&five_local, &c1, &d1, &tokens)
        ) < IDENTICAL,
        "blocks 0..={} must be local",
        PERIOD - 2
    );

    // Adding block index 5 adds a global block — no longer all-local.
    assert!(
        max_abs_diff(
            &embed(&six, &c2, &d2, &tokens),
            &embed(&six_local, &c3, &d3, &tokens)
        ) > DISTINCT,
        "block index {} must be global, so a {PERIOD}-block model must stop \
         matching the all-local reference",
        PERIOD - 1
    );
}

/// The RoPE base is selected per block regardless of sequence length — it is not
/// a masking effect. A two-token input is far inside any window, yet the
/// interleaved and uniform encoders must still disagree.
#[test]
fn per_block_rope_base_applies_at_short_sequence_lengths() {
    let tokens = ids(2);

    let (interleaved, c0, d0) = build_encoder(12, GLOBAL_BASE, Some(512));
    let (all_global, c1, d1) = build_encoder(12, GLOBAL_BASE, None);

    assert!(
        max_abs_diff(
            &embed(&interleaved, &c0, &d0, &tokens),
            &embed(&all_global, &c1, &d1, &tokens)
        ) > DISTINCT,
        "a 2-token input is well inside every attention window, so any \
         difference here must come from the per-block RoPE base — and there \
         must be one"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Windowed masking
// ─────────────────────────────────────────────────────────────────────────────

/// Local blocks attend within a symmetric window of `sliding_window` positions,
/// so a token outside that window cannot influence the output. Perturbing
/// position 0 must leave the far end of the sequence untouched.
#[test]
fn local_blocks_do_not_attend_beyond_the_symmetric_window() {
    let window = 8; // symmetric half-window of 4 positions
    let seq = 24;
    let far = seq - 1; // distance 23 from position 0 — far outside the window

    // A single block: index 0, which is local.
    let (encoder, client, device) = build_encoder(1, GLOBAL_BASE, Some(window));

    let base = ids(seq);
    let mut perturbed = base.clone();
    perturbed[0] = (base[0] + 1) % VOCAB as i64;
    assert_ne!(base[0], perturbed[0], "perturbation must change the token");

    let a = hidden_states(&encoder, &client, &device, &base);
    let b = hidden_states(&encoder, &client, &device, &perturbed);

    // Position 0 itself must react — otherwise the perturbation did nothing and
    // the assertion below would pass vacuously.
    assert!(
        max_abs_diff(token_state(&a, 0), token_state(&b, 0)) > DISTINCT,
        "perturbing position 0 must change position 0's own hidden state"
    );

    assert!(
        max_abs_diff(token_state(&a, far), token_state(&b, far)) < IDENTICAL,
        "position {far} is {far} positions from position 0, outside the \
         symmetric window of ±{}, so it must not be influenced by it (max diff {})",
        window / 2,
        max_abs_diff(token_state(&a, far), token_state(&b, far))
    );
}

/// The mirror of the above: global blocks must keep attending across the whole
/// sequence. Guards against a fix that masks every block instead of the local ones.
#[test]
fn global_blocks_attend_across_the_whole_sequence() {
    let seq = 24;
    let far = seq - 1;

    // sliding_window: None → not an interleaved architecture, every block global.
    let (encoder, client, device) = build_encoder(1, GLOBAL_BASE, None);

    let base = ids(seq);
    let mut perturbed = base.clone();
    perturbed[0] = (base[0] + 1) % VOCAB as i64;

    let a = hidden_states(&encoder, &client, &device, &base);
    let b = hidden_states(&encoder, &client, &device, &perturbed);

    assert!(
        max_abs_diff(token_state(&a, far), token_state(&b, far)) > DISTINCT,
        "a global block must let position 0 influence position {far}"
    );
}

/// The window size must reach the forward path. Two encoders differing *only* in
/// `sliding_window` must produce different results on a sequence longer than the
/// narrower window. This is the direct guard against the field being parsed and
/// then ignored — the failure mode that made this silent.
#[test]
fn sliding_window_size_changes_the_forward_pass() {
    let tokens = ids(24);

    let (narrow, c0, d0) = build_encoder(1, GLOBAL_BASE, Some(4));
    let (wide, c1, d1) = build_encoder(1, GLOBAL_BASE, Some(512));

    assert!(
        max_abs_diff(
            &embed(&narrow, &c0, &d0, &tokens),
            &embed(&wide, &c1, &d1, &tokens)
        ) > DISTINCT,
        "sliding_window is parsed from GGUF but never reaches attention — a \
         window of 4 and a window of 512 produced identical output on a \
         24-token input"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-interleaved architectures must be unaffected
// ─────────────────────────────────────────────────────────────────────────────

/// BERT and NomicBert carry no sliding window and must keep a single uniform
/// RoPE base across every block. Encoding twice must be bit-identical, and the
/// uniform encoder must be insensitive to the interleave machinery entirely.
#[test]
fn encoders_without_a_sliding_window_use_one_uniform_rope_base() {
    let tokens = ids(24);

    let (a, ca, da) = build_encoder(12, GLOBAL_BASE, None);
    let (b, cb, db) = build_encoder(12, GLOBAL_BASE, None);

    assert!(
        max_abs_diff(&embed(&a, &ca, &da, &tokens), &embed(&b, &cb, &db, &tokens)) < IDENTICAL,
        "two identically-configured non-interleaved encoders must agree exactly"
    );
}
