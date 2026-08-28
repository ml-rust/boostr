//! Split out of `model.rs` to keep that file under the crate's 500-line hard
//! limit for model-architecture files after `MiniCpm4Model::apply_lora` was
//! added. `use super::*;` below reaches every item `model.rs` itself
//! imported, exactly as if this module were still inline.
//!
//! The module path stays `minicpm4::model::tests` (this file is declared
//! `pub(crate) mod tests;` in `model.rs`, unchanged) — other tests in this
//! crate reach `tiny_model`/`tiny_nope_model`/`HIDDEN` through that exact
//! path (`minicpm4/decode/tests.rs`,
//! `model/generate/tests/support.rs`), so nothing there needed to change.

use super::*;
use crate::model::audio::voxcpm::minicpm4::attention::MiniCpm4Attention;
use crate::model::audio::voxcpm::minicpm4::mlp::MiniCpm4Mlp;
use crate::nn::{LoraTargets, MaybeLoraLinear, MaybeQuantLinear, Weight};
use crate::test_utils::cpu_setup;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

pub(crate) const HIDDEN: usize = 8;
const NUM_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 1;
const HEAD_DIM: usize = 4;
const FFN: usize = 16;
const NUM_LAYERS: usize = 2;

/// Deterministic, non-degenerate weights: zeros would make every
/// causality/shape assertion below pass vacuously.
pub(crate) fn filled(shape: &[usize], salt: usize, device: &CpuDevice) -> Tensor<CpuRuntime> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| (((i * 37 + salt * 11) % 13) as f32 - 6.0) / 20.0)
        .collect();
    Tensor::<CpuRuntime>::from_slice(&data, shape, device).expect("weights")
}

fn linear(
    out: usize,
    in_dim: usize,
    salt: usize,
    device: &CpuDevice,
) -> MaybeLoraLinear<CpuRuntime> {
    MaybeQuantLinear::from_weight(Weight::Standard(filled(&[out, in_dim], salt, device)), None)
        .into()
}

/// `base_lm`-shaped tiny model: rotary, exactly as before `no_rope`
/// existed.
pub(crate) fn tiny_model(device: &CpuDevice) -> MiniCpm4Model<CpuRuntime> {
    tiny_model_with(device, false)
}

/// `residual_lm`-shaped tiny model: NoPE, no RoPE table at all.
pub(crate) fn tiny_nope_model(device: &CpuDevice) -> MiniCpm4Model<CpuRuntime> {
    tiny_model_with(device, true)
}

/// Same weights either way, so any output difference between the two is
/// the rotation and nothing else.
fn tiny_model_with(device: &CpuDevice, no_rope: bool) -> MiniCpm4Model<CpuRuntime> {
    let q_dim = NUM_HEADS * HEAD_DIM;
    let kv_dim = NUM_KV_HEADS * HEAD_DIM;
    let layers = (0..NUM_LAYERS)
        .map(|i| MiniCpm4Layer {
            input_layernorm: RmsNorm::new(
                Tensor::<CpuRuntime>::ones(&[HIDDEN], DType::F32, device).expect("norm"),
                1e-5,
                false,
            ),
            self_attn: MiniCpm4Attention {
                q_proj: linear(q_dim, HIDDEN, i * 8 + 1, device),
                k_proj: linear(kv_dim, HIDDEN, i * 8 + 2, device),
                v_proj: linear(kv_dim, HIDDEN, i * 8 + 3, device),
                o_proj: linear(HIDDEN, q_dim, i * 8 + 4, device),
                num_heads: NUM_HEADS,
                num_kv_heads: NUM_KV_HEADS,
                head_dim: HEAD_DIM,
                no_rope,
            },
            post_attention_layernorm: RmsNorm::new(
                Tensor::<CpuRuntime>::ones(&[HIDDEN], DType::F32, device).expect("norm"),
                1e-5,
                false,
            ),
            mlp: MiniCpm4Mlp {
                gate_proj: linear(FFN, HIDDEN, i * 8 + 5, device),
                up_proj: linear(FFN, HIDDEN, i * 8 + 6, device),
                down_proj: linear(HIDDEN, FFN, i * 8 + 7, device),
            },
        })
        .collect();
    MiniCpm4Model {
        embed_tokens: None,
        layers,
        norm: RmsNorm::new(
            Tensor::<CpuRuntime>::ones(&[HIDDEN], DType::F32, device).expect("norm"),
            1e-5,
            false,
        ),
        rope: (!no_rope).then(|| {
            RoPE::<CpuRuntime>::precompute_freqs(16, HEAD_DIM, 10000.0, None, device).expect("rope")
        }),
        hidden_size: HIDDEN,
        activation_checkpointing: false,
    }
}

/// Q4_0's block size — every dimension below is a multiple of it so a
/// single block quantizes cleanly (`QuantFormat` requires the logical
/// element count be a multiple of `block_size`).
const QDIM: usize = 32;

/// A `MiniCpm4Model` built ENTIRELY from `MaybeQuantLinear::Quantized`
/// projections (via `client.quantize`, the same helper
/// `local_dit/tests.rs::quantized_projections_contribute_nothing_dense_norms_still_appear`
/// uses) — the checkpoint shape that made QLoRA validation reject a valid
/// target (see `apply_lora_adapts_quantized_projections_instead_of_rejecting_them`).
/// `num_heads`/`num_kv_heads` are both 1 so every projection is a plain
/// `QDIM -> QDIM` square, matching `quantized_linear`'s fixed shape.
fn quantized_tiny_model(device: &CpuDevice, client: &CpuClient) -> MiniCpm4Model<CpuRuntime> {
    use crate::nn::QuantLinear;
    use crate::quant::format::QuantFormat;
    use crate::quant::traits::QuantizeOps;

    let quantized_linear = |seed: f32| -> MaybeLoraLinear<CpuRuntime> {
        let data: Vec<f32> = (0..QDIM * QDIM)
            .map(|i| (i as f32 * 0.01 + seed).sin())
            .collect();
        let w = Tensor::<CpuRuntime>::from_slice(&data, &[QDIM, QDIM], device).unwrap();
        let qt = client.quantize(&w, QuantFormat::Q4_0).unwrap();
        MaybeQuantLinear::Quantized(QuantLinear::new(qt, None)).into()
    };
    let quantized_norm = || {
        RmsNorm::new(
            Tensor::<CpuRuntime>::ones(&[QDIM], DType::F32, device).expect("norm"),
            1e-5,
            false,
        )
    };
    let layers: Vec<MiniCpm4Layer<CpuRuntime>> = (0..NUM_LAYERS)
        .map(|i| MiniCpm4Layer {
            input_layernorm: quantized_norm(),
            self_attn: MiniCpm4Attention {
                q_proj: quantized_linear(i as f32 * 8.0 + 1.0),
                k_proj: quantized_linear(i as f32 * 8.0 + 2.0),
                v_proj: quantized_linear(i as f32 * 8.0 + 3.0),
                o_proj: quantized_linear(i as f32 * 8.0 + 4.0),
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: QDIM,
                no_rope: true,
            },
            post_attention_layernorm: quantized_norm(),
            mlp: MiniCpm4Mlp {
                gate_proj: quantized_linear(i as f32 * 8.0 + 5.0),
                up_proj: quantized_linear(i as f32 * 8.0 + 6.0),
                down_proj: quantized_linear(i as f32 * 8.0 + 7.0),
            },
        })
        .collect();
    MiniCpm4Model {
        embed_tokens: None,
        layers,
        norm: quantized_norm(),
        rope: None,
        hidden_size: QDIM,
        activation_checkpointing: false,
    }
}

fn out_values(v: &Var<CpuRuntime>) -> Vec<f32> {
    v.tensor().contiguous().expect("contiguous").to_vec::<f32>()
}

#[test]
fn forward_preserves_shape() {
    let (client, device) = cpu_setup();
    let model = tiny_model(&device);
    let x = Var::new(filled(&[1, 4, HIDDEN], 99, &device), false);
    let out = model.forward(&client, &x).expect("forward");
    assert_eq!(out.shape(), &[1, 4, HIDDEN]);
    assert!(out_values(&out).iter().all(|v| v.is_finite()));
}

/// The load-bearing property of this port: attention is CAUSAL. Perturbing
/// the LAST position must leave every earlier output bit-identical.
#[test]
fn attention_is_causal() {
    let (client, device) = cpu_setup();
    let model = tiny_model(&device);

    let seq = 4;
    let mut data: Vec<f32> = (0..seq * HIDDEN)
        .map(|i| ((i % 7) as f32 - 3.0) / 10.0)
        .collect();
    let base = Var::new(
        Tensor::<CpuRuntime>::from_slice(&data, &[1, seq, HIDDEN], &device).expect("x"),
        false,
    );
    let base_out = out_values(&model.forward(&client, &base).expect("forward"));

    // Perturb only the final position.
    for value in data.iter_mut().skip((seq - 1) * HIDDEN) {
        *value += 3.0;
    }
    let perturbed = Var::new(
        Tensor::<CpuRuntime>::from_slice(&data, &[1, seq, HIDDEN], &device).expect("x"),
        false,
    );
    let perturbed_out = out_values(&model.forward(&client, &perturbed).expect("forward"));

    let prefix = (seq - 1) * HIDDEN;
    assert_eq!(
        base_out[..prefix],
        perturbed_out[..prefix],
        "earlier positions changed: attention is not causal"
    );
    assert!(
        base_out[prefix..]
            .iter()
            .zip(&perturbed_out[prefix..])
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "final position did not react to its own perturbation"
    );
}

/// The flag has to be load-bearing: a NoPE stack must not reproduce the
/// rotary stack's numbers on the same weights and the same input.
#[test]
fn nope_output_differs_from_rotary_output() {
    let (client, device) = cpu_setup();
    let rotary = tiny_model(&device);
    let nope = tiny_nope_model(&device);
    assert!(rotary.uses_rope());
    assert!(!nope.uses_rope());

    let x = Var::new(filled(&[1, 4, HIDDEN], 99, &device), false);
    let rotary_out = out_values(&rotary.forward(&client, &x).expect("forward"));
    let nope_out = out_values(&nope.forward(&client, &x).expect("forward"));

    assert!(
        rotary_out
            .iter()
            .zip(&nope_out)
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "no_rope changed nothing: the flag is not reaching the attention blocks"
    );
    assert!(nope_out.iter().all(|v| v.is_finite()));
}

/// NoPE drops the rotation and substitutes NOTHING, but the causal mask
/// still applies: earlier positions may not see a later one.
#[test]
fn nope_attention_is_still_causal() {
    let (client, device) = cpu_setup();
    let model = tiny_nope_model(&device);

    let seq = 4;
    let mut data: Vec<f32> = (0..seq * HIDDEN)
        .map(|i| ((i % 7) as f32 - 3.0) / 10.0)
        .collect();
    let base = Var::new(
        Tensor::<CpuRuntime>::from_slice(&data, &[1, seq, HIDDEN], &device).expect("x"),
        false,
    );
    let base_out = out_values(&model.forward(&client, &base).expect("forward"));

    for value in data.iter_mut().skip((seq - 1) * HIDDEN) {
        *value += 3.0;
    }
    let perturbed = Var::new(
        Tensor::<CpuRuntime>::from_slice(&data, &[1, seq, HIDDEN], &device).expect("x"),
        false,
    );
    let perturbed_out = out_values(&model.forward(&client, &perturbed).expect("forward"));

    let prefix = (seq - 1) * HIDDEN;
    assert_eq!(
        base_out[..prefix],
        perturbed_out[..prefix],
        "earlier positions changed: NoPE attention is not causal"
    );
    assert!(
        base_out[prefix..]
            .iter()
            .zip(&perturbed_out[prefix..])
            .any(|(a, b)| (a - b).abs() > 1e-4),
        "final position did not react to its own perturbation"
    );
}

#[test]
fn rejects_wrong_hidden_size() {
    let (client, device) = cpu_setup();
    let model = tiny_model(&device);
    let x = Var::new(filled(&[1, 2, HIDDEN + 1], 5, &device), false);
    let err = model.forward(&client, &x).unwrap_err();
    assert!(err.to_string().contains("hidden_size"), "got {err}");
}

#[test]
fn rejects_non_3d_input() {
    let (client, device) = cpu_setup();
    let model = tiny_model(&device);
    let x = Var::new(filled(&[2, HIDDEN], 5, &device), false);
    assert!(model.forward(&client, &x).is_err());
}

#[test]
fn embed_errors_without_table() {
    let (client, device) = cpu_setup();
    let model = tiny_model(&device);
    assert!(!model.has_embedding());
    let ids = Tensor::<CpuRuntime>::zeros(&[1, 2], DType::I64, &device).expect("ids");
    let err = model.embed(&client, &ids).unwrap_err();
    assert!(err.to_string().contains("vocab_size"), "got {err}");
}

/// [`MiniCpm4Model::parameters`]/[`named_parameters`] (via `Module`) on
/// the same tiny fixture the forward-pass tests above build.
/// `tiny_model` has NO `embed_tokens` (like the real `residual_lm` half
/// of this model, though here for a different reason — the fixture
/// simply never sets one), so this also pins that the absent table
/// contributes nothing rather than a placeholder entry.
#[test]
fn module_enumeration_is_non_empty_with_unique_ids_and_names() {
    let (_client, device) = cpu_setup();
    let model = tiny_model(&device);

    let params = model.parameters();
    assert!(!params.is_empty());
    let ids: std::collections::HashSet<_> = params.iter().map(|v| v.id()).collect();
    assert_eq!(ids.len(), params.len(), "duplicate TensorId");

    let named = model.named_parameters();
    assert_eq!(named.len(), params.len());
    let names: std::collections::HashSet<_> = named.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names.len(), named.len(), "duplicate parameter name");

    assert!(!named.iter().any(|(n, _)| n.starts_with("embed_tokens")));
    assert!(
        named
            .iter()
            .any(|(n, _)| n == "layers.0.self_attn.q_proj.weight")
    );
    assert!(named.iter().any(|(n, _)| n == "norm.weight"));
}

/// [`MiniCpm4Model::apply_lora`] on `["q_proj", "v_proj"]` wraps exactly
/// those two projections in EVERY layer and returns their count —
/// `k_proj`/`o_proj` and every MLP projection stay `Plain`.
#[test]
fn apply_lora_adapts_exactly_targeted_projections_across_layers() {
    let (_client, device) = cpu_setup();
    let mut model = tiny_model(&device);
    let targets = LoraTargets::new(["q_proj", "v_proj"]);

    let adapted = model
        .apply_lora(&targets, 2, 4.0, &device, "")
        .expect("apply_lora");
    assert_eq!(adapted, 2 * NUM_LAYERS);

    for layer in &model.layers {
        assert!(layer.self_attn.q_proj.is_adapted());
        assert!(layer.self_attn.v_proj.is_adapted());
        assert!(!layer.self_attn.k_proj.is_adapted());
        assert!(!layer.self_attn.o_proj.is_adapted());
        assert!(!layer.mlp.gate_proj.is_adapted());
        assert!(!layer.mlp.up_proj.is_adapted());
        assert!(!layer.mlp.down_proj.is_adapted());
    }
}

/// `trainable_parameters()` after adapting is EXACTLY the adapter factors
/// (`lora_a`/`lora_b` per adapted projection) and no base weight — because
/// every VoxCPM2 base loads `requires_grad = false` (see
/// `local_encoder/encoder.rs:19`, `minicpm4/model.rs:30`), so nothing but
/// the freshly-created adapters can pass the trait's `requires_grad` filter.
#[test]
fn apply_lora_trainable_parameters_are_exactly_the_adapters() {
    let (_client, device) = cpu_setup();
    let mut model = tiny_model(&device);
    let targets = LoraTargets::new(["q_proj", "v_proj"]);
    let adapted = model
        .apply_lora(&targets, 2, 4.0, &device, "")
        .expect("apply_lora");

    let trainable_names: Vec<String> = model
        .named_parameters()
        .into_iter()
        .filter(|(_, var)| var.requires_grad())
        .map(|(name, _)| name)
        .collect();
    assert_eq!(trainable_names.len(), adapted * 2);
    assert!(
        trainable_names
            .iter()
            .all(|n| n.ends_with("lora_a") || n.ends_with("lora_b")),
        "a trainable parameter was not an adapter factor: {trainable_names:?}"
    );
}

/// After adapting, `named_parameters()` still covers every ORIGINAL
/// checkpoint key: an unadapted projection keeps its exact name, and an
/// adapted one is still enumerated through its `LoraLinear`'s own `base.*`
/// naming (see `LoraLinear::named_parameters`) rather than dropped. Checked
/// per PROJECTION path (the original name with its trailing `weight`/`bias`
/// segment removed), since adapting literally renames the leaf segment from
/// e.g. `q_proj.weight` to `q_proj.base.weight` — a LoRA wrap must not lose
/// that checkpoint key even though its exact leaf name changes.
#[test]
fn apply_lora_named_parameters_still_covers_every_original_checkpoint_key() {
    let (_client, device) = cpu_setup();
    let mut model = tiny_model(&device);
    let original_names: Vec<String> = model
        .named_parameters()
        .into_iter()
        .map(|(name, _)| name)
        .collect();

    let targets = LoraTargets::new(["q_proj", "v_proj"]);
    model
        .apply_lora(&targets, 2, 4.0, &device, "")
        .expect("apply_lora");
    let post_names: Vec<String> = model
        .named_parameters()
        .into_iter()
        .map(|(name, _)| name)
        .collect();

    for name in &original_names {
        let proj_path = name.rsplit_once('.').map_or(name.as_str(), |(p, _)| p);
        let covered = post_names
            .iter()
            .any(|pn| pn == name || pn.starts_with(&format!("{proj_path}.")));
        assert!(
            covered,
            "checkpoint key {name} (projection {proj_path}) missing after LoRA adaptation"
        );
    }
}

/// A target that matches no projection anywhere in the tree errors, naming
/// the offending target — never a silent `Ok(0)`.
#[test]
fn apply_lora_errors_when_a_target_matches_nothing() {
    let (_client, device) = cpu_setup();
    let mut model = tiny_model(&device);
    let targets = LoraTargets::new(["q_projj"]);
    let err = model.apply_lora(&targets, 2, 4.0, &device, "").unwrap_err();
    assert!(err.to_string().contains("q_projj"), "got {err}");
}

/// Dot-segment matching, not substring: `"roj"` is a substring of every
/// `*_proj` name but is not itself a `.`-separated segment of any of them,
/// so it matches nothing and errors — pinned against the same zero-match
/// trap as the test above.
#[test]
fn apply_lora_dot_segment_matching_rejects_bare_substring() {
    let (_client, device) = cpu_setup();
    let mut model = tiny_model(&device);
    let targets = LoraTargets::new(["roj"]);
    let err = model.apply_lora(&targets, 2, 4.0, &device, "").unwrap_err();
    assert!(err.to_string().contains("roj"), "got {err}");
}

/// The regression this fix targets: measured on a real VoxCPM2 GGUF,
/// `named_parameters()` shrank from 577 candidates (dense) to 131
/// (quantized), rejecting a valid `["q_proj", "v_proj"]` target as matching
/// nothing — QLoRA's entire reason to exist is adapting a quantized base, so
/// that checkpoint is exactly the one `apply_lora` must not reject. Pinned
/// here at the smallest reproducing shape: every projection in every layer
/// is `MaybeQuantLinear::Quantized` (block-quantized via `client.quantize`,
/// the same helper
/// `local_dit/tests.rs::quantized_projections_contribute_nothing_dense_norms_still_appear`
/// uses), so `named_parameters()` is provably EMPTY, yet `apply_lora(["q_proj"])`
/// must still adapt exactly one projection per layer.
#[test]
fn apply_lora_adapts_quantized_projections_instead_of_rejecting_them() {
    let (client, device) = cpu_setup();
    let mut model = quantized_tiny_model(&device, &client);

    // The old, buggy candidate source. It is not EMPTY — the RMSNorm weights
    // are dense `Var`s and still appear — but it contains no PROJECTION at
    // all, because block-quantized storage has no `Var` to enumerate. That is
    // exactly why validating targets against it broke QLoRA: `q_proj` was
    // absent from the candidate list, so a valid target was rejected.
    let param_names: Vec<String> = model
        .named_parameters()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    assert!(
        !param_names.iter().any(|n| n.contains("_proj")),
        "a fully-quantized model must expose no projection parameters, got {param_names:?}"
    );
    assert!(
        param_names.iter().any(|n| n.contains("layernorm")),
        "the dense RMSNorm weights should still be enumerated, got {param_names:?}"
    );

    let targets = LoraTargets::new(["q_proj"]);
    let adapted = model
        .apply_lora(&targets, 2, 4.0, &device, "")
        .expect("apply_lora must accept a valid target on a fully-quantized model");
    assert_eq!(adapted, NUM_LAYERS);
    for layer in &model.layers {
        assert!(layer.self_attn.q_proj.is_adapted());
        assert!(!layer.self_attn.k_proj.is_adapted());
        assert!(!layer.self_attn.v_proj.is_adapted());
        assert!(!layer.self_attn.o_proj.is_adapted());
        assert!(!layer.mlp.gate_proj.is_adapted());
    }
}

/// The zero-match trap must survive this fix: a genuinely bogus target
/// still errors on a fully-quantized model, exactly as it does on the dense
/// [`tiny_model`] in `apply_lora_errors_when_a_target_matches_nothing`. The
/// structural candidate source must not accidentally accept everything —
/// only real projection names.
#[test]
fn apply_lora_on_quantized_model_still_errors_on_bogus_target() {
    let (client, device) = cpu_setup();
    let mut model = quantized_tiny_model(&device, &client);
    let targets = LoraTargets::new(["q_projj"]);
    let err = model.apply_lora(&targets, 2, 4.0, &device, "").unwrap_err();
    assert!(err.to_string().contains("q_projj"), "got {err}");
}

/// [`MiniCpm4Model::lora_projection_names`] is STRUCTURAL, not
/// parameter-derived: the dense [`tiny_model`] and an all-quantized model
/// built with the SAME layer/projection shape return the identical name
/// list, even though their `named_parameters()` differ completely (full
/// coverage vs. empty). This is the property that makes the fix correct —
/// not merely that the regression test above happens to pass.
#[test]
fn lora_projection_names_is_identical_for_dense_and_quantized_models() {
    let (client, device) = cpu_setup();
    let quantized_model = quantized_tiny_model(&device, &client);
    let dense_model = tiny_model(&device);

    assert_eq!(
        dense_model.lora_projection_names(""),
        quantized_model.lora_projection_names("")
    );
    // The property is only meaningful because the two candidate sources
    // actually disagree here — otherwise this test would pass vacuously.
    assert_ne!(
        dense_model
            .named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        quantized_model
            .named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>()
    );
}

/// Adapting an already-adapted model errors on the second call rather than
/// silently discarding the first call's adapters.
#[test]
fn apply_lora_twice_errs_on_second_call() {
    let (_client, device) = cpu_setup();
    let mut model = tiny_model(&device);
    let targets = LoraTargets::new(["q_proj", "v_proj"]);
    model
        .apply_lora(&targets, 2, 4.0, &device, "")
        .expect("first apply_lora");
    let err = model.apply_lora(&targets, 2, 4.0, &device, "").unwrap_err();
    assert!(err.to_string().contains("already carries"), "got {err}");
}

/// The defect this split fixes: a target absent from THIS subtree (e.g.
/// `stop_proj`, which only `aux` owns in the real `VoxCpm2Model`) must not
/// be rejected by a child that a parent has already validated the full
/// target list against. `apply_lora_unchecked` skips
/// `LoraTargets::ensure_all_match` entirely and still adapts every target it
/// DOES own, while `apply_lora` on the same model/target list still errors —
/// pinning exactly the behavioural difference the split introduces.
#[test]
fn apply_lora_unchecked_does_not_reject_a_target_absent_from_this_subtree() {
    let (_client, device) = cpu_setup();
    let targets = LoraTargets::new(["q_proj", "stop_proj"]);

    let mut unchecked_model = tiny_model(&device);
    let adapted = unchecked_model
        .apply_lora_unchecked(&targets, 2, 4.0, &device, "")
        .expect("apply_lora_unchecked must not validate against this subtree");
    assert_eq!(adapted, NUM_LAYERS);
    for layer in &unchecked_model.layers {
        assert!(layer.self_attn.q_proj.is_adapted());
        assert!(!layer.self_attn.v_proj.is_adapted());
    }

    let mut checked_model = tiny_model(&device);
    let err = checked_model
        .apply_lora(&targets, 2, 4.0, &device, "")
        .unwrap_err();
    assert!(err.to_string().contains("stop_proj"), "got {err}");
}

/// Round-trip: adapt `["q_proj", "v_proj"]` across every layer, snapshot the
/// fresh adapter values and ids, build a `params` map with DIFFERENT values
/// for every adapter id, call [`MiniCpm4Model::load_lora_parameters`], and
/// confirm both the returned count and that reading the adapters back
/// through `trainable_parameters()` yields the NEW values under the SAME
/// [`TensorId`]s — the optimizer keys its moment state on those ids, so an
/// id change would silently reset Adam's state every step.
#[test]
fn load_lora_parameters_round_trips_new_values_and_preserves_ids() {
    let (_client, device) = cpu_setup();
    let mut model = tiny_model(&device);
    let targets = LoraTargets::new(["q_proj", "v_proj"]);
    let adapted = model
        .apply_lora(&targets, 2, 4.0, &device, "")
        .expect("apply_lora");

    let original: Vec<(TensorId, Vec<usize>)> = model
        .trainable_parameters()
        .into_iter()
        .map(|(id, var)| (id, var.tensor().shape().to_vec()))
        .collect();
    assert_eq!(original.len(), adapted * 2);

    let mut params: std::collections::HashMap<TensorId, Tensor<CpuRuntime>> =
        std::collections::HashMap::new();
    for (i, (id, shape)) in original.iter().enumerate() {
        params.insert(*id, filled(shape, 900 + i, &device));
    }

    let written = model
        .load_lora_parameters(&params)
        .expect("load_lora_parameters");
    assert_eq!(written, original.len());

    let updated = model.trainable_parameters();
    assert_eq!(updated.len(), original.len());
    for (id, _) in &original {
        let (_, var) = updated
            .iter()
            .find(|(uid, _)| uid == id)
            .unwrap_or_else(|| panic!("adapter {id} missing after load_lora_parameters"));
        let expected = params
            .get(id)
            .expect("params has this id")
            .contiguous()
            .expect("contiguous")
            .to_vec::<f32>();
        let actual = var
            .tensor()
            .contiguous()
            .expect("contiguous")
            .to_vec::<f32>();
        assert_eq!(actual, expected, "adapter {id} did not take the new value");
    }
}

/// A `params` map missing every adapter id writes nothing: `Ok(0)`, and
/// every adapter value stays exactly what `apply_lora` set it to.
#[test]
fn load_lora_parameters_with_no_matching_ids_returns_zero_and_leaves_values() {
    let (_client, device) = cpu_setup();
    let mut model = tiny_model(&device);
    let targets = LoraTargets::new(["q_proj"]);
    model
        .apply_lora(&targets, 2, 4.0, &device, "")
        .expect("apply_lora");

    let before: Vec<(TensorId, Vec<f32>)> = model
        .trainable_parameters()
        .into_iter()
        .map(|(id, var)| {
            (
                id,
                var.tensor()
                    .contiguous()
                    .expect("contiguous")
                    .to_vec::<f32>(),
            )
        })
        .collect();

    let params: std::collections::HashMap<TensorId, Tensor<CpuRuntime>> =
        std::collections::HashMap::new();
    let written = model
        .load_lora_parameters(&params)
        .expect("load_lora_parameters");
    assert_eq!(written, 0);

    let after: Vec<(TensorId, Vec<f32>)> = model
        .trainable_parameters()
        .into_iter()
        .map(|(id, var)| {
            (
                id,
                var.tensor()
                    .contiguous()
                    .expect("contiguous")
                    .to_vec::<f32>(),
            )
        })
        .collect();
    assert_eq!(before, after);
}

/// A `params` map that carries `lora_a` for an adapted projection but not
/// its paired `lora_b` is a torn update and errors, naming the projection.
#[test]
fn load_lora_parameters_errs_on_torn_update() {
    let (_client, device) = cpu_setup();
    let mut model = tiny_model(&device);
    let targets = LoraTargets::new(["q_proj"]);
    model
        .apply_lora(&targets, 2, 4.0, &device, "")
        .expect("apply_lora");

    let (lora_a, _) = model.layers[0]
        .self_attn
        .q_proj
        .adapters()
        .expect("q_proj is adapted");
    let lora_a_id = lora_a.id();
    let lora_a_shape = lora_a.tensor().shape().to_vec();

    let mut params: std::collections::HashMap<TensorId, Tensor<CpuRuntime>> =
        std::collections::HashMap::new();
    params.insert(lora_a_id, filled(&lora_a_shape, 42, &device));

    let err = model.load_lora_parameters(&params).unwrap_err();
    assert!(err.to_string().contains("q_proj"), "got {err}");
}

/// An unadapted (every projection `Plain`) model has no adapters to write
/// back, so `load_lora_parameters` is `Ok(0)` regardless of `params`.
#[test]
fn load_lora_parameters_on_unadapted_model_returns_zero() {
    let (_client, device) = cpu_setup();
    let mut model = tiny_model(&device);
    let params: std::collections::HashMap<TensorId, Tensor<CpuRuntime>> =
        std::collections::HashMap::new();
    let written = model
        .load_lora_parameters(&params)
        .expect("load_lora_parameters");
    assert_eq!(written, 0);
}
