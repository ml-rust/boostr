//! LoRA adapter attachment across the layer stack, the structural projection
//! path list it validates against, and the trainable toggle.

use super::stack::MiniCpm4Model;
use crate::error::Result;
use crate::nn::LoraTargets;
use numr::dtype::DType;
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> MiniCpm4Model<R> {
    /// Wrap every layer's targeted projections with a fresh LoRA adapter,
    /// returning the total adapted. `prefix` mirrors
    /// `Module::named_parameters` exactly: each layer is joined at
    /// `"layers.{i}"`. `embed_tokens`/`norm` carry no [`crate::nn::MaybeLoraLinear`]
    /// projections, so neither is touched here.
    ///
    /// This is the entry point for adapting this sub-model DIRECTLY (VoxCPM2
    /// instantiates this type twice — `base_lm`/`residual_lm` — and a caller
    /// may adapt either on its own), so it validates every target up front
    /// with [`LoraTargets::ensure_all_match`] against this tree's OWN full
    /// candidate set — [`Self::lora_projection_names`], NOT
    /// `self.named_parameters()` — before delegating to
    /// [`Self::apply_lora_unchecked`].
    ///
    /// The candidate set MUST be structural (every projection this model
    /// COULD adapt), not parameter-derived (every projection that HAPPENS
    /// to carry a dense `Var<R>` right now): on a GGUF checkpoint every
    /// `MiniCpm4Attention`/`MiniCpm4Mlp` projection is block-quantized, so
    /// `named_parameters()` returns EMPTY for all of them and a valid
    /// `q_proj`/`v_proj` target would be rejected as matching nothing —
    /// exactly the QLoRA-unusable bug this candidate source fixes.
    pub fn apply_lora(
        &mut self,
        targets: &LoraTargets,
        rank: usize,
        alpha: f32,
        device: &R::Device,
        prefix: &str,
    ) -> Result<usize> {
        let candidates = self.lora_projection_names(prefix);
        targets.ensure_all_match(&candidates)?;
        self.apply_lora_unchecked(targets, rank, alpha, device, prefix)
    }

    /// Every dotted projection path [`Self::apply_lora`] would adapt under
    /// `prefix` — INDEPENDENT of whether each layer's projections are
    /// dense, block-quantized, or decomposed-quantized. `embed_tokens`/
    /// `norm` carry no [`crate::nn::MaybeLoraLinear`] projections, so
    /// neither contributes a name, matching [`Self::apply_lora_unchecked`]'s
    /// walk exactly: each layer is joined at the SAME `"layers.{i}"` prefix
    /// [`Self::apply_lora_unchecked`] passes to
    /// [`crate::model::audio::voxcpm::minicpm4::layer::MiniCpm4Layer::apply_lora`],
    /// so a path here is never built by separately hand-written logic.
    pub fn lora_projection_names(&self, prefix: &str) -> Vec<String> {
        let mut names = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            names.extend(
                layer.lora_projection_names(&LoraTargets::join(prefix, &format!("layers.{i}"))),
            );
        }
        names
    }

    /// Same walk as [`Self::apply_lora`] but skips
    /// [`LoraTargets::ensure_all_match`]. Exists for a parent
    /// (`VoxCpm2Model`) that has already validated `targets` against the
    /// WHOLE model: re-validating here against only this subtree would
    /// reject a target that lives in a sibling (`feat_encoder`,
    /// `feat_decoder`, `aux`), even though it is perfectly valid at root.
    pub(crate) fn apply_lora_unchecked(
        &mut self,
        targets: &LoraTargets,
        rank: usize,
        alpha: f32,
        device: &R::Device,
        prefix: &str,
    ) -> Result<usize> {
        let mut adapted = 0;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            adapted += layer.apply_lora(
                targets,
                rank,
                alpha,
                device,
                &LoraTargets::join(prefix, &format!("layers.{i}")),
            )?;
        }
        Ok(adapted)
    }

    /// Set every attached adapter's `lora_a`/`lora_b` to `trainable` across
    /// every layer, returning how many projections carry an adapter. No
    /// `targets`/`prefix` needed — unlike [`Self::apply_lora`], this is a
    /// blanket toggle over whatever is already adapted.
    pub fn set_lora_trainable(&mut self, trainable: bool) -> usize {
        let mut touched = 0;
        for layer in self.layers.iter_mut() {
            touched += layer.set_lora_trainable(trainable);
        }
        touched
    }
}

#[cfg(test)]
mod tests {
    use super::super::stack::tests::{NUM_LAYERS, tiny_model};
    use super::*;
    use crate::model::audio::voxcpm::minicpm4::attention::MiniCpm4Attention;
    use crate::model::audio::voxcpm::minicpm4::layer::MiniCpm4Layer;
    use crate::model::audio::voxcpm::minicpm4::mlp::MiniCpm4Mlp;
    use crate::nn::{MaybeLoraLinear, MaybeQuantLinear, Module, RmsNorm};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

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
}
