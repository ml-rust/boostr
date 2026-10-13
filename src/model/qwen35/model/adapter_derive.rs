//! [`Qwen35Model::derive_with_adapter`]: a second model over a base
//! model's weight storage, with a PEFT LoRA adapter attached unmerged to
//! its own projection objects.
//!
//! Every module duplicates through its `alias`: a dense weight is a
//! `Var::alias` (same storage, same `TensorId`), a block-quantized weight a
//! `QuantTensor` clone (same packed storage), a Hadamard rotation a clone
//! (same sign storage). No weight data is copied. The attach then swaps
//! projection objects in the derived model only.

use super::build::{Qwen35AttentionLayer, Qwen35Block, Qwen35GdnLayer, Qwen35Model};
use crate::error::{Error, Result};
use crate::format::peft::PeftAdapter;
use numr::dtype::DType;
use numr::ops::{ScalarOps, ShapeOps, TypeConversionOps};
use numr::runtime::{Device, Runtime, RuntimeClient};

impl<R: Runtime<DType = DType>> Qwen35Block<R> {
    /// The layer over the same weight storage, with its own modules.
    fn alias(&self) -> Self {
        match self {
            Self::Gdn(layer) => Self::Gdn(Box::new(Qwen35GdnLayer {
                attn_norm: layer.attn_norm.alias(),
                mixer: layer.mixer.alias(),
                post_attention_norm: layer.post_attention_norm.alias(),
                mlp: layer.mlp.alias(),
            })),
            Self::Attention(layer) => Self::Attention(Box::new(Qwen35AttentionLayer {
                attn_norm: layer.attn_norm.alias(),
                mixer: layer.mixer.alias(),
                post_attention_norm: layer.post_attention_norm.alias(),
                mlp: layer.mlp.alias(),
            })),
        }
    }
}

impl<R: Runtime<DType = DType>> Qwen35Model<R> {
    /// A model over `self`'s weight storage with `adapter` attached
    /// unmerged, built without reading a weight again.
    ///
    /// The derived model shares every weight with `self`: dense, block
    /// quantized, and Hadamard sign tensors alike. Only the adapter factors
    /// take new memory. `self` is not modified and stays usable, so one
    /// base serves several adapters at once. The attach is the one
    /// [`Self::from_varbuilder_with_adapter`] runs, with the same module
    /// names, layout moves and casts, on `client`.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when `self` already carries an adapter
    /// (adapters do not stack) or `client` is on another device than the
    /// model. The attach errors of [`Self::from_varbuilder_with_adapter`]:
    /// an adapter module that matches no projection, a factor shape that
    /// does not fit its base (both shapes named), a factor on another
    /// device, or a Hadamard-rotated projection.
    pub fn derive_with_adapter<C>(&self, client: &C, adapter: &PeftAdapter<R>) -> Result<Self>
    where
        C: RuntimeClient<R> + ScalarOps<R> + ShapeOps<R> + TypeConversionOps<R>,
        R::Client: ShapeOps<R> + ScalarOps<R> + TypeConversionOps<R>,
    {
        if !self.adapted_modules.is_empty() {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35: derive_with_adapter needs a base without an adapter; this model \
                     already carries one on {} modules, and adapters do not stack. Derive from \
                     the model loaded without an adapter",
                    self.adapted_modules.len()
                ),
            });
        }
        let device = self.norm.weight().tensor().device();
        if !client.device().is_same(device) {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35: derive_with_adapter got a client on {} for a model on {}; pass a \
                     client on the model's device",
                    client.device().name(),
                    device.name()
                ),
            });
        }
        let mut derived = self.alias();
        derived.attach_peft_adapter(client, adapter)?;
        Ok(derived)
    }

    /// The model over the same weight storage, with its own modules.
    fn alias(&self) -> Self {
        Self {
            config: self.config.clone(),
            gdn_config: self.gdn_config.clone(),
            attention_config: self.attention_config.clone(),
            layers: self.layers.clone(),
            embed_tokens: self.embed_tokens.alias(),
            blocks: self.blocks.iter().map(Qwen35Block::alias).collect(),
            norm: self.norm.alias(),
            lm_head: self.lm_head.alias(),
            rope: self.rope.alias(),
            adapted_modules: self.adapted_modules.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::adapter::tests::{
        ALL_MODULES, build, factors, logits, max_diff, write_adapter,
    };
    use super::super::build::tests::rotated_model;
    use super::super::tiny_config;
    use super::*;
    use crate::format::hf_qwen35::restructure::tests::maps;
    use crate::model::config::UniversalConfig;
    use crate::nn::{MaybeQuantLinear, MaybeRotatedLinear, VarMap};
    use crate::quant::QuantFormat;
    use crate::quant::traits::QuantizeOps;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;
    use std::path::Path;

    const RANK: usize = 2;

    type Entries = Vec<(String, Tensor<CpuRuntime>, Tensor<CpuRuntime>)>;

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// The base in the loader layout of `config`, after `edit`.
    fn base_map(
        client: &CpuClient,
        device: &CpuDevice,
        config: &UniversalConfig,
        edit: fn(&CpuClient, &mut VarMap<CpuRuntime>),
    ) -> VarMap<CpuRuntime> {
        let (_, mut map) = maps(client, device, config);
        edit(client, &mut map);
        map
    }

    fn keep(_: &CpuClient, _: &mut VarMap<CpuRuntime>) {}

    /// `modules`' factors, scaled by `scale`, written to `dir` and loaded.
    fn make_adapter(
        client: &CpuClient,
        device: &CpuDevice,
        config: &UniversalConfig,
        dir: &Path,
        modules: &[&str],
        scale: f64,
    ) -> PeftAdapter<CpuRuntime> {
        let (hf, _) = maps(client, device, config);
        let entries: Entries = factors(&hf, modules, device)
            .into_iter()
            .map(|(m, a, b)| (m, client.mul_scalar(&a, scale).unwrap(), b))
            .collect();
        write_adapter(dir, &entries, device)
    }

    /// Derived logits equal the direct build's, for each adapter in turn
    /// from one base, and the base logits stay bit-identical throughout.
    fn derive_matches_direct(
        config: &UniversalConfig,
        edit: fn(&CpuClient, &mut VarMap<CpuRuntime>),
    ) {
        let (client, device) = cpu_setup();
        let dirs = [tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap()];
        let adapters = [
            make_adapter(&client, &device, config, dirs[0].path(), &ALL_MODULES, 1.0),
            make_adapter(
                &client,
                &device,
                config,
                dirs[1].path(),
                &ALL_MODULES[3..9],
                -0.5,
            ),
        ];
        let mut map = base_map(&client, &device, config, edit);
        let base = build(&mut map, config, &device, None).unwrap();
        let before = logits(&client, &device, &base, config);

        let mut derived_logits = Vec::new();
        for adapter in &adapters {
            let derived = base.derive_with_adapter(&client, adapter).unwrap();
            let mut map = base_map(&client, &device, config, edit);
            let direct = build(&mut map, config, &device, Some(adapter)).unwrap();
            assert_eq!(derived.adapted_modules(), direct.adapted_modules());
            let got = logits(&client, &device, &derived, config);
            let want = logits(&client, &device, &direct, config);
            let diff = max_diff(&got, &want);
            assert!(diff < 1e-6, "derived vs direct differ by {diff}");
            derived_logits.push(got);
        }
        assert!(base.adapted_modules().is_empty());
        let after = logits(&client, &device, &base, config);
        assert_eq!(bits(&after), bits(&before), "the base logits moved");
        let apart = max_diff(&derived_logits[0], &derived_logits[1]);
        assert!(apart > 1e-3, "the two adapters agree to {apart}");
        assert!(max_diff(&derived_logits[0], &before) > 1e-3);
    }

    #[test]
    fn derived_logits_equal_the_direct_build() {
        derive_matches_direct(&tiny_config(), keep);
    }

    /// [`tiny_config`] widened so every projection with a 32-wide input
    /// can hold Q8_0 blocks.
    fn wide_config() -> UniversalConfig {
        let mut config = tiny_config();
        config.hidden_size = 32;
        config.intermediate_size = Some(32);
        if let Some(gdn) = config.gdn.as_mut() {
            gdn.hidden_size = 32;
        }
        if let Some(attn) = config.qwen35_attention.as_mut() {
            attn.hidden_size = 32;
        }
        config
    }

    /// Every projection with a 32-wide input as Q8_0: both MLPs, the GDN
    /// `in_proj_qkv`, `in_proj_z` and gate pair, attention `q`/`k`/`v`,
    /// and `lm_head`.
    fn quantize(client: &CpuClient, map: &mut VarMap<CpuRuntime>) {
        let mut names: Vec<String> = ["gate_proj", "up_proj", "down_proj"]
            .iter()
            .flat_map(|p| (0..2).map(move |l| format!("model.layers.{l}.mlp.{p}.weight")))
            .collect();
        for p in ["in_proj_qkv", "in_proj_z", "alpha_proj", "beta_proj"] {
            names.push(format!("model.layers.0.linear_attn.{p}.weight"));
        }
        for p in ["q_proj", "k_proj", "v_proj"] {
            names.push(format!("model.layers.1.self_attn.{p}.weight"));
        }
        names.push("lm_head.weight".into());
        for name in names {
            let dense = map.get_tensor(&name).unwrap().clone();
            let quant = client.quantize(&dense, QuantFormat::Q8_0).unwrap();
            map.remove(&name);
            map.insert_quant(name, quant);
        }
    }

    #[test]
    fn quantized_base_derives_like_the_direct_build() {
        derive_matches_direct(&wide_config(), quantize);
    }

    fn ptr(t: &Tensor<CpuRuntime>) -> u64 {
        t.storage().ptr()
    }

    /// `(mlp gate, the layer's own projection)` of block `i`.
    fn projections(
        model: &Qwen35Model<CpuRuntime>,
        i: usize,
    ) -> (
        &MaybeRotatedLinear<CpuRuntime>,
        &MaybeRotatedLinear<CpuRuntime>,
    ) {
        match &model.blocks[i] {
            Qwen35Block::Gdn(l) => (l.mlp.gate(), l.mlp.down()),
            Qwen35Block::Attention(l) => (l.mlp.gate(), l.mlp.up()),
        }
    }

    fn dense_ptr(linear: &MaybeRotatedLinear<CpuRuntime>) -> u64 {
        ptr(linear.base().weight().unwrap().tensor())
    }

    #[test]
    fn derived_model_shares_the_base_storage() {
        let (client, device) = cpu_setup();
        let config = tiny_config();
        let dir = tempfile::tempdir().unwrap();
        let adapter = make_adapter(&client, &device, &config, dir.path(), &ALL_MODULES, 1.0);
        let mut map = base_map(&client, &device, &config, keep);
        let base = build(&mut map, &config, &device, None).unwrap();

        let norm = base.norm.weight();
        let refs = norm.tensor().storage().ref_count();
        let derived = base.derive_with_adapter(&client, &adapter).unwrap();
        assert_eq!(norm.tensor().storage().ref_count(), refs + 1);
        assert_eq!(ptr(derived.norm.weight().tensor()), ptr(norm.tensor()));
        assert_eq!(derived.norm.weight().id(), norm.id(), "alias keeps the id");

        let embed = |m: &Qwen35Model<CpuRuntime>| ptr(m.embed_tokens.weight().unwrap().tensor());
        assert_eq!(embed(&derived), embed(&base));
        assert_eq!(dense_ptr(&derived.lm_head), dense_ptr(&base.lm_head));
        for i in 0..config.num_layers {
            let (base_gate, base_other) = projections(&base, i);
            let (gate, other) = projections(&derived, i);
            assert!(!base_gate.is_adapted(), "layer {i}: the base was adapted");
            assert!(
                gate.is_adapted(),
                "layer {i}: the derived gate has no adapter"
            );
            assert_eq!(dense_ptr(gate), dense_ptr(base_gate), "layer {i} gate");
            assert_eq!(dense_ptr(other), dense_ptr(base_other), "layer {i}");
        }
        drop(derived);
        assert_eq!(norm.tensor().storage().ref_count(), refs);
    }

    #[test]
    fn quantized_weights_share_their_packed_storage() {
        let (client, device) = cpu_setup();
        let config = wide_config();
        let dir = tempfile::tempdir().unwrap();
        let adapter = make_adapter(&client, &device, &config, dir.path(), &ALL_MODULES, 1.0);
        let mut map = base_map(&client, &device, &config, quantize);
        let base = build(&mut map, &config, &device, None).unwrap();
        let derived = base.derive_with_adapter(&client, &adapter).unwrap();
        let packed = |linear: &MaybeRotatedLinear<CpuRuntime>| match linear.base() {
            MaybeQuantLinear::Quantized(q) => q.weight().storage().ptr(),
            _ => panic!("expected a Q8_0 base"),
        };
        assert_eq!(packed(&derived.lm_head), packed(&base.lm_head));
        let (base_gate, _) = projections(&base, 0);
        let (gate, _) = projections(&derived, 0);
        assert!(gate.is_adapted() && !base_gate.is_adapted());
        assert_eq!(packed(gate), packed(base_gate));
    }

    #[test]
    fn rotated_base_shares_its_signs_and_refuses_the_adapter() {
        let (client, device) = cpu_setup();
        let base = rotated_model(&device, 0x3535_d001);
        let derived = base.alias();
        let signs = |m: &Qwen35Model<CpuRuntime>| m.lm_head.rotation().and_then(|r| r.signs_ptr());
        assert!(signs(&base).is_some());
        assert_eq!(signs(&derived), signs(&base));
        assert_eq!(dense_ptr(&derived.lm_head), dense_ptr(&base.lm_head));

        let h = base.config().hidden_size;
        let inter = base.config().intermediate_size();
        let text = derive_error(
            &client,
            &device,
            &base,
            "layers.0.mlp.gate_proj",
            &[RANK, h],
            &[inter, RANK],
        );
        assert!(text.contains("model.layers.0.mlp.gate_proj"), "{text}");
        assert!(text.contains("Hadamard-rotated"), "{text}");
        assert!(!projections(&base, 0).0.is_adapted());
    }

    /// The error of deriving `base` with one zero adapter module.
    fn derive_error(
        client: &CpuClient,
        device: &CpuDevice,
        base: &Qwen35Model<CpuRuntime>,
        module: &str,
        a: &[usize],
        b: &[usize],
    ) -> String {
        let dir = tempfile::tempdir().unwrap();
        let zeros = |shape: &[usize]| Tensor::<CpuRuntime>::zeros(shape, DType::F32, device);
        let entries = vec![(module.to_string(), zeros(a).unwrap(), zeros(b).unwrap())];
        let adapter = write_adapter(dir.path(), &entries, device);
        match base.derive_with_adapter(client, &adapter) {
            Ok(_) => panic!("{module}: the adapter must be refused"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn attach_errors_match_the_direct_build() {
        let (client, device) = cpu_setup();
        let config = tiny_config();
        let mut map = base_map(&client, &device, &config, keep);
        let base = build(&mut map, &config, &device, None).unwrap();
        let h = config.hidden_size;

        let text = derive_error(
            &client,
            &device,
            &base,
            "layers.0.self_attn.q_proj",
            &[RANK, h],
            &[4, RANK],
        );
        assert!(text.contains("model.layers.0.self_attn.q_proj"), "{text}");
        assert!(text.contains("matches no qwen35 projection"), "{text}");

        let value_dim = config.gdn.as_ref().unwrap().value_dim();
        let text = derive_error(
            &client,
            &device,
            &base,
            "layers.0.linear_attn.in_proj_z",
            &[RANK, h],
            &[value_dim - 1, RANK],
        );
        assert!(
            text.contains("model.layers.0.linear_attn.in_proj_z"),
            "{text}"
        );
        assert!(
            text.contains(&format!("[{}, {RANK}]", value_dim - 1)),
            "{text}"
        );
        assert!(text.contains(&format!("[{value_dim}, {h}]")), "{text}");
        assert!(base.adapted_modules().is_empty());
    }

    #[test]
    fn an_adapted_base_is_refused() {
        let (client, device) = cpu_setup();
        let config = tiny_config();
        let dir = tempfile::tempdir().unwrap();
        let adapter = make_adapter(&client, &device, &config, dir.path(), &ALL_MODULES, 1.0);
        let mut map = base_map(&client, &device, &config, keep);
        let adapted = build(&mut map, &config, &device, Some(&adapter)).unwrap();
        let stacked =
            |model: &Qwen35Model<CpuRuntime>| match model.derive_with_adapter(&client, &adapter) {
                Ok(_) => panic!("a second adapter must be refused"),
                Err(e) => e.to_string(),
            };
        let err = stacked(&adapted);
        assert!(err.contains("adapters do not stack"), "{err}");

        let mut map = base_map(&client, &device, &config, keep);
        let base = build(&mut map, &config, &device, None).unwrap();
        let derived = base.derive_with_adapter(&client, &adapter).unwrap();
        let err = stacked(&derived);
        assert!(err.contains("adapters do not stack"), "{err}");
    }
}
