//! [`Qwen35Model::from_varbuilder_with_adapter`]: the model with a PEFT LoRA
//! adapter attached unmerged at load, and [`Qwen35Model::adapted_modules`].
//!
//! The attach itself is in `adapter_attach`, the module-to-projection
//! mapping and layout moves in `adapter_targets`.

use super::build::Qwen35Model;
use crate::error::Result;
use crate::format::peft::PeftAdapter;
use crate::model::config::UniversalConfig;
use crate::nn::VarBuilder;
use numr::dtype::DType;
use numr::ops::{ScalarOps, ShapeOps, TypeConversionOps};
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> Qwen35Model<R> {
    /// [`Self::from_varbuilder`], then `adapter` attached unmerged when
    /// given.
    ///
    /// The adapter's module names are base-checkpoint names
    /// (`model.layers.N.<sub>`, as [`PeftAdapter::load`] normalizes them)
    /// and its factors are in the Hugging Face layout, whether `vb` holds
    /// a GGUF or a restructured Hugging Face checkpoint. Adaptable
    /// submodules: `self_attn.{q,k,v,o}_proj`,
    /// `linear_attn.{in_proj_qkv,in_proj_z,out_proj,in_proj_a,in_proj_b}`
    /// and `mlp.{gate,up,down}_proj`. `in_proj_a` and `in_proj_b` share
    /// one fused base; either or both may be adapted.
    ///
    /// # Errors
    ///
    /// [`Self::from_varbuilder`] errors, and
    /// [`Error::ModelError`](crate::error::Error::ModelError) naming
    /// the module when an adapter module matches no projection, a factor
    /// shape does not fit its base (both shapes named), a factor is on
    /// another device, or the projection is Hadamard-rotated.
    pub fn from_varbuilder_with_adapter(
        vb: &mut VarBuilder<R>,
        config: &UniversalConfig,
        adapter: Option<&PeftAdapter<R>>,
    ) -> Result<Self>
    where
        R::Client: ShapeOps<R> + ScalarOps<R> + TypeConversionOps<R>,
    {
        let mut model = Self::from_varbuilder(vb, config)?;
        if let Some(adapter) = adapter {
            let client = R::default_client(model.norm.weight().tensor().device());
            model.attach_peft_adapter(&client, adapter)?;
        }
        Ok(model)
    }

    /// Base-checkpoint names of the modules the attached adapter targets,
    /// sorted. Empty without an adapter.
    pub fn adapted_modules(&self) -> &[String] {
        &self.adapted_modules
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::tiny_config;
    use super::*;
    use crate::format::hf_qwen35::restructure::tests::maps;
    use crate::format::peft::{ADAPTER_CONFIG_FILE, ADAPTER_WEIGHTS_FILE};
    use crate::format::restructure_hf_checkpoint;
    use crate::format::safetensors::save_safetensors;
    use crate::inference::{LayeredGdnState, LayeredKvCache, LayeredKvCacheConfig};
    use crate::nn::VarMap;
    use crate::test_utils::cpu_setup;
    use numr::ops::{BinaryOps, MatmulOps};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;
    use std::collections::HashMap;
    use std::path::Path;

    const RANK: usize = 2;
    /// `lora_alpha = 3`, `r = 2`: scaling 1.5.
    const ALPHA: f32 = 3.0;
    const SCALING: f32 = ALPHA / RANK as f32;

    /// Every adaptable module of `tiny_config` (layer 0 Gated DeltaNet,
    /// layer 1 full attention), in the bare `layers.N.<sub>` form PEFT
    /// writes for an adapter trained on the text model.
    pub(crate) const ALL_MODULES: [&str; 13] = [
        "layers.0.linear_attn.in_proj_qkv",
        "layers.0.linear_attn.in_proj_z",
        "layers.0.linear_attn.out_proj",
        "layers.0.linear_attn.in_proj_a",
        "layers.0.linear_attn.in_proj_b",
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
        "layers.0.mlp.down_proj",
        "layers.1.self_attn.q_proj",
        "layers.1.self_attn.k_proj",
        "layers.1.self_attn.v_proj",
        "layers.1.self_attn.o_proj",
        "layers.1.mlp.gate_proj",
    ];

    /// Deterministic values in `[-0.4, 0.4)`, seeded from `name`.
    fn values(name: &str, n: usize) -> Vec<f32> {
        let mut s = name.bytes().fold(0x9e37_79b9_7f4a_7c15u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        });
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.8
            })
            .collect()
    }

    /// The Hugging Face weight a bare module adapts.
    fn hf_weight(module: &str) -> String {
        format!("model.language_model.{module}.weight")
    }

    /// `(module, lora_A, lora_B)` for `modules`, shaped from the Hugging
    /// Face weights in `hf`.
    pub(crate) fn factors(
        hf: &VarMap<CpuRuntime>,
        modules: &[&str],
        device: &CpuDevice,
    ) -> Vec<(String, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        modules
            .iter()
            .map(|m| {
                let shape = hf.get_tensor(&hf_weight(m)).unwrap().shape().to_vec();
                let (out, inp) = (shape[0], shape[1]);
                let a = Tensor::from_slice(
                    &values(&format!("{m}.A"), RANK * inp),
                    &[RANK, inp],
                    device,
                )
                .unwrap();
                let b = Tensor::from_slice(
                    &values(&format!("{m}.B"), out * RANK),
                    &[out, RANK],
                    device,
                )
                .unwrap();
                (m.to_string(), a, b)
            })
            .collect()
    }

    /// Write `adapter_config.json` and `adapter_model.safetensors` with
    /// `base_model.model.<module>.lora_{A,B}.weight` keys into `dir`, and
    /// load them.
    pub(crate) fn write_adapter(
        dir: &Path,
        entries: &[(String, Tensor<CpuRuntime>, Tensor<CpuRuntime>)],
        device: &CpuDevice,
    ) -> PeftAdapter<CpuRuntime> {
        let config = format!(r#"{{"peft_type": "LORA", "r": {RANK}, "lora_alpha": {ALPHA}}}"#);
        std::fs::write(dir.join(ADAPTER_CONFIG_FILE), config).unwrap();
        let mut map: HashMap<String, Tensor<CpuRuntime>> = HashMap::new();
        for (module, a, b) in entries {
            map.insert(
                format!("base_model.model.{module}.lora_A.weight"),
                a.clone(),
            );
            map.insert(
                format!("base_model.model.{module}.lora_B.weight"),
                b.clone(),
            );
        }
        save_safetensors(dir.join(ADAPTER_WEIGHTS_FILE), &map, None).unwrap();
        PeftAdapter::load(dir, device).unwrap()
    }

    /// `hf` with each module's weight replaced by `W + s B A`, in the
    /// Hugging Face layout the adapter was trained in.
    fn merge_into(
        client: &CpuClient,
        hf: &mut VarMap<CpuRuntime>,
        entries: &[(String, Tensor<CpuRuntime>, Tensor<CpuRuntime>)],
    ) {
        for (module, a, b) in entries {
            let name = hf_weight(module);
            let w = hf.get_tensor(&name).unwrap().clone();
            let ba = client.matmul(b, a).unwrap();
            let delta = client.mul_scalar(&ba, f64::from(SCALING)).unwrap();
            hf.insert(name, client.add(&w, &delta).unwrap());
        }
    }

    pub(crate) fn logits(
        client: &CpuClient,
        device: &CpuDevice,
        model: &Qwen35Model<CpuRuntime>,
        config: &UniversalConfig,
    ) -> Vec<f32> {
        let attn = model.attention_config();
        let kv_config = LayeredKvCacheConfig {
            batch_size: 1,
            num_kv_heads: attn.num_kv_heads,
            initial_capacity: config.max_seq_len,
            max_seq_len: config.max_seq_len,
            head_dim: attn.head_dim,
            dtype: DType::F32,
        };
        let mut kv =
            LayeredKvCache::<CpuRuntime>::new(model.num_attention_layers(), &kv_config, device)
                .unwrap();
        let mut gdn = LayeredGdnState::<CpuRuntime>::zeros(
            model.num_gdn_layers(),
            model.gdn_config(),
            1,
            DType::F32,
            device,
        )
        .unwrap();
        let ids: Vec<i64> = (0..6)
            .map(|i| ((i * 5 + 3) % config.vocab_size) as i64)
            .collect();
        let ids = Tensor::<CpuRuntime>::from_slice(&ids, &[1, 6], device).unwrap();
        model
            .forward_qwen35(client, &ids, &mut kv, &mut gdn, 0)
            .unwrap()
            .to_vec()
    }

    pub(crate) fn build(
        map: &mut VarMap<CpuRuntime>,
        config: &UniversalConfig,
        device: &CpuDevice,
        adapter: Option<&PeftAdapter<CpuRuntime>>,
    ) -> Result<Qwen35Model<CpuRuntime>> {
        let mut vb = VarBuilder::new(map, device);
        Qwen35Model::<CpuRuntime>::from_varbuilder_with_adapter(&mut vb, config, adapter)
    }

    pub(crate) fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// Logits of the restructured Hugging Face base with `modules` adapted
    /// unmerged, and of the same base with the adapter merged into the
    /// Hugging Face weights before the restructure.
    fn adapted_and_merged(modules: &[&str]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (client, device) = cpu_setup();
        let config = tiny_config();
        let dir = tempfile::tempdir().unwrap();
        let (mut hf, _) = maps(&client, &device, &config);
        let entries = factors(&hf, modules, &device);
        let adapter = write_adapter(dir.path(), &entries, &device);

        restructure_hf_checkpoint(&mut hf, &config, &device).unwrap();
        let adapted = build(&mut hf, &config, &device, Some(&adapter)).unwrap();
        assert_eq!(adapted.adapted_modules().len(), modules.len());
        assert_eq!(hf.len(), 0, "loader left tensors behind");

        let (mut merged_hf, _) = maps(&client, &device, &config);
        merge_into(&client, &mut merged_hf, &entries);
        restructure_hf_checkpoint(&mut merged_hf, &config, &device).unwrap();
        let merged = build(&mut merged_hf, &config, &device, None).unwrap();
        assert!(merged.adapted_modules().is_empty());

        let (mut plain_hf, _) = maps(&client, &device, &config);
        restructure_hf_checkpoint(&mut plain_hf, &config, &device).unwrap();
        let plain = build(&mut plain_hf, &config, &device, None).unwrap();
        (
            logits(&client, &device, &adapted, &config),
            logits(&client, &device, &merged, &config),
            logits(&client, &device, &plain, &config),
        )
    }

    fn assert_adapted_equals_merged(modules: &[&str]) {
        let (adapted, merged, plain) = adapted_and_merged(modules);
        assert!(adapted.iter().all(|v| v.is_finite()));
        let diff = max_diff(&adapted, &merged);
        assert!(
            diff < 1e-4,
            "{modules:?}: adapted vs merged differ by {diff}"
        );
        let effect = max_diff(&adapted, &plain);
        assert!(
            effect > 1e-3,
            "{modules:?}: the adapter moves the logits by only {effect}"
        );
    }

    /// Every projection kind at once, the fused gate pair with both halves.
    #[test]
    fn every_projection_equals_the_merged_weights() {
        assert_adapted_equals_merged(&ALL_MODULES);
    }

    /// One half of the fused gate pair: the other half's rows of the
    /// block-diagonal `lora_b` are zero.
    #[test]
    fn alpha_only_equals_the_merged_weights() {
        assert_adapted_equals_merged(&["layers.0.linear_attn.in_proj_a"]);
    }

    #[test]
    fn beta_only_equals_the_merged_weights() {
        assert_adapted_equals_merged(&["layers.0.linear_attn.in_proj_b"]);
    }

    /// `q_proj` rows interleave query and gate per head in the adapter;
    /// the block regroups them with the base rows.
    #[test]
    fn q_proj_equals_the_merged_weights() {
        assert_adapted_equals_merged(&["layers.1.self_attn.q_proj"]);
    }

    /// The V-head moves: a value-head adapter on its own.
    #[test]
    fn value_head_projections_equal_the_merged_weights() {
        assert_adapted_equals_merged(&[
            "layers.0.linear_attn.in_proj_qkv",
            "layers.0.linear_attn.in_proj_z",
            "layers.0.linear_attn.out_proj",
        ]);
    }

    /// The same Hugging Face-layout adapter over a restructured Hugging
    /// Face base and over a base already in the GGUF loader layout.
    #[test]
    fn hf_and_gguf_layout_bases_take_the_same_adapter() {
        let (client, device) = cpu_setup();
        let config = tiny_config();
        let dir = tempfile::tempdir().unwrap();
        let (mut hf, mut gguf) = maps(&client, &device, &config);
        let entries = factors(&hf, &ALL_MODULES, &device);
        let adapter = write_adapter(dir.path(), &entries, &device);

        restructure_hf_checkpoint(&mut hf, &config, &device).unwrap();
        let from_hf = build(&mut hf, &config, &device, Some(&adapter)).unwrap();
        let from_gguf = build(&mut gguf, &config, &device, Some(&adapter)).unwrap();
        assert_eq!(gguf.len(), 0, "loader left tensors behind");
        assert_eq!(from_gguf.adapted_modules(), from_hf.adapted_modules());
        let a = logits(&client, &device, &from_hf, &config);
        let b = logits(&client, &device, &from_gguf, &config);
        let diff = max_diff(&a, &b);
        assert!(diff < 1e-5, "HF and GGUF layouts differ by {diff}");
    }
}
