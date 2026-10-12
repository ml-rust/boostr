//! [`restructure`]: a Hugging Face Qwen3.5 checkpoint in the layout
//! `Qwen35Model::from_varbuilder` reads.
//!
//! That layout is the one the llama.cpp converter writes to GGUF
//! (`Qwen3_5TextModel` in `conversion/qwen.py`). The edits:
//!
//! 1. Names follow `rename`. MTP and vision tensors leave the map.
//! 2. Each Gated DeltaNet layer moves its value heads from grouped to tiled
//!    order (`reorder_v_heads`):
//!    - rows: the V part of `in_proj_qkv` and of `conv1d`, all of
//!      `in_proj_z`, `alpha_proj`, `beta_proj`, `a_neg_exp`, `dt_bias`
//!    - columns: `out_proj`
//! 3. `a_neg_exp` becomes `-exp(A_log)`, after the reorder.
//! 4. Each norm weight `w` becomes `1 + w`: `input_layernorm`,
//!    `post_attention_layernorm`, `self_attn.q_norm`, `self_attn.k_norm` and
//!    `model.norm`. `linear_attn.norm` keeps its weight.
//!
//! `conv1d` keeps its `[channels, 1, kernel]` shape. `q_proj` keeps its
//! per-head `[query | gate]` rows. Every edit stays in the tensor's dtype
//! and on its device.

use super::edits::{edit, edit_gdn_layer, model_error, plus_one};
use super::rename::qwen35_name;
use crate::error::Result;
use crate::model::config::UniversalConfig;
use crate::nn::VarMap;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ShapeOps, UnaryOps};
use numr::runtime::Runtime;

/// Rename a Hugging Face Qwen3.5 checkpoint's tensors and make the layout
/// edits the module docs list. Marks the map as restructured.
///
/// `config` is the `qwen35` config of the checkpoint, from
/// `qwen35_config_from_hf`.
///
/// # Errors
///
/// [`Error::ModelError`](crate::error::Error::ModelError) when the map is
/// already restructured, `config` is not a `qwen35` config without a
/// Hadamard contract, or a tensor the edits need is missing, quantized, or
/// the wrong size.
pub fn restructure<R: Runtime<DType = DType>>(
    map: &mut VarMap<R>,
    config: &UniversalConfig,
) -> Result<()>
where
    R::Client: ShapeOps<R> + BinaryOps<R> + UnaryOps<R>,
{
    if map.is_hf_restructured() {
        return Err(model_error(
            "qwen35: the checkpoint is already restructured; a second run would apply \
             its weight edits twice"
                .into(),
        ));
    }
    if config.model_type != "qwen35" {
        return Err(model_error(format!(
            "qwen35 restructure got model_type '{}'",
            config.model_type
        )));
    }
    let gdn = config
        .gdn
        .as_ref()
        .ok_or_else(|| model_error("qwen35 requires a gdn config".into()))?;
    let layers = config
        .hybrid_layers
        .as_ref()
        .ok_or_else(|| model_error("qwen35 requires hybrid_layers".into()))?;
    if config.hadamard.is_some() || gdn.v_grouped {
        return Err(model_error(
            "qwen35: a Hugging Face checkpoint carries no Hadamard contract".into(),
        ));
    }

    let names: Vec<String> = map.names().map(str::to_string).collect();
    for name in names {
        let target = qwen35_name(&name);
        if target.as_deref() == Some(name.as_str()) {
            continue;
        }
        let Some(weight) = map.remove(&name) else {
            continue;
        };
        if let Some(target) = target {
            map.insert_weight(target, weight);
        }
    }

    for layer in 0..config.num_layers {
        let p = format!("model.layers.{layer}");
        if layers.is_attention_layer(layer) {
            for norm in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
                edit(map, &format!("{p}.{norm}"), plus_one)?;
            }
        } else {
            edit_gdn_layer(map, &format!("{p}.linear_attn"), gdn)?;
        }
        for norm in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            edit(map, &format!("{p}.{norm}"), plus_one)?;
        }
    }
    edit(map, "model.norm.weight", plus_one)?;
    map.mark_hf_restructured();
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::format::restructure_hf_checkpoint;
    use crate::inference::{LayeredGdnState, LayeredKvCache, LayeredKvCacheConfig};
    use crate::model::config::GdnConfig;
    use crate::model::qwen35::Qwen35Model;
    use crate::model::qwen35::model::tiny_config;
    use crate::nn::VarBuilder;
    use crate::test_utils::cpu_setup;
    use numr::ops::{IndexingOps, ScalarOps};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    /// What `restructure` does to a tensor, written out here by index
    /// arithmetic rather than through `reorder_v_heads`.
    #[derive(Clone, Copy)]
    enum Edit {
        Keep,
        Norm,
        /// Rows in value heads of this width.
        Rows(usize),
        /// Rows after `2 * key_dim`, value heads of width `state_size`.
        Tail,
        /// Columns in value heads of width `state_size`.
        Cols,
        /// `Rows(1)`, then `-exp`.
        ALog,
    }

    /// (Hugging Face name, loader name, shape, edit).
    type Entry = (String, String, Vec<usize>, Edit);

    /// Deterministic values in `[-0.3, 0.3)`, seeded from `name`.
    fn values(name: &str, n: usize) -> Vec<f32> {
        let mut s = name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        });
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.6
            })
            .collect()
    }

    fn tensor(name: &str, shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
        let n = shape.iter().product();
        Tensor::from_slice(&values(name, n), shape, device).unwrap()
    }

    /// Every tensor of `tiny_config()` in both namings.
    fn entries(config: &UniversalConfig) -> Vec<Entry> {
        let gdn = config.gdn.as_ref().unwrap();
        let attn = config.qwen35_attention.as_ref().unwrap();
        let (h, v, inter) = (
            config.hidden_size,
            config.vocab_size,
            config.intermediate_size(),
        );
        let mut out: Vec<Entry> = Vec::new();
        let mut add = |hf: &str, ours: &str, shape: &[usize], edit: Edit| {
            out.push((hf.into(), ours.into(), shape.to_vec(), edit));
        };
        let hf = "model.language_model";
        add(
            &format!("{hf}.embed_tokens.weight"),
            "model.embed_tokens.weight",
            &[v, h],
            Edit::Keep,
        );
        add(
            &format!("{hf}.norm.weight"),
            "model.norm.weight",
            &[h],
            Edit::Norm,
        );
        add("lm_head.weight", "lm_head.weight", &[v, h], Edit::Keep);
        for i in 0..config.num_layers {
            let (a, b) = (format!("{hf}.layers.{i}"), format!("model.layers.{i}"));
            let mut layer = |s: &str, t: &str, shape: &[usize], edit: Edit| {
                add(&format!("{a}.{s}"), &format!("{b}.{t}"), shape, edit);
            };
            for n in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                layer(n, n, &[h], Edit::Norm);
            }
            for (n, shape) in [
                ("mlp.gate_proj.weight", [inter, h]),
                ("mlp.up_proj.weight", [inter, h]),
                ("mlp.down_proj.weight", [h, inter]),
            ] {
                layer(n, n, &shape, Edit::Keep);
            }
            if config.hybrid_layers.as_ref().unwrap().is_attention_layer(i) {
                for (n, shape, edit) in [
                    ("q_proj.weight", vec![attn.q_gate_dim(), h], Edit::Keep),
                    ("k_proj.weight", vec![attn.kv_dim(), h], Edit::Keep),
                    ("v_proj.weight", vec![attn.kv_dim(), h], Edit::Keep),
                    ("o_proj.weight", vec![h, attn.q_dim()], Edit::Keep),
                    ("q_norm.weight", vec![attn.head_dim], Edit::Norm),
                    ("k_norm.weight", vec![attn.head_dim], Edit::Norm),
                ] {
                    let n = format!("self_attn.{n}");
                    layer(&n, &n, &shape, edit);
                }
                continue;
            }
            let (qkv, vd, hv) = (gdn.qkv_dim(), gdn.value_dim(), gdn.value_heads);
            for (s, t, shape, edit) in [
                (
                    "in_proj_qkv.weight",
                    "in_proj_qkv.weight",
                    vec![qkv, h],
                    Edit::Tail,
                ),
                (
                    "conv1d.weight",
                    "conv1d.weight",
                    vec![qkv, 1, gdn.conv_kernel],
                    Edit::Tail,
                ),
                (
                    "in_proj_z.weight",
                    "in_proj_z.weight",
                    vec![vd, h],
                    Edit::Rows(gdn.state_size),
                ),
                (
                    "in_proj_a.weight",
                    "alpha_proj.weight",
                    vec![hv, h],
                    Edit::Rows(1),
                ),
                (
                    "in_proj_b.weight",
                    "beta_proj.weight",
                    vec![hv, h],
                    Edit::Rows(1),
                ),
                ("A_log", "a_neg_exp", vec![hv], Edit::ALog),
                ("dt_bias", "dt_bias", vec![hv], Edit::Rows(1)),
                (
                    "norm.weight",
                    "norm.weight",
                    vec![gdn.state_size],
                    Edit::Keep,
                ),
                (
                    "out_proj.weight",
                    "out_proj.weight",
                    vec![h, vd],
                    Edit::Cols,
                ),
            ] {
                layer(
                    &format!("linear_attn.{s}"),
                    &format!("linear_attn.{t}"),
                    &shape,
                    edit,
                );
            }
        }
        out
    }

    /// Source position of each tiled position: tiled head `r * h_k + k`
    /// reads grouped head `k * rep + r`.
    fn tiled_from_grouped(h_k: usize, rep: usize, hd: usize, offset: usize) -> Vec<i64> {
        let mut out = Vec::with_capacity(h_k * rep * hd);
        for r in 0..rep {
            for k in 0..h_k {
                for d in 0..hd {
                    out.push((offset + (k * rep + r) * hd + d) as i64);
                }
            }
        }
        out
    }

    fn select(
        client: &CpuClient,
        t: &Tensor<CpuRuntime>,
        dim: usize,
        index: Vec<i64>,
    ) -> Tensor<CpuRuntime> {
        let index = Tensor::<CpuRuntime>::from_slice(&index, &[index.len()], t.device()).unwrap();
        client.index_select(t, dim, &index).unwrap()
    }

    /// The loader-layout tensor for one entry, built from the same values.
    fn expected(
        client: &CpuClient,
        gdn: &GdnConfig,
        t: &Tensor<CpuRuntime>,
        edit: Edit,
    ) -> Tensor<CpuRuntime> {
        let (h_k, rep, s) = (gdn.key_heads, gdn.head_repeat(), gdn.state_size);
        match edit {
            Edit::Keep => t.clone(),
            Edit::Norm => client.add_scalar(t, 1.0).unwrap(),
            Edit::Rows(hd) => select(client, t, 0, tiled_from_grouped(h_k, rep, hd, 0)),
            Edit::Tail => {
                let qk = 2 * gdn.key_dim();
                let mut index: Vec<i64> = (0..qk as i64).collect();
                index.extend(tiled_from_grouped(h_k, rep, s, qk));
                select(client, t, 0, index)
            }
            Edit::Cols => select(client, t, 1, tiled_from_grouped(h_k, rep, s, 0)),
            Edit::ALog => {
                let a = select(client, t, 0, tiled_from_grouped(h_k, rep, 1, 0));
                client.neg(&client.exp(&a).unwrap()).unwrap()
            }
        }
    }

    /// The checkpoint in Hugging Face naming, plus one MTP and one vision
    /// tensor, and the same weights in the loader layout.
    pub(crate) fn maps(
        client: &CpuClient,
        device: &CpuDevice,
        config: &UniversalConfig,
    ) -> (VarMap<CpuRuntime>, VarMap<CpuRuntime>) {
        let gdn = config.gdn.as_ref().unwrap();
        let (mut hf, mut ours) = (VarMap::new(), VarMap::new());
        for (hf_name, our_name, shape, edit) in entries(config) {
            let t = tensor(&hf_name, &shape, device);
            ours.insert(our_name, expected(client, gdn, &t, edit));
            hf.insert(hf_name, t);
        }
        let h = config.hidden_size;
        for (name, shape) in [
            ("mtp.fc.weight", vec![h, 2 * h]),
            ("mtp.layers.0.input_layernorm.weight", vec![h]),
            ("model.visual.patch_embed.proj.weight", vec![4, 4]),
        ] {
            hf.insert(name.into(), tensor(name, &shape, device));
        }
        (hf, ours)
    }

    fn logits(
        client: &CpuClient,
        device: &CpuDevice,
        map: &mut VarMap<CpuRuntime>,
        config: &UniversalConfig,
    ) -> Vec<f32> {
        let mut vb = VarBuilder::new(map, device);
        let model = Qwen35Model::<CpuRuntime>::from_varbuilder(&mut vb, config).unwrap();
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

    #[test]
    fn hf_checkpoint_matches_loader_layout_bit_for_bit() {
        let (client, device) = cpu_setup();
        let config = tiny_config();
        let gdn = config.gdn.as_ref().unwrap();
        assert!(
            gdn.head_repeat() > 1,
            "the tiny config must exercise the reorder"
        );
        let (mut hf, mut ours) = maps(&client, &device, &config);

        restructure_hf_checkpoint(&mut hf, &config, &device).unwrap();
        assert!(hf.is_hf_restructured());
        let mut names: Vec<&str> = hf.names().collect();
        let mut want: Vec<&str> = ours.names().collect();
        names.sort_unstable();
        want.sort_unstable();
        assert_eq!(names, want, "mtp and vision tensors must leave the map");
        for name in want {
            let (got, exp) = (hf.get_tensor(name).unwrap(), ours.get_tensor(name).unwrap());
            assert_eq!(got.shape(), exp.shape(), "{name}");
            assert_eq!(got.to_vec::<f32>(), exp.to_vec::<f32>(), "{name}");
        }

        let from_hf = logits(&client, &device, &mut hf, &config);
        let direct = logits(&client, &device, &mut ours, &config);
        assert_eq!(hf.len(), 0, "loader left tensors behind");
        assert!(from_hf.iter().all(|v| v.is_finite()));
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&from_hf), bits(&direct));
    }

    #[test]
    fn missing_a_log_names_the_tensor() {
        let (client, device) = cpu_setup();
        let config = tiny_config();
        let (mut hf, _) = maps(&client, &device, &config);
        hf.remove("model.language_model.layers.0.linear_attn.A_log");
        let err = restructure(&mut hf, &config).unwrap_err().to_string();
        assert!(
            err.contains("model.layers.0.linear_attn.a_neg_exp"),
            "{err}"
        );
    }

    #[test]
    fn second_run_is_an_error() {
        let (client, device) = cpu_setup();
        let config = tiny_config();
        let (mut hf, _) = maps(&client, &device, &config);
        restructure(&mut hf, &config).unwrap();
        assert!(restructure(&mut hf, &config).is_err());
        assert!(restructure_hf_checkpoint(&mut hf, &config, &device).is_err());
    }

    #[test]
    fn wrong_qkv_rows_is_an_error() {
        let (client, device) = cpu_setup();
        let config = tiny_config();
        let (mut hf, _) = maps(&client, &device, &config);
        let name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        hf.insert(name.into(), tensor(name, &[3, config.hidden_size], &device));
        let err = restructure(&mut hf, &config).unwrap_err().to_string();
        assert!(err.contains("in_proj_qkv"), "{err}");
    }
}
