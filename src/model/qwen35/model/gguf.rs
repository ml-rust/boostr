//! [`Qwen35Model::from_varbuilder`]: build the model from a `VarMap` in the
//! `qwen35` loader layout.
//!
//! Two sources fill that layout:
//!
//! - [`VarMap::from_gguf`](crate::nn::VarMap::from_gguf) over a `qwen35`
//!   GGUF. Names are the HF-style ones
//!   `gguf_to_hf_name_for_arch(Some("qwen35"), _)` produces.
//! - A Hugging Face Qwen3.5 SafeTensors checkpoint after
//!   [`restructure_hf_checkpoint`](crate::format::restructure_hf_checkpoint).
//!
//! # Activation dtype
//!
//! Every Hadamard rotation is built in `F32`. `VarMap::from_gguf` decodes
//! each dense tensor to `F32` through `Gguf::load_tensor_f32`, and the
//! quantized matmul path returns `F32`, so `F32` is the dtype every
//! activation carries on the GGUF path. A SafeTensors map carries the dtype
//! its loader gave the tensors.
//!
//! The Hadamard attach lives in `attach`, per-layer builders in
//! `gguf_layers`.

use super::attach::Attach;
use super::build::{Qwen35Block, Qwen35Model};
use super::gguf_layers::{attention_layer, gdn_layer, rms_norm};
use crate::error::{Error, Result};
use crate::model::config::UniversalConfig;
use crate::nn::{MaybeQuantEmbedding, MaybeQuantLinear, VarBuilder, Weight};
use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::Runtime;

/// The checkpoint key of the token embedding, which a tied head reads.
const EMBED_NAME: &str = "model.embed_tokens.weight";

/// A second handle on `weight`'s storage. `Tensor::clone` and
/// `QuantTensor::clone` share it; nothing is copied.
fn share_weight<R: Runtime>(weight: &Weight<R>) -> Result<Weight<R>> {
    match weight {
        Weight::Standard(t) => Ok(Weight::Standard(t.clone())),
        Weight::Quantized(q) => Ok(Weight::Quantized(q.clone())),
        Weight::DecomposedQuant(_) => Err(Error::ModelError {
            reason: format!(
                "qwen35: {EMBED_NAME} is AWQ/GPTQ-decomposed; a tied head needs it dense \
                 or block-quantized"
            ),
        }),
    }
}

/// The tied `lm_head` over the embedding's storage, bound to the
/// embedding's checkpoint key for an importance collection.
fn tied_lm_head<R: Runtime>(weight: Weight<R>) -> MaybeQuantLinear<R> {
    let head = MaybeQuantLinear::from_weight(weight, None);
    if let MaybeQuantLinear::Standard(linear) = &head {
        crate::quant::imatrix::register_name(linear.weight().id(), EMBED_NAME);
    }
    head
}

impl<R: Runtime<DType = DType>> Qwen35Model<R> {
    /// Build from a `VarBuilder` over a `VarMap` in the loader layout.
    ///
    /// Reads `hadamard` from `config`; `None` builds every projection
    /// `Plain`. With `tie_word_embeddings` set and no `lm_head.weight`, the
    /// head shares the embedding's storage. Without the flag,
    /// `lm_head.weight` must be present.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when `config` fails validation, lacks `gdn`,
    /// `qwen35_attention` or `hybrid_layers`, a tensor is missing or has
    /// the wrong shape, or a rotation's sign vector is missing for a
    /// width. A tied head with a Hadamard contract is an error.
    /// `Qwen35Model::new` errors propagate.
    pub fn from_varbuilder(vb: &mut VarBuilder<R>, config: &UniversalConfig) -> Result<Self>
    where
        R::Client: ShapeOps<R>,
    {
        config.validate()?;
        let gdn_config = config.gdn.as_ref().ok_or_else(|| Error::ModelError {
            reason: "qwen35 requires a gdn config".into(),
        })?;
        let attention_config =
            config
                .qwen35_attention
                .as_ref()
                .ok_or_else(|| Error::ModelError {
                    reason: "qwen35 requires a qwen35_attention config".into(),
                })?;
        let layers = config
            .hybrid_layers
            .as_ref()
            .ok_or_else(|| Error::ModelError {
                reason: "qwen35 requires hybrid_layers (attention vs gdn per layer)".into(),
            })?;
        layers.validate(config.num_layers)?;

        let device = vb.device().clone();
        let attach = Attach::<R>::new(config.hadamard.as_ref(), &device);
        let hidden_size = config.hidden_size;
        let eps = config.rms_norm_eps as f32;

        let rope = attention_config.rope_table::<R>(config.max_seq_len, &device)?;

        let tied = config.tie_word_embeddings && !vb.contains("lm_head.weight");
        if tied && config.hadamard.is_some() {
            return Err(Error::ModelError {
                reason: "qwen35: tie_word_embeddings is set, lm_head.weight is absent, and a \
                         Hadamard contract is present; a tied head under a Hadamard contract \
                         is not supported"
                    .into(),
            });
        }

        let mut model_vb = vb.pp("model");

        let embed_weight = model_vb.take_weight("embed_tokens.weight")?;
        let tied_head = if tied {
            Some(share_weight(&embed_weight)?)
        } else {
            None
        };
        let embed_inner = MaybeQuantEmbedding::from_weight(embed_weight, false)?;
        let embed_tokens = attach.embedding("token_embd.weight", embed_inner, hidden_size)?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let mut layers_vb = model_vb.pp("layers");
            let mut layer_vb = layers_vb.pp(&i.to_string());
            let block = if layers.is_attention_layer(i) {
                Qwen35Block::Attention(Box::new(attention_layer(
                    &mut layer_vb,
                    &attach,
                    i,
                    attention_config,
                    eps,
                )?))
            } else {
                Qwen35Block::Gdn(Box::new(gdn_layer(
                    &mut layer_vb,
                    &attach,
                    i,
                    gdn_config,
                    eps,
                )?))
            };
            blocks.push(block);
        }

        let norm = rms_norm(&mut model_vb, "norm.weight", eps)?;

        let lm_head_inner = match tied_head {
            Some(weight) => tied_lm_head(weight),
            None => vb.take_maybe_quant_linear("lm_head.weight", None)?,
        };
        let lm_head = attach.linear("output.weight", lm_head_inner)?;

        Self::new(config.clone(), embed_tokens, blocks, norm, lm_head, rope)
    }
}

#[cfg(test)]
mod tests {
    use super::super::build::tests::{HIDDEN, VOCAB, tiny_config};
    use super::*;
    use crate::format::gguf::HadamardContract;
    use crate::model::config::{GdnConfig, Qwen35AttentionConfig};
    use crate::nn::{MaybeRotatedLinear, VarMap};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    fn insert(map: &mut VarMap<CpuRuntime>, device: &CpuDevice, name: &str, shape: &[usize]) {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
        map.insert(
            name.into(),
            Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap(),
        );
    }

    /// Every tensor `from_varbuilder` reads for `tiny_config()` (layer 0
    /// gdn, layer 1 attention), under the mapped HF names.
    fn fill(map: &mut VarMap<CpuRuntime>, device: &CpuDevice, config: &UniversalConfig) {
        let gdn: &GdnConfig = config.gdn.as_ref().unwrap();
        let attn: &Qwen35AttentionConfig = config.qwen35_attention.as_ref().unwrap();
        let inter = config.intermediate_size();
        let h = HIDDEN;
        insert(map, device, "model.embed_tokens.weight", &[VOCAB, h]);
        insert(map, device, "model.norm.weight", &[h]);
        insert(map, device, "lm_head.weight", &[VOCAB, h]);
        for i in 0..config.num_layers {
            let p = format!("model.layers.{i}");
            insert(map, device, &format!("{p}.input_layernorm.weight"), &[h]);
            insert(
                map,
                device,
                &format!("{p}.post_attention_layernorm.weight"),
                &[h],
            );
            insert(
                map,
                device,
                &format!("{p}.mlp.gate_proj.weight"),
                &[inter, h],
            );
            insert(map, device, &format!("{p}.mlp.up_proj.weight"), &[inter, h]);
            insert(
                map,
                device,
                &format!("{p}.mlp.down_proj.weight"),
                &[h, inter],
            );
        }
        let g = "model.layers.0.linear_attn";
        insert(
            map,
            device,
            &format!("{g}.in_proj_qkv.weight"),
            &[gdn.qkv_dim(), h],
        );
        insert(
            map,
            device,
            &format!("{g}.in_proj_z.weight"),
            &[gdn.value_dim(), h],
        );
        insert(
            map,
            device,
            &format!("{g}.conv1d.weight"),
            &[gdn.qkv_dim(), gdn.conv_kernel],
        );
        insert(
            map,
            device,
            &format!("{g}.alpha_proj.weight"),
            &[gdn.value_heads, h],
        );
        insert(
            map,
            device,
            &format!("{g}.beta_proj.weight"),
            &[gdn.value_heads, h],
        );
        insert(map, device, &format!("{g}.a_neg_exp"), &[gdn.value_heads]);
        insert(map, device, &format!("{g}.dt_bias"), &[gdn.value_heads]);
        insert(map, device, &format!("{g}.norm.weight"), &[gdn.state_size]);
        insert(
            map,
            device,
            &format!("{g}.out_proj.weight"),
            &[h, gdn.value_dim()],
        );
        let a = "model.layers.1.self_attn";
        insert(
            map,
            device,
            &format!("{a}.q_proj.weight"),
            &[attn.q_gate_dim(), h],
        );
        insert(
            map,
            device,
            &format!("{a}.k_proj.weight"),
            &[attn.kv_dim(), h],
        );
        insert(
            map,
            device,
            &format!("{a}.v_proj.weight"),
            &[attn.kv_dim(), h],
        );
        insert(
            map,
            device,
            &format!("{a}.o_proj.weight"),
            &[h, attn.q_dim()],
        );
        insert(map, device, &format!("{a}.q_norm.weight"), &[attn.head_dim]);
        insert(map, device, &format!("{a}.k_norm.weight"), &[attn.head_dim]);
    }

    #[test]
    fn builds_from_mapped_names_and_drains_the_map() {
        let (_client, device) = cpu_setup();
        let config = tiny_config();
        let mut map = VarMap::<CpuRuntime>::new();
        fill(&mut map, &device, &config);
        let mut vb = VarBuilder::new(&mut map, &device);
        let model = Qwen35Model::<CpuRuntime>::from_varbuilder(&mut vb, &config).unwrap();
        assert_eq!(model.num_gdn_layers(), 1);
        assert_eq!(model.num_attention_layers(), 1);
        assert!(!model.embed_tokens.is_rotated());
        assert!(!model.lm_head.is_rotated());
        assert_eq!(map.len(), 0, "loader left tensors behind");
    }

    /// A `prism.hadamard.*` block naming `output.weight`, layer 0's
    /// `ffn_down` and layer 1's `attn_q` as rotated, and `token_embd.weight`
    /// as inverse. Sign widths 8 (`HIDDEN`) and 16 (`INTER`), block 4.
    fn hadamard_contract() -> HadamardContract {
        use crate::format::gguf::{GgufMetadata, GgufValue};
        let strings = |names: &[&str]| {
            GgufValue::Array(
                names
                    .iter()
                    .map(|s| GgufValue::String((*s).into()))
                    .collect(),
            )
        };
        let ints =
            |vals: Vec<i32>| GgufValue::Array(vals.into_iter().map(GgufValue::Int32).collect());
        let signs: Vec<i32> = (0..24).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
        let mut m = GgufMetadata::default();
        for (k, v) in [
            ("prism.hadamard.version", GgufValue::Uint32(1)),
            ("prism.hadamard.block_size", GgufValue::Uint32(4)),
            (
                "prism.hadamard.transform",
                GgufValue::String("normalized-sylvester-walsh-hadamard".into()),
            ),
            (
                "prism.hadamard.axis",
                GgufValue::String("input-last-dimension".into()),
            ),
            (
                "prism.hadamard.sign_mode",
                GgufValue::String("explicit".into()),
            ),
            (
                "prism.hadamard.weight_names",
                strings(&[
                    "output.weight",
                    "blk.0.ffn_down.weight",
                    "blk.1.attn_q.weight",
                ]),
            ),
            (
                "prism.hadamard.inverse_weight_names",
                strings(&["token_embd.weight"]),
            ),
            ("prism.hadamard.sign_widths", ints(vec![8, 16])),
            ("prism.hadamard.sign_values", ints(signs)),
        ] {
            m.kv.insert(k.into(), v);
        }
        HadamardContract::from_metadata(&m).unwrap().unwrap()
    }

    #[test]
    fn hadamard_contract_wraps_named_tensors_only() {
        let (_client, device) = cpu_setup();
        let mut config = tiny_config();
        config.hadamard = Some(hadamard_contract());
        let mut map = VarMap::<CpuRuntime>::new();
        fill(&mut map, &device, &config);
        let mut vb = VarBuilder::new(&mut map, &device);
        let model = Qwen35Model::<CpuRuntime>::from_varbuilder(&mut vb, &config).unwrap();
        assert!(model.embed_tokens.is_rotated());
        assert!(model.lm_head.is_rotated());
        let Qwen35Block::Gdn(gdn) = &model.blocks[0] else {
            panic!("layer 0 is gdn");
        };
        assert!(gdn.mlp.down().is_rotated());
        assert!(!gdn.mlp.gate().is_rotated());
        assert!(!gdn.mlp.up().is_rotated());
        let Qwen35Block::Attention(attn) = &model.blocks[1] else {
            panic!("layer 1 is attention");
        };
        assert!(!attn.mlp.down().is_rotated());
    }

    #[test]
    fn missing_tensor_is_an_error() {
        let (_client, device) = cpu_setup();
        let config = tiny_config();
        let mut map = VarMap::<CpuRuntime>::new();
        fill(&mut map, &device, &config);
        map.remove("model.layers.0.linear_attn.a_neg_exp");
        let mut vb = VarBuilder::new(&mut map, &device);
        assert!(Qwen35Model::<CpuRuntime>::from_varbuilder(&mut vb, &config).is_err());
    }

    #[test]
    fn tied_head_shares_the_embedding_storage() {
        let (_client, device) = cpu_setup();
        let mut config = tiny_config();
        config.tie_word_embeddings = true;
        let mut map = VarMap::<CpuRuntime>::new();
        fill(&mut map, &device, &config);
        map.remove("lm_head.weight");
        let embed = map.get_tensor("model.embed_tokens.weight").unwrap().clone();
        let mut vb = VarBuilder::new(&mut map, &device);
        let model = Qwen35Model::<CpuRuntime>::from_varbuilder(&mut vb, &config).unwrap();
        assert_eq!(map.len(), 0, "loader left tensors behind");
        let MaybeRotatedLinear::Plain(MaybeQuantLinear::Standard(head)) = &model.lm_head else {
            panic!("a tied head over a dense embedding is a plain dense linear");
        };
        let head = head.weight().tensor();
        assert_eq!(head.shape(), &[VOCAB, HIDDEN]);
        assert!(head.shares_storage_with(&embed));
        assert_eq!(head.to_vec::<f32>(), embed.to_vec::<f32>());
    }

    /// The flag alone ties nothing when the file carries its own head.
    #[test]
    fn tied_flag_with_lm_head_present_loads_the_head() {
        let (_client, device) = cpu_setup();
        let mut config = tiny_config();
        config.tie_word_embeddings = true;
        let mut map = VarMap::<CpuRuntime>::new();
        fill(&mut map, &device, &config);
        let mut vb = VarBuilder::new(&mut map, &device);
        Qwen35Model::<CpuRuntime>::from_varbuilder(&mut vb, &config).unwrap();
        assert_eq!(map.len(), 0, "lm_head.weight must be consumed");
    }

    #[test]
    fn tied_head_with_hadamard_contract_is_an_error() {
        let (_client, device) = cpu_setup();
        let mut config = tiny_config();
        config.tie_word_embeddings = true;
        config.hadamard = Some(hadamard_contract());
        let mut map = VarMap::<CpuRuntime>::new();
        fill(&mut map, &device, &config);
        map.remove("lm_head.weight");
        let mut vb = VarBuilder::new(&mut map, &device);
        let err = match Qwen35Model::<CpuRuntime>::from_varbuilder(&mut vb, &config) {
            Ok(_) => panic!("a tied head under a Hadamard contract must fail"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("Hadamard"), "{err}");
    }

    #[test]
    fn untied_config_without_lm_head_is_an_error() {
        let (_client, device) = cpu_setup();
        let config = tiny_config();
        let mut map = VarMap::<CpuRuntime>::new();
        fill(&mut map, &device, &config);
        map.remove("lm_head.weight");
        let mut vb = VarBuilder::new(&mut map, &device);
        assert!(Qwen35Model::<CpuRuntime>::from_varbuilder(&mut vb, &config).is_err());
    }
}
