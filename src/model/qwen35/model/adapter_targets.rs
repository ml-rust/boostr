//! PEFT adapter modules on `qwen35` projections: the projection a
//! normalized module name (`model.layers.N.<sub>`, see
//! [`parse_adapter_key`](crate::format::peft::parse_adapter_key)) targets,
//! the base shape its factors must fit, and the factors moved from the
//! Hugging Face layout to the loader layout.
//!
//! # Layout
//!
//! A PEFT adapter for Qwen3.5 is trained on the Hugging Face checkpoint, so
//! its factors follow that layout whatever the base came from. The loader
//! layout differs from it by the Gated DeltaNet value-head reorder, grouped
//! to tiled (`restructure` applies it to a Hugging Face base, the llama.cpp
//! converter to a GGUF one). A base row permutation `P` turns `W + s B A`
//! into `P W + s (P B) A`, a base column permutation into
//! `W P^T + s B (A P^T)`, so the adapter takes the move the base took:
//!
//! - `in_proj_qkv`: `lora_B` V rows (after `2 * key_dim`), Q and K rows kept
//! - `in_proj_z`: `lora_B` rows, value heads of width `state_size`
//! - `in_proj_a`, `in_proj_b`: `lora_B` rows, value heads of width 1
//! - `out_proj`: `lora_A` columns, unless the base keeps grouped columns
//!   (`v_grouped`)
//!
//! The `1 + w` norm offset and `-exp(A_log)` touch no adapted tensor.
//! `q_proj` keeps its per-head `[query | gate]` rows in both layouts; the
//! attention block regroups its adapter's rows with its base's.

use crate::error::{Error, Result};
use crate::format::hf_qwen35::{reorder_qkv_v_rows, reorder_v_heads};
use crate::format::peft::LoraPair;
use crate::model::config::UniversalConfig;
use crate::model::hybrid::{GdnProjection, Qwen35AttentionProjection};
use crate::nn::{LoraFactors, MlpProjection};
use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::{Device, Runtime};

/// The projection an adapter module targets within its layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Projection {
    Attention(Qwen35AttentionProjection),
    Gdn(GdnProjection),
    /// `linear_attn.in_proj_a`: the loader's `alpha_proj`, `ssm_alpha`.
    GdnAlpha,
    /// `linear_attn.in_proj_b`: the loader's `beta_proj`, `ssm_beta`.
    GdnBeta,
    Mlp(MlpProjection),
}

/// One adapter module resolved to a layer and a projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Target {
    pub(super) layer: usize,
    pub(super) projection: Projection,
}

/// Hugging Face submodule names under `model.layers.N.`, by projection.
const SUBMODULES: [(&str, Projection); 12] = [
    (
        "self_attn.q_proj",
        Projection::Attention(Qwen35AttentionProjection::Q),
    ),
    (
        "self_attn.k_proj",
        Projection::Attention(Qwen35AttentionProjection::K),
    ),
    (
        "self_attn.v_proj",
        Projection::Attention(Qwen35AttentionProjection::V),
    ),
    (
        "self_attn.o_proj",
        Projection::Attention(Qwen35AttentionProjection::Output),
    ),
    (
        "linear_attn.in_proj_qkv",
        Projection::Gdn(GdnProjection::AttnQkv),
    ),
    (
        "linear_attn.in_proj_z",
        Projection::Gdn(GdnProjection::AttnGate),
    ),
    (
        "linear_attn.out_proj",
        Projection::Gdn(GdnProjection::SsmOut),
    ),
    ("linear_attn.in_proj_a", Projection::GdnAlpha),
    ("linear_attn.in_proj_b", Projection::GdnBeta),
    ("mlp.gate_proj", Projection::Mlp(MlpProjection::Gate)),
    ("mlp.up_proj", Projection::Mlp(MlpProjection::Up)),
    ("mlp.down_proj", Projection::Mlp(MlpProjection::Down)),
];

const LAYER_PREFIX: &str = "model.layers.";

fn no_projection(module: &str, why: String) -> Error {
    Error::ModelError {
        reason: format!("PEFT adapter module `{module}` matches no qwen35 projection: {why}"),
    }
}

/// The layer and projection `module` targets.
///
/// # Errors
///
/// [`Error::ModelError`] naming `module` when it is not
/// `model.layers.N.<sub>` with `<sub>` an adaptable projection, `N` is past
/// the last layer, or `<sub>` belongs to the other mixer kind than layer
/// `N` has.
pub(super) fn resolve(module: &str, config: &UniversalConfig) -> Result<Target> {
    let (index, sub) = module
        .strip_prefix(LAYER_PREFIX)
        .and_then(|rest| rest.split_once('.'))
        .ok_or_else(|| no_projection(module, format!("expected `{LAYER_PREFIX}N.<module>`")))?;
    let layer: usize = index
        .parse()
        .map_err(|_| no_projection(module, format!("`{index}` is not a layer index")))?;
    if layer >= config.num_layers {
        return Err(no_projection(
            module,
            format!("layer {layer} is past num_layers = {}", config.num_layers),
        ));
    }
    let projection = SUBMODULES
        .iter()
        .find(|(name, _)| *name == sub)
        .map(|(_, projection)| *projection)
        .ok_or_else(|| {
            let known: Vec<&str> = SUBMODULES.iter().map(|(name, _)| *name).collect();
            no_projection(
                module,
                format!("`{sub}` is not one of {}", known.join(", ")),
            )
        })?;
    let attention_layer = config
        .hybrid_layers
        .as_ref()
        .ok_or_else(|| no_projection(module, "the config has no hybrid_layers".into()))?
        .is_attention_layer(layer);
    let wants_attention = match projection {
        Projection::Attention(_) => Some(true),
        Projection::Gdn(_) | Projection::GdnAlpha | Projection::GdnBeta => Some(false),
        Projection::Mlp(_) => None,
    };
    if let Some(wants) = wants_attention
        && wants != attention_layer
    {
        let kind = if attention_layer {
            "a full-attention"
        } else {
            "a Gated DeltaNet"
        };
        return Err(no_projection(
            module,
            format!("layer {layer} is {kind} layer, which has no `{sub}`"),
        ));
    }
    Ok(Target { layer, projection })
}

impl Projection {
    /// `[out_features, in_features]` of the base this projection holds.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when `config` lacks the `gdn` or
    /// `qwen35_attention` section the projection reads.
    pub(super) fn base_shape(self, config: &UniversalConfig) -> Result<[usize; 2]> {
        let hidden = config.hidden_size;
        let missing = |section: &str| Error::ModelError {
            reason: format!("qwen35 requires a {section} config"),
        };
        let gdn = || config.gdn.as_ref().ok_or_else(|| missing("gdn"));
        let attn = || {
            config
                .qwen35_attention
                .as_ref()
                .ok_or_else(|| missing("qwen35_attention"))
        };
        Ok(match self {
            Self::Attention(Qwen35AttentionProjection::Q) => [attn()?.q_gate_dim(), hidden],
            Self::Attention(Qwen35AttentionProjection::K | Qwen35AttentionProjection::V) => {
                [attn()?.kv_dim(), hidden]
            }
            Self::Attention(Qwen35AttentionProjection::Output) => [hidden, attn()?.q_dim()],
            Self::Gdn(GdnProjection::AttnQkv) => [gdn()?.qkv_dim(), hidden],
            Self::Gdn(GdnProjection::AttnGate) => [gdn()?.value_dim(), hidden],
            Self::Gdn(GdnProjection::SsmOut) => [hidden, gdn()?.value_dim()],
            Self::GdnAlpha | Self::GdnBeta => [gdn()?.value_heads, hidden],
            Self::Mlp(MlpProjection::Gate | MlpProjection::Up) => {
                [config.intermediate_size(), hidden]
            }
            Self::Mlp(MlpProjection::Down) => [hidden, config.intermediate_size()],
        })
    }
}

/// Check that `pair` fits a base of shape `[out, in]` on `device`:
/// `lora_A` `[r, in]`, `lora_B` `[out, r]`, `r > 0`.
///
/// # Errors
///
/// [`Error::ModelError`] naming `module`, both factor shapes and the base
/// shape, or the two devices.
pub(super) fn check_pair<R: Runtime>(
    module: &str,
    pair: &LoraPair<R>,
    [out, inp]: [usize; 2],
    device: &R::Device,
) -> Result<()> {
    let (a, b) = (pair.a.shape(), pair.b.shape());
    let fits = matches!(
        (a, b),
        ([r, ai], [bo, br]) if *r > 0 && *ai == inp && *bo == out && br == r
    );
    if !fits {
        return Err(Error::ModelError {
            reason: format!(
                "PEFT adapter module `{module}`: lora_A {a:?} and lora_B {b:?} do not fit the \
                 base projection [{out}, {inp}], which needs lora_A [r, {inp}] and lora_B \
                 [{out}, r]"
            ),
        });
    }
    for (name, factor) in [("lora_A", &pair.a), ("lora_B", &pair.b)] {
        if !factor.device().is_same(device) {
            return Err(Error::ModelError {
                reason: format!(
                    "PEFT adapter module `{module}`: {name} is on {} but the model is on {}; \
                     load the adapter onto the model's device",
                    factor.device().name(),
                    device.name()
                ),
            });
        }
    }
    Ok(())
}

/// `pair` moved from the Hugging Face layout to the loader layout of
/// `projection` (see the module doc), as [`LoraFactors`] with the pair's
/// scaling. Factors that need no move are shared, not copied.
///
/// # Errors
///
/// [`Error::ModelError`] when `config` has no `gdn` section and the
/// projection needs one; reorder errors propagate.
pub(super) fn to_loader_layout<R: Runtime<DType = DType>>(
    projection: Projection,
    pair: &LoraPair<R>,
    config: &UniversalConfig,
) -> Result<LoraFactors<R>>
where
    R::Client: ShapeOps<R>,
{
    let (mut lora_a, mut lora_b) = (pair.a.clone(), pair.b.clone());
    let gdn_projection = matches!(
        projection,
        Projection::Gdn(_) | Projection::GdnAlpha | Projection::GdnBeta
    );
    if gdn_projection {
        let gdn = config.gdn.as_ref().ok_or_else(|| Error::ModelError {
            reason: "qwen35 requires a gdn config".into(),
        })?;
        let (h_k, rep, s) = (gdn.key_heads, gdn.head_repeat(), gdn.state_size);
        match projection {
            Projection::Gdn(GdnProjection::AttnQkv) => lora_b = reorder_qkv_v_rows(&lora_b, gdn)?,
            Projection::Gdn(GdnProjection::AttnGate) => {
                lora_b = reorder_v_heads(&lora_b, 0, h_k, rep, s)?;
            }
            Projection::Gdn(GdnProjection::SsmOut) if !gdn.v_grouped => {
                lora_a = reorder_v_heads(&lora_a, 1, h_k, rep, s)?;
            }
            Projection::GdnAlpha | Projection::GdnBeta => {
                lora_b = reorder_v_heads(&lora_b, 0, h_k, rep, 1)?;
            }
            _ => {}
        }
    }
    Ok(LoraFactors {
        lora_a,
        lora_b,
        scaling: pair.scaling,
    })
}

#[cfg(test)]
mod tests {
    use super::super::tiny_config;
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    fn resolved(module: &str) -> Target {
        resolve(module, &tiny_config()).unwrap()
    }

    fn refused(module: &str) -> String {
        let text = resolve(module, &tiny_config()).unwrap_err().to_string();
        assert!(text.contains(module), "{text}");
        assert!(text.contains("matches no qwen35 projection"), "{text}");
        text
    }

    /// `tiny_config`: layer 0 Gated DeltaNet, layer 1 full attention.
    #[test]
    fn every_submodule_resolves_on_its_layer_kind() {
        assert_eq!(
            resolved("model.layers.1.self_attn.q_proj"),
            Target {
                layer: 1,
                projection: Projection::Attention(Qwen35AttentionProjection::Q)
            }
        );
        assert_eq!(
            resolved("model.layers.0.linear_attn.in_proj_a").projection,
            Projection::GdnAlpha
        );
        assert_eq!(
            resolved("model.layers.0.linear_attn.in_proj_b").projection,
            Projection::GdnBeta
        );
        assert_eq!(
            resolved("model.layers.0.linear_attn.out_proj").projection,
            Projection::Gdn(GdnProjection::SsmOut)
        );
        for layer in 0..2 {
            assert_eq!(
                resolved(&format!("model.layers.{layer}.mlp.down_proj")).projection,
                Projection::Mlp(MlpProjection::Down)
            );
        }
    }

    #[test]
    fn unknown_or_misplaced_modules_are_refused() {
        refused("model.layers.0.self_attn.q_proj");
        refused("model.layers.1.linear_attn.in_proj_qkv");
        assert!(refused("model.layers.2.mlp.up_proj").contains("past num_layers"));
        assert!(refused("model.layers.0.linear_attn.conv1d").contains("is not one of"));
        refused("model.layers.0.linear_attn.in_proj_ba");
        refused("model.embed_tokens");
        refused("model.layers.x.mlp.up_proj");
    }

    #[test]
    fn base_shapes_follow_the_config() {
        let config = tiny_config();
        let gdn = config.gdn.clone().unwrap();
        let attn = config.qwen35_attention.clone().unwrap();
        let h = config.hidden_size;
        let shape = |m: &str| resolved(m).projection.base_shape(&config).unwrap();
        assert_eq!(
            shape("model.layers.1.self_attn.q_proj"),
            [attn.q_gate_dim(), h]
        );
        assert_eq!(shape("model.layers.1.self_attn.o_proj"), [h, attn.q_dim()]);
        assert_eq!(
            shape("model.layers.0.linear_attn.in_proj_qkv"),
            [gdn.qkv_dim(), h]
        );
        assert_eq!(
            shape("model.layers.0.linear_attn.out_proj"),
            [h, gdn.value_dim()]
        );
        assert_eq!(
            shape("model.layers.0.linear_attn.in_proj_b"),
            [gdn.value_heads, h]
        );
        assert_eq!(
            shape("model.layers.0.mlp.gate_proj"),
            [config.intermediate_size(), h]
        );
    }

    fn pair(a: &[usize], b: &[usize], device: &CpuDevice) -> LoraPair<CpuRuntime> {
        let iota = |shape: &[usize]| {
            let n: usize = shape.iter().product();
            let v: Vec<f32> = (0..n).map(|i| i as f32).collect();
            Tensor::<CpuRuntime>::from_slice(&v, shape, device).unwrap()
        };
        LoraPair {
            a: iota(a),
            b: iota(b),
            scaling: 2.0,
        }
    }

    #[test]
    fn pair_shapes_are_checked_against_the_base() {
        let device = CpuDevice::new();
        check_pair("m", &pair(&[2, 8], &[4, 2], &device), [4, 8], &device).unwrap();
        for (a, b) in [
            (vec![2, 7], vec![4, 2]),
            (vec![2, 8], vec![5, 2]),
            (vec![2, 8], vec![4, 3]),
            (vec![0, 8], vec![4, 0]),
        ] {
            let text = check_pair("mod.x", &pair(&a, &b, &device), [4, 8], &device)
                .unwrap_err()
                .to_string();
            assert!(text.contains("mod.x"), "{text}");
            assert!(text.contains(&format!("{a:?}")), "{text}");
            assert!(text.contains(&format!("{b:?}")), "{text}");
            assert!(text.contains("[4, 8]"), "{text}");
        }
    }

    /// `tiny_config` GDN: `H_k = 2`, two value heads each, `state_size = 4`.
    /// Grouped value head `k * 2 + r` lands at tiled head `r * 2 + k`, so
    /// tiled heads read grouped heads `0, 2, 1, 3`.
    #[test]
    fn gdn_factors_take_the_value_head_move() {
        let device = CpuDevice::new();
        let config = tiny_config();
        let gdn = config.gdn.clone().unwrap();
        let s = gdn.state_size;
        let tiled_heads = [0usize, 2, 1, 3];
        let rank = 2;

        let z = pair(&[rank, 8], &[gdn.value_dim(), rank], &device);
        let moved =
            to_loader_layout(Projection::Gdn(GdnProjection::AttnGate), &z, &config).unwrap();
        assert_eq!(moved.lora_a.to_vec::<f32>(), z.a.to_vec::<f32>());
        let (got, src) = (moved.lora_b.to_vec::<f32>(), z.b.to_vec::<f32>());
        for (head, &from) in tiled_heads.iter().enumerate() {
            let w = s * rank;
            assert_eq!(
                &got[head * w..(head + 1) * w],
                &src[from * w..(from + 1) * w]
            );
        }
        assert_eq!(moved.scaling, 2.0);

        let qkv = pair(&[rank, 8], &[gdn.qkv_dim(), rank], &device);
        let moved =
            to_loader_layout(Projection::Gdn(GdnProjection::AttnQkv), &qkv, &config).unwrap();
        let (got, src) = (moved.lora_b.to_vec::<f32>(), qkv.b.to_vec::<f32>());
        let qk = 2 * gdn.key_dim() * rank;
        assert_eq!(&got[..qk], &src[..qk], "q and k rows stay");
        for (head, &from) in tiled_heads.iter().enumerate() {
            let w = s * rank;
            assert_eq!(
                &got[qk + head * w..qk + (head + 1) * w],
                &src[qk + from * w..qk + (from + 1) * w]
            );
        }

        let alpha = pair(&[rank, 8], &[gdn.value_heads, rank], &device);
        let moved = to_loader_layout(Projection::GdnAlpha, &alpha, &config).unwrap();
        let (got, src) = (moved.lora_b.to_vec::<f32>(), alpha.b.to_vec::<f32>());
        for (head, &from) in tiled_heads.iter().enumerate() {
            assert_eq!(
                &got[head * rank..(head + 1) * rank],
                &src[from * rank..(from + 1) * rank]
            );
        }

        let out = pair(&[rank, gdn.value_dim()], &[8, rank], &device);
        let moved =
            to_loader_layout(Projection::Gdn(GdnProjection::SsmOut), &out, &config).unwrap();
        assert_eq!(moved.lora_b.to_vec::<f32>(), out.b.to_vec::<f32>());
        let (got, src) = (moved.lora_a.to_vec::<f32>(), out.a.to_vec::<f32>());
        let cols = gdn.value_dim();
        for row in 0..rank {
            for (head, &from) in tiled_heads.iter().enumerate() {
                assert_eq!(
                    &got[row * cols + head * s..row * cols + (head + 1) * s],
                    &src[row * cols + from * s..row * cols + (from + 1) * s]
                );
            }
        }

        let mut grouped = config.clone();
        if let Some(g) = grouped.gdn.as_mut() {
            g.v_grouped = true;
        }
        let kept =
            to_loader_layout(Projection::Gdn(GdnProjection::SsmOut), &out, &grouped).unwrap();
        assert_eq!(kept.lora_a.to_vec::<f32>(), out.a.to_vec::<f32>());

        let q = pair(&[rank, 8], &[32, rank], &device);
        let kept = to_loader_layout(
            Projection::Attention(Qwen35AttentionProjection::Q),
            &q,
            &config,
        )
        .unwrap();
        assert_eq!(kept.lora_b.to_vec::<f32>(), q.b.to_vec::<f32>());
    }
}
