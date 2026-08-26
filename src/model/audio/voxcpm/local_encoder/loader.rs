//! Loads `feat_encoder.*` tensors and assembles a [`LocalEncoder`].
//!
//! Verified key layout (112 tensors total; no `embed_tokens`, `vocab_size`
//! is 0). Layer indices are NUMERIC 0..`num_layers` — this loader requests
//! every tensor by explicit numeric index, so the safetensors header's
//! string-sorted order (0,1,10,11,2,...) never matters.
//!
//! ```text
//! feat_encoder.in_proj.{weight[1024,64],bias[1024]}
//! feat_encoder.special_token                                          [1,1,1,1024]
//! feat_encoder.encoder.layers.{i}.input_layernorm.weight              [1024]
//! feat_encoder.encoder.layers.{i}.self_attn.q_proj.weight             [2048,1024]
//! feat_encoder.encoder.layers.{i}.self_attn.k_proj.weight             [256,1024]
//! feat_encoder.encoder.layers.{i}.self_attn.v_proj.weight             [256,1024]
//! feat_encoder.encoder.layers.{i}.self_attn.o_proj.weight             [1024,2048]
//! feat_encoder.encoder.layers.{i}.post_attention_layernorm.weight     [1024]
//! feat_encoder.encoder.layers.{i}.mlp.gate_proj.weight                [4096,1024]
//! feat_encoder.encoder.layers.{i}.mlp.up_proj.weight                  [4096,1024]
//! feat_encoder.encoder.layers.{i}.mlp.down_proj.weight                [1024,4096]
//! feat_encoder.encoder.norm.weight                                    [1024]
//! ```
//! `in_proj` is the only biased `Linear` in the module — every attention and
//! MLP projection is bias-free.

use crate::error::Result;
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::model::audio::voxcpm::loader::support::TensorLoader;
use crate::model::audio::voxcpm::local_encoder::attention::LocalEncoderAttention;
use crate::model::audio::voxcpm::local_encoder::config::LocalEncoderConfig;
use crate::model::audio::voxcpm::local_encoder::encoder::LocalEncoder;
use crate::model::audio::voxcpm::local_encoder::layer::LocalEncoderLayer;
use crate::model::audio::voxcpm::local_encoder::mlp::LocalEncoderMlp;
use crate::model::audio::voxcpm::long_rope::build_long_rope_cache;
use crate::nn::{Linear, RmsNorm};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use std::path::Path;

/// Default top-level prefix for `feat_encoder`'s tensors in the VoxCPM2
/// checkpoint.
pub const DEFAULT_LOCAL_ENCODER_PREFIX: &str = "feat_encoder";

impl<R: Runtime<DType = DType>> LocalEncoder<R>
where
    R::Client: TypeConversionOps<R>,
{
    /// Load `feat_encoder` from a VoxCPM2 checkpoint using `cfg`'s
    /// architecture/RoPE parameters (see [`LocalEncoderConfig::from_config_json`]
    /// for reading those out of the checkpoint's `config.json`). `path` may
    /// be the `model.safetensors` file or its containing directory.
    /// `dtype`: cast every loaded tensor to this dtype (`None` keeps the
    /// checkpoint's own) — see
    /// [`checked_tensor`](crate::model::audio::voxcpm::loader::support::checked_tensor).
    pub fn from_safetensors<P: AsRef<Path>>(
        path: P,
        cfg: LocalEncoderConfig,
        device: &R::Device,
        dtype: Option<DType>,
    ) -> Result<Self> {
        Self::from_safetensors_with(path, DEFAULT_LOCAL_ENCODER_PREFIX, cfg, device, dtype)
    }

    /// Load with an explicit checkpoint prefix (e.g. when `feat_encoder`'s
    /// tensors live in the same `model.safetensors` as the rest of
    /// VoxCPM2's stack). `dtype`: cast every loaded tensor to this dtype
    /// (`None` keeps the checkpoint's own).
    pub fn from_safetensors_with<P: AsRef<Path>>(
        path: P,
        prefix: &str,
        cfg: LocalEncoderConfig,
        device: &R::Device,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let mut loader = SafeTensorsLoader::open(path)?;
        let mut tl = TensorLoader::<R> {
            loader: &mut loader,
            device,
            prefix: prefix.to_string(),
            dtype,
        };

        let in_proj = {
            let weight = tl.tensor("in_proj.weight", &[cfg.hidden_dim, cfg.patch_dim])?;
            let bias = tl.tensor("in_proj.bias", &[cfg.hidden_dim])?;
            Linear::new(weight, Some(bias), false)
        };

        let special_token = tl.tensor("special_token", &[1, 1, 1, cfg.hidden_dim])?;

        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(load_layer::<R>(&mut tl, i, &cfg, q_dim, kv_dim)?);
        }

        let norm = RmsNorm::new(
            tl.tensor("encoder.norm.weight", &[cfg.hidden_dim])?,
            cfg.rms_norm_eps,
            false,
        );

        let rope = build_long_rope_cache::<R>(
            cfg.num_positions,
            cfg.head_dim,
            cfg.rope_theta,
            &cfg.rope_short_factor,
            cfg.max_position_embeddings,
            cfg.original_max_position_embeddings,
            device,
        )?;

        Ok(Self {
            in_proj,
            special_token: Var::new(special_token, false),
            layers,
            norm,
            rope,
            hidden_dim: cfg.hidden_dim,
        })
    }
}

/// Load one `feat_encoder.encoder.layers.{i}` transformer layer.
fn load_layer<R: Runtime<DType = DType>>(
    tl: &mut TensorLoader<'_, R>,
    i: usize,
    cfg: &LocalEncoderConfig,
    q_dim: usize,
    kv_dim: usize,
) -> Result<LocalEncoderLayer<R>>
where
    R::Client: TypeConversionOps<R>,
{
    let layer_prefix = format!("encoder.layers.{i}");

    let input_layernorm = RmsNorm::new(
        tl.tensor(
            &format!("{layer_prefix}.input_layernorm.weight"),
            &[cfg.hidden_dim],
        )?,
        cfg.rms_norm_eps,
        false,
    );

    let self_attn = {
        let attn_prefix = format!("{layer_prefix}.self_attn");
        let q_proj = Linear::new(
            tl.tensor(
                &format!("{attn_prefix}.q_proj.weight"),
                &[q_dim, cfg.hidden_dim],
            )?,
            None,
            false,
        );
        let k_proj = Linear::new(
            tl.tensor(
                &format!("{attn_prefix}.k_proj.weight"),
                &[kv_dim, cfg.hidden_dim],
            )?,
            None,
            false,
        );
        let v_proj = Linear::new(
            tl.tensor(
                &format!("{attn_prefix}.v_proj.weight"),
                &[kv_dim, cfg.hidden_dim],
            )?,
            None,
            false,
        );
        let o_proj = Linear::new(
            tl.tensor(
                &format!("{attn_prefix}.o_proj.weight"),
                &[cfg.hidden_dim, q_dim],
            )?,
            None,
            false,
        );
        LocalEncoderAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
        }
    };

    let post_attention_layernorm = RmsNorm::new(
        tl.tensor(
            &format!("{layer_prefix}.post_attention_layernorm.weight"),
            &[cfg.hidden_dim],
        )?,
        cfg.rms_norm_eps,
        false,
    );

    let mlp = {
        let mlp_prefix = format!("{layer_prefix}.mlp");
        let gate_proj = Linear::new(
            tl.tensor(
                &format!("{mlp_prefix}.gate_proj.weight"),
                &[cfg.ffn_dim, cfg.hidden_dim],
            )?,
            None,
            false,
        );
        let up_proj = Linear::new(
            tl.tensor(
                &format!("{mlp_prefix}.up_proj.weight"),
                &[cfg.ffn_dim, cfg.hidden_dim],
            )?,
            None,
            false,
        );
        let down_proj = Linear::new(
            tl.tensor(
                &format!("{mlp_prefix}.down_proj.weight"),
                &[cfg.hidden_dim, cfg.ffn_dim],
            )?,
            None,
            false,
        );
        LocalEncoderMlp {
            gate_proj,
            up_proj,
            down_proj,
        }
    };

    Ok(LocalEncoderLayer {
        input_layernorm,
        self_attn,
        post_attention_layernorm,
        mlp,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn rejects_missing_file() {
        let device = <CpuRuntime as Runtime>::default_device();
        assert!(
            LocalEncoder::<CpuRuntime>::from_safetensors(
                "/nonexistent/model.safetensors",
                LocalEncoderConfig::default(),
                &device,
                None
            )
            .is_err()
        );
    }
}
