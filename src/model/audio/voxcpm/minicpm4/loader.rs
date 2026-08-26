//! Loads `base_lm.*` tensors and assembles a [`MiniCpm4Model`].
//!
//! Verified key layout (254 tensors: `1 + 28*9 + 1`). Layer indices are
//! NUMERIC `0..num_layers` — this loader requests every tensor by explicit
//! numeric index, so the safetensors header's string-sorted order
//! (0,1,10,11,2,...) never matters.
//!
//! ```text
//! base_lm.embed_tokens.weight                          [73448, 2048]
//! base_lm.layers.{i}.input_layernorm.weight            [2048]
//! base_lm.layers.{i}.self_attn.q_proj.weight           [2048, 2048]
//! base_lm.layers.{i}.self_attn.k_proj.weight           [256, 2048]
//! base_lm.layers.{i}.self_attn.v_proj.weight           [256, 2048]
//! base_lm.layers.{i}.self_attn.o_proj.weight           [2048, 2048]
//! base_lm.layers.{i}.post_attention_layernorm.weight   [2048]
//! base_lm.layers.{i}.mlp.gate_proj.weight              [6144, 2048]
//! base_lm.layers.{i}.mlp.up_proj.weight                [6144, 2048]
//! base_lm.layers.{i}.mlp.down_proj.weight              [2048, 6144]
//! base_lm.norm.weight                                  [2048]
//! ```
//!
//! Everything is bias-free — every attention projection, every MLP
//! projection, and both norms. `embed_tokens` is skipped entirely when
//! `cfg.vocab_size == 0`, which is how VoxCPM2's `residual_lm` (same
//! architecture, 8 layers, no embedding table) loads through this same code.

use crate::error::Result;
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::model::audio::voxcpm::loader::support::TensorLoader;
use crate::model::audio::voxcpm::minicpm4::attention::MiniCpm4Attention;
use crate::model::audio::voxcpm::minicpm4::config::MiniCpm4Config;
use crate::model::audio::voxcpm::minicpm4::layer::MiniCpm4Layer;
use crate::model::audio::voxcpm::minicpm4::mlp::MiniCpm4Mlp;
use crate::model::audio::voxcpm::minicpm4::model::MiniCpm4Model;
use crate::model::config::RopeScalingConfig;
use crate::nn::{Embedding, Linear, RmsNorm, RoPE};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use std::path::Path;

/// Default top-level prefix for the MiniCPM4 decoder's tensors in the VoxCPM2
/// checkpoint.
pub const DEFAULT_MINICPM4_PREFIX: &str = "base_lm";

impl<R: Runtime<DType = DType>> MiniCpm4Model<R>
where
    R::Client: TypeConversionOps<R>,
{
    /// Load `base_lm` from a VoxCPM2 checkpoint using `cfg`'s
    /// architecture/RoPE parameters (see [`MiniCpm4Config::from_config_json`]
    /// for reading those out of the checkpoint's `config.json`). `path` may
    /// be the `model.safetensors` file or its containing directory.
    /// `dtype`: cast every loaded tensor to this dtype (`None` keeps the
    /// checkpoint's own) — see
    /// [`checked_tensor`](crate::model::audio::voxcpm::loader::support::checked_tensor).
    pub fn from_safetensors<P: AsRef<Path>>(
        path: P,
        cfg: MiniCpm4Config,
        device: &R::Device,
        dtype: Option<DType>,
    ) -> Result<Self> {
        Self::from_safetensors_with(path, DEFAULT_MINICPM4_PREFIX, cfg, device, dtype)
    }

    /// Load with an explicit checkpoint prefix. `residual_lm` is this same
    /// architecture under a different prefix and a different `cfg`, so the
    /// prefix is a caller argument rather than a constant baked into the
    /// walk. A trailing `.` on `prefix` is absorbed. `dtype`: cast every
    /// loaded tensor to this dtype (`None` keeps the checkpoint's own).
    pub fn from_safetensors_with<P: AsRef<Path>>(
        path: P,
        prefix: &str,
        cfg: MiniCpm4Config,
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

        // `vocab_size == 0` means the checkpoint has no `embed_tokens` tensor
        // at all; requesting it would fail the load rather than yield `None`.
        let embed_tokens = if cfg.has_embedding() {
            Some(Embedding::new(
                tl.tensor("embed_tokens.weight", &[cfg.vocab_size, cfg.hidden_size])?,
                false,
            ))
        } else {
            None
        };

        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(load_layer::<R>(&mut tl, i, &cfg, q_dim, kv_dim)?);
        }

        let norm = RmsNorm::new(
            tl.tensor("norm.weight", &[cfg.hidden_size])?,
            cfg.rms_norm_eps,
            false,
        );

        // LongRoPE: a per-dimension divisor list plus an `attention_scaling`
        // (mscale) folded into the caches, all owned by
        // `RoPE::precompute_freqs`. Nothing is hand-rolled here, and nothing
        // special-cases this checkpoint: because `max_position_embeddings ==
        // original_max_position_embeddings` (32768), the shared code selects
        // `short_factor` and computes `attention_scaling == 1.0` on its own.
        //
        // The cache is built at `max_position_embeddings`, matching every
        // other model in this codebase: `apply_rope` narrows the
        // `[max_seq_len, head_dim/2]` caches down to the actual sequence
        // length at call time. Building it shorter would feed that shorter
        // length into the LongRoPE regime choice and silently change the
        // scale.
        let rope_scaling = RopeScalingConfig {
            scaling_type: "longrope".to_string(),
            factor: 1.0,
            original_max_position_embeddings: Some(cfg.original_max_position_embeddings),
            low_freq_factor: None,
            high_freq_factor: None,
            attention_factor: None,
            beta_fast: None,
            beta_slow: None,
            short_factor: Some(cfg.rope_short_factor.clone()),
            long_factor: Some(cfg.rope_long_factor.clone()),
        };
        let rope = RoPE::<R>::precompute_freqs(
            cfg.max_position_embeddings,
            cfg.head_dim,
            cfg.rope_theta,
            Some(&rope_scaling),
            device,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rope,
            hidden_size: cfg.hidden_size,
        })
    }
}

/// Load one `{prefix}.layers.{i}` decoder layer.
fn load_layer<R: Runtime<DType = DType>>(
    tl: &mut TensorLoader<'_, R>,
    i: usize,
    cfg: &MiniCpm4Config,
    q_dim: usize,
    kv_dim: usize,
) -> Result<MiniCpm4Layer<R>>
where
    R::Client: TypeConversionOps<R>,
{
    let layer_prefix = format!("layers.{i}");

    let input_layernorm = RmsNorm::new(
        tl.tensor(
            &format!("{layer_prefix}.input_layernorm.weight"),
            &[cfg.hidden_size],
        )?,
        cfg.rms_norm_eps,
        false,
    );

    let self_attn = {
        let attn_prefix = format!("{layer_prefix}.self_attn");
        let q_proj = Linear::new(
            tl.tensor(
                &format!("{attn_prefix}.q_proj.weight"),
                &[q_dim, cfg.hidden_size],
            )?,
            None,
            false,
        );
        let k_proj = Linear::new(
            tl.tensor(
                &format!("{attn_prefix}.k_proj.weight"),
                &[kv_dim, cfg.hidden_size],
            )?,
            None,
            false,
        );
        let v_proj = Linear::new(
            tl.tensor(
                &format!("{attn_prefix}.v_proj.weight"),
                &[kv_dim, cfg.hidden_size],
            )?,
            None,
            false,
        );
        let o_proj = Linear::new(
            tl.tensor(
                &format!("{attn_prefix}.o_proj.weight"),
                &[cfg.hidden_size, q_dim],
            )?,
            None,
            false,
        );
        MiniCpm4Attention {
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
            &[cfg.hidden_size],
        )?,
        cfg.rms_norm_eps,
        false,
    );

    let mlp = {
        let mlp_prefix = format!("{layer_prefix}.mlp");
        let gate_proj = Linear::new(
            tl.tensor(
                &format!("{mlp_prefix}.gate_proj.weight"),
                &[cfg.intermediate_size, cfg.hidden_size],
            )?,
            None,
            false,
        );
        let up_proj = Linear::new(
            tl.tensor(
                &format!("{mlp_prefix}.up_proj.weight"),
                &[cfg.intermediate_size, cfg.hidden_size],
            )?,
            None,
            false,
        );
        let down_proj = Linear::new(
            tl.tensor(
                &format!("{mlp_prefix}.down_proj.weight"),
                &[cfg.hidden_size, cfg.intermediate_size],
            )?,
            None,
            false,
        );
        MiniCpm4Mlp {
            gate_proj,
            up_proj,
            down_proj,
        }
    };

    Ok(MiniCpm4Layer {
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
            MiniCpm4Model::<CpuRuntime>::from_safetensors(
                "/nonexistent/model.safetensors",
                MiniCpm4Config::default(),
                &device,
                None
            )
            .is_err()
        );
    }
}
