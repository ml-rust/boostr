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
use crate::model::audio::voxcpm::bidirectional::{
    BidirectionalLayerDims, load_bidirectional_layer,
};
use crate::model::audio::voxcpm::loader::support::{TensorLoader, WeightSource};
use crate::model::audio::voxcpm::local_encoder::config::LocalEncoderConfig;
use crate::model::audio::voxcpm::local_encoder::encoder::LocalEncoder;
use crate::model::config::RopeScalingConfig;
use crate::nn::{Linear, RmsNorm, RoPE};
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
    /// checkpoint's own) — the cast happens once, in
    /// `TensorLoader::tensor`.
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
        let mut source = SafeTensorsLoader::open(path)?;
        Self::from_source(&mut source, prefix, cfg, device, dtype)
    }

    /// Load from an ALREADY-OPEN checkpoint (safetensors or GGUF — see
    /// [`WeightSource`]), so the VoxCPM2 orchestrator opens its one
    /// multi-gigabyte weight file once for all seven sub-models instead of
    /// reopening and re-parsing its header per sub-model.
    pub fn from_source<S: WeightSource<R>>(
        source: &mut S,
        prefix: &str,
        cfg: LocalEncoderConfig,
        device: &R::Device,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let mut tl = TensorLoader::<R, S> {
            loader: source,
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

        let dims = BidirectionalLayerDims {
            hidden_dim: cfg.hidden_dim,
            ffn_dim: cfg.ffn_dim,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
        };
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(load_bidirectional_layer::<R, S>(
                &mut tl,
                &format!("encoder.layers.{i}"),
                &dims,
            )?);
        }

        let norm = RmsNorm::new(
            tl.tensor("encoder.norm.weight", &[cfg.hidden_dim])?,
            cfg.rms_norm_eps,
            false,
        );

        // Cache is built at `max_position_embeddings` (not `num_positions`), matching
        // every other model in this codebase (e.g. `model/llama/model/forward.rs`):
        // `RoPE::forward`/`apply_rope` narrow `[max_seq_len, D/2]` caches down to the
        // actual sequence length at call time (`ops/impl_generic/attention/rope/common.rs`).
        // Building at `num_positions` (5) instead would feed 5 as the "max_position_embeddings"
        // role in the longrope `attention_scaling` formula, which is wrong for
        // this checkpoint (max_position_embeddings == original_max_position_embeddings
        // == 32768) and would silently produce a different scale than the
        // checkpoint's real config implies.
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
        // The cache above is built long (32768 rows) only so `max_seq_len` picks
        // the right longrope scaling regime; `feat_encoder` only ever rotates
        // `cfg.num_positions` (5) positions per (batch, frame), so the rest is
        // dead memory once the scaling has been baked in.
        let rope = rope.narrow_positions(cfg.num_positions)?;

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
