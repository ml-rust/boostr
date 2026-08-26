//! Loads `feat_decoder.estimator.*` tensors and assembles a [`LocalDit`].
//!
//! Verified key layout (123 tensors total, `i` = 0..12):
//!
//! ```text
//! feat_decoder.estimator.in_proj.{weight[1024,64],bias[1024]}
//! feat_decoder.estimator.cond_proj.{weight[1024,64],bias[1024]}
//! feat_decoder.estimator.out_proj.{weight[64,1024],bias[64]}
//! feat_decoder.estimator.time_mlp.linear_1.{weight[1024,1024],bias[1024]}
//! feat_decoder.estimator.time_mlp.linear_2.{weight[1024,1024],bias[1024]}
//! feat_decoder.estimator.delta_time_mlp.linear_1.{weight[1024,1024],bias[1024]}
//! feat_decoder.estimator.delta_time_mlp.linear_2.{weight[1024,1024],bias[1024]}
//! feat_decoder.estimator.decoder.norm.weight                                [1024]
//! feat_decoder.estimator.decoder.layers.{i}.input_layernorm.weight          [1024]
//! feat_decoder.estimator.decoder.layers.{i}.self_attn.q_proj.weight         [2048,1024]
//! feat_decoder.estimator.decoder.layers.{i}.self_attn.k_proj.weight         [256,1024]
//! feat_decoder.estimator.decoder.layers.{i}.self_attn.v_proj.weight         [256,1024]
//! feat_decoder.estimator.decoder.layers.{i}.self_attn.o_proj.weight         [1024,2048]
//! feat_decoder.estimator.decoder.layers.{i}.post_attention_layernorm.weight [1024]
//! feat_decoder.estimator.decoder.layers.{i}.mlp.gate_proj.weight            [4096,1024]
//! feat_decoder.estimator.decoder.layers.{i}.mlp.up_proj.weight              [4096,1024]
//! feat_decoder.estimator.decoder.layers.{i}.mlp.down_proj.weight            [1024,4096]
//! ```
//! 15 + 12*9 = 123. `in_proj`/`cond_proj`/`out_proj` and both `TimestepEmbedding`
//! MLPs (`time_mlp`, `delta_time_mlp`) are biased; every projection/norm inside
//! the 12-layer block stack is bias-free — same split as `feat_encoder`.
//!
//! Does not load a `SinusoidalPosEmb` from the checkpoint (it carries no
//! weights — see `crate::nn::timestep_embedding::SinusoidalPosEmb`), but
//! still builds one once here, alongside the RoPE cache, so the estimator
//! forward pass never reconstructs it per call.

use crate::error::Result;
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::model::audio::voxcpm::bidirectional::layer::BidirectionalLayer;
use crate::model::audio::voxcpm::bidirectional::{
    BidirectionalLayerDims, load_bidirectional_layer,
};
use crate::model::audio::voxcpm::loader::support::TensorLoader;
use crate::model::audio::voxcpm::local_dit::config::LocalDitConfig;
use crate::model::config::RopeScalingConfig;
use crate::nn::{Linear, RmsNorm, RoPE, SinusoidalPosEmb, TimestepEmbedding};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use std::path::Path;

/// Default top-level prefix for `feat_decoder`'s tensors in the VoxCPM2
/// checkpoint.
pub const DEFAULT_LOCAL_DIT_PREFIX: &str = "feat_decoder";

/// VoxCPM2's `feat_decoder` local DiT ("locdit"): loaded weights only. The
/// estimator forward pass (assembling `[mu, t, cond, x]` and running
/// `decoder`) is a separate unit that reads these fields directly (they are
/// `pub(crate)`); the accessors below are this type's public API for
/// consumers outside this crate, which cannot reach `pub(crate)` fields.
pub struct LocalDit<R: Runtime> {
    pub(crate) in_proj: Linear<R>,
    pub(crate) cond_proj: Linear<R>,
    pub(crate) out_proj: Linear<R>,
    pub(crate) time_mlp: TimestepEmbedding<R>,
    pub(crate) delta_time_mlp: TimestepEmbedding<R>,
    pub(crate) layers: Vec<BidirectionalLayer<R>>,
    pub(crate) norm: RmsNorm<R>,
    pub(crate) rope: RoPE<R>,
    pub(crate) time_embeddings: SinusoidalPosEmb<R>,
    pub(crate) hidden_dim: usize,
    pub(crate) feat_dim: usize,
    pub(crate) patch_size: usize,
}

impl<R: Runtime<DType = DType>> LocalDit<R> {
    pub fn in_proj(&self) -> &Linear<R> {
        &self.in_proj
    }

    pub fn cond_proj(&self) -> &Linear<R> {
        &self.cond_proj
    }

    pub fn out_proj(&self) -> &Linear<R> {
        &self.out_proj
    }

    pub fn time_mlp(&self) -> &TimestepEmbedding<R> {
        &self.time_mlp
    }

    pub fn delta_time_mlp(&self) -> &TimestepEmbedding<R> {
        &self.delta_time_mlp
    }

    pub fn layers(&self) -> &[BidirectionalLayer<R>] {
        &self.layers
    }

    pub fn norm(&self) -> &RmsNorm<R> {
        &self.norm
    }

    pub fn rope(&self) -> &RoPE<R> {
        &self.rope
    }

    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    pub fn feat_dim(&self) -> usize {
        self.feat_dim
    }

    pub fn patch_size(&self) -> usize {
        self.patch_size
    }
}

impl<R: Runtime<DType = DType>> LocalDit<R>
where
    R::Client: TypeConversionOps<R>,
{
    /// Load `feat_decoder` from a VoxCPM2 checkpoint using `cfg`'s
    /// architecture/RoPE parameters (see [`LocalDitConfig::from_config_json`]
    /// for reading those out of the checkpoint's `config.json`). `path` may
    /// be the `model.safetensors` file or its containing directory. `dtype`:
    /// cast every loaded tensor to this dtype (`None` keeps the checkpoint's
    /// own).
    pub fn from_safetensors<P: AsRef<Path>>(
        path: P,
        cfg: LocalDitConfig,
        device: &R::Device,
        dtype: Option<DType>,
    ) -> Result<Self> {
        Self::from_safetensors_with(path, DEFAULT_LOCAL_DIT_PREFIX, cfg, device, dtype)
    }

    /// Load with an explicit checkpoint prefix (e.g. when `feat_decoder`'s
    /// tensors live in the same `model.safetensors` as the rest of VoxCPM2's
    /// stack). `dtype`: cast every loaded tensor to this dtype (`None` keeps
    /// the checkpoint's own).
    pub fn from_safetensors_with<P: AsRef<Path>>(
        path: P,
        prefix: &str,
        cfg: LocalDitConfig,
        device: &R::Device,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let mut loader = SafeTensorsLoader::open(path)?;
        let mut tl = TensorLoader::<R> {
            loader: &mut loader,
            device,
            prefix: format!("{prefix}.estimator"),
            dtype,
        };

        let in_proj = {
            let weight = tl.tensor("in_proj.weight", &[cfg.hidden_dim, cfg.feat_dim])?;
            let bias = tl.tensor("in_proj.bias", &[cfg.hidden_dim])?;
            Linear::new(weight, Some(bias), false)
        };
        let cond_proj = {
            let weight = tl.tensor("cond_proj.weight", &[cfg.hidden_dim, cfg.feat_dim])?;
            let bias = tl.tensor("cond_proj.bias", &[cfg.hidden_dim])?;
            Linear::new(weight, Some(bias), false)
        };
        let out_proj = {
            let weight = tl.tensor("out_proj.weight", &[cfg.feat_dim, cfg.hidden_dim])?;
            let bias = tl.tensor("out_proj.bias", &[cfg.feat_dim])?;
            Linear::new(weight, Some(bias), false)
        };

        let time_mlp = load_timestep_mlp(&mut tl, "time_mlp", cfg.hidden_dim)?;
        let delta_time_mlp = load_timestep_mlp(&mut tl, "delta_time_mlp", cfg.hidden_dim)?;

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
            layers.push(load_bidirectional_layer::<R>(
                &mut tl,
                &format!("decoder.layers.{i}"),
                &dims,
            )?);
        }

        let norm = RmsNorm::new(
            tl.tensor("decoder.norm.weight", &[cfg.hidden_dim])?,
            cfg.rms_norm_eps,
            false,
        );

        // Cache is built at `max_position_embeddings` (selects the longrope
        // scaling regime, matching every other model in this codebase), then
        // narrowed to the DiT's actual assembled-sequence length — see
        // `local_encoder/loader.rs`'s identical reasoning.
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
        // `sequence_len()` derives `2 (mu) + 1 (t) + patch_size (cond) +
        // patch_size (x)` — never hardcoded 11.
        let rope = rope.narrow_positions(cfg.sequence_len())?;

        // No learned weights (see the module doc); built once here, like
        // `rope` above, so `forward` never re-uploads its frequency table.
        let time_embeddings = SinusoidalPosEmb::<R>::new(cfg.hidden_dim, device)?;

        Ok(Self {
            in_proj,
            cond_proj,
            out_proj,
            time_mlp,
            delta_time_mlp,
            layers,
            norm,
            rope,
            time_embeddings,
            hidden_dim: cfg.hidden_dim,
            feat_dim: cfg.feat_dim,
            patch_size: cfg.patch_size,
        })
    }
}

/// Load a `{name}.linear_1`/`{name}.linear_2` [`TimestepEmbedding`] (both
/// linears biased, `dim -> dim -> dim`) — shared shape for `time_mlp` and
/// `delta_time_mlp`.
fn load_timestep_mlp<R: Runtime<DType = DType>>(
    tl: &mut TensorLoader<'_, R>,
    name: &str,
    dim: usize,
) -> Result<TimestepEmbedding<R>>
where
    R::Client: TypeConversionOps<R>,
{
    let linear_1 = {
        let weight = tl.tensor(&format!("{name}.linear_1.weight"), &[dim, dim])?;
        let bias = tl.tensor(&format!("{name}.linear_1.bias"), &[dim])?;
        Linear::new(weight, Some(bias), false)
    };
    let linear_2 = {
        let weight = tl.tensor(&format!("{name}.linear_2.weight"), &[dim, dim])?;
        let bias = tl.tensor(&format!("{name}.linear_2.bias"), &[dim])?;
        Linear::new(weight, Some(bias), false)
    };
    Ok(TimestepEmbedding::new(linear_1, linear_2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn rejects_missing_file() {
        let device = <CpuRuntime as Runtime>::default_device();
        assert!(
            LocalDit::<CpuRuntime>::from_safetensors(
                "/nonexistent/model.safetensors",
                LocalDitConfig::default(),
                &device,
                None
            )
            .is_err()
        );
    }
}
