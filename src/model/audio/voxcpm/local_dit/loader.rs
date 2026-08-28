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
use crate::model::audio::voxcpm::loader::support::{TensorLoader, WeightSource};
use crate::model::audio::voxcpm::local_dit::config::LocalDitConfig;
use crate::model::config::RopeScalingConfig;
use crate::nn::{
    MaybeQuantLinear, Module, RmsNorm, RoPE, SinusoidalPosEmb, TimestepEmbedding, child_params,
    extend_named,
};
use numr::autograd::Var;
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
///
/// The three projections are [`MaybeQuantLinear`], not plain `Linear` (as are
/// the two [`TimestepEmbedding`] MLPs and every projection in the block
/// stack): a GGUF stores them block-quantized, and the quantized variant
/// multiplies the weight PACKED through `quant_matmul` instead of expanding
/// it to dense F32 at load. This is the hottest stack in the model — 32 CFM
/// timesteps x 2 CFG branches per generated patch — so it is also where the
/// integer activation dot product pays the most. A safetensors checkpoint
/// yields the `Standard` variant and runs exactly the dense path it always
/// did.
pub struct LocalDit<R: Runtime> {
    pub(crate) in_proj: MaybeQuantLinear<R>,
    pub(crate) cond_proj: MaybeQuantLinear<R>,
    pub(crate) out_proj: MaybeQuantLinear<R>,
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
    pub fn in_proj(&self) -> &MaybeQuantLinear<R> {
        &self.in_proj
    }

    pub fn cond_proj(&self) -> &MaybeQuantLinear<R> {
        &self.cond_proj
    }

    pub fn out_proj(&self) -> &MaybeQuantLinear<R> {
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
        cfg: LocalDitConfig,
        device: &R::Device,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let mut tl = TensorLoader::<R, S> {
            loader: source,
            device,
            prefix: format!("{prefix}.estimator"),
            dtype,
        };

        // `TensorLoader::linear` keeps a block-quantized weight PACKED, so on
        // a GGUF these three multiply through `quant_matmul`; on safetensors
        // they are the same dense `Linear` they always were. All three are
        // biased, and a packed weight forces that bias to F32 (see
        // `TensorLoader::linear`).
        let in_proj = tl.linear("in_proj", cfg.hidden_dim, cfg.feat_dim, true)?;
        let cond_proj = tl.linear("cond_proj", cfg.hidden_dim, cfg.feat_dim, true)?;
        let out_proj = tl.linear("out_proj", cfg.feat_dim, cfg.hidden_dim, true)?;

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
            layers.push(load_bidirectional_layer::<R, S>(
                &mut tl,
                &format!("decoder.layers.{i}"),
                &dims,
            )?);
        }

        // DENSE, deliberately: an RmsNorm weight is an element-wise scale,
        // not a matmul weight, and a GGUF stores it unquantized anyway.
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
fn load_timestep_mlp<R: Runtime<DType = DType>, S: WeightSource<R>>(
    tl: &mut TensorLoader<'_, R, S>,
    name: &str,
    dim: usize,
) -> Result<TimestepEmbedding<R>>
where
    R::Client: TypeConversionOps<R>,
{
    let linear_1 = tl.linear(&format!("{name}.linear_1"), dim, dim, true)?;
    let linear_2 = tl.linear(&format!("{name}.linear_2"), dim, dim, true)?;
    Ok(TimestepEmbedding::new(linear_1, linear_2))
}

/// Names mirror `feat_decoder.estimator.*` (checkpoint prefix `feat_decoder`
/// added by [`VoxCpm2Model`](crate::model::audio::voxcpm::model::VoxCpm2Model)).
/// `estimator.decoder.layers.{i}`/`estimator.decoder.norm` hardcode a
/// `decoder.` segment this struct's own field names (`layers`, `norm`) do
/// not carry — the checkpoint nests the transformer block stack under
/// `feat_decoder.estimator.decoder.*` (see the module doc's key layout).
/// `rope` and `time_embeddings` carry no `Var<R>` (`time_embeddings`'s
/// frequency table is a fixed, non-learned constant — see
/// [`SinusoidalPosEmb`]) and are correctly absent from every collection
/// below.
impl<R: Runtime> Module<R> for LocalDit<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        let mut params = child_params(&self.in_proj);
        params.extend(child_params(&self.cond_proj));
        params.extend(child_params(&self.out_proj));
        params.extend(child_params(&self.time_mlp));
        params.extend(child_params(&self.delta_time_mlp));
        for layer in &self.layers {
            params.extend(child_params(layer));
        }
        params.extend(child_params(&self.norm));
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        extend_named(
            &mut params,
            "estimator.in_proj",
            self.in_proj.named_parameters(),
        );
        extend_named(
            &mut params,
            "estimator.cond_proj",
            self.cond_proj.named_parameters(),
        );
        extend_named(
            &mut params,
            "estimator.out_proj",
            self.out_proj.named_parameters(),
        );
        extend_named(
            &mut params,
            "estimator.time_mlp",
            self.time_mlp.named_parameters(),
        );
        extend_named(
            &mut params,
            "estimator.delta_time_mlp",
            self.delta_time_mlp.named_parameters(),
        );
        for (i, layer) in self.layers.iter().enumerate() {
            extend_named(
                &mut params,
                &format!("estimator.decoder.layers.{i}"),
                layer.named_parameters(),
            );
        }
        extend_named(
            &mut params,
            "estimator.decoder.norm",
            self.norm.named_parameters(),
        );
        params
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
