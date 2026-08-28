//! Loads `fsq_layer.*` and the six root-level auxiliary-projection tensors
//! into [`ScalarQuantization`]/[`AuxProjections`].
//!
//! Verified key layout (15 tensors total, no shared prefix):
//!
//! ```text
//! fsq_layer.in_proj.{weight[512,2048],bias[512]}
//! fsq_layer.out_proj.{weight[2048,512],bias[2048]}
//! enc_to_lm_proj.{weight[2048,1024],bias[2048]}
//! lm_to_dit_proj.{weight[1024,2048],bias[1024]}
//! res_to_dit_proj.{weight[1024,2048],bias[1024]}
//! fusion_concat_proj.{weight[2048,4096],bias[2048]}
//! stop_proj.{weight[2048,2048],bias[2048]}
//! stop_head.weight[2,2048]                          (NO bias)
//! ```
//!
//! Every projection above is biased except `stop_head`, which the checkpoint
//! carries no bias tensor for — loading it as biased would either fail (no
//! such key) or, if a zero-filled default were substituted, silently add
//! zeros the reference never does. `stop_head` is loaded as weight-only.

use crate::error::Result;
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::model::audio::voxcpm::fsq::config::FsqConfig;
use crate::model::audio::voxcpm::fsq::layer::{AuxProjections, ScalarQuantization};
use crate::model::audio::voxcpm::loader::support::{TensorLoader, WeightSource};
use crate::nn::MaybeQuantLinear;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use std::path::Path;

/// Checkpoint prefix for `fsq_layer`'s own tensors. The other six
/// projections live at the checkpoint root (empty prefix).
///
/// `pub(crate)` so [`VoxCpm2Model`](crate::model::audio::voxcpm::model::VoxCpm2Model)'s
/// `Module::named_parameters` reuses this exact constant instead of a
/// duplicated `"fsq_layer"` literal that could drift from the loader.
pub(crate) const FSQ_LAYER_PREFIX: &str = "fsq_layer";

impl<R: Runtime<DType = DType>> ScalarQuantization<R>
where
    R::Client: TypeConversionOps<R>,
{
    /// Load `fsq_layer.in_proj`/`fsq_layer.out_proj` from a VoxCPM2
    /// checkpoint. `path` may be the `model.safetensors` file or its
    /// containing directory. `dtype`: cast every loaded tensor to this dtype
    /// (`None` keeps the checkpoint's own).
    pub fn from_safetensors<P: AsRef<Path>>(
        path: P,
        cfg: FsqConfig,
        device: &R::Device,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let mut source = SafeTensorsLoader::open(path)?;
        Self::from_source(&mut source, cfg, device, dtype)
    }

    /// Load from an ALREADY-OPEN checkpoint (safetensors or GGUF — see
    /// [`WeightSource`]), so the VoxCPM2 orchestrator opens its one
    /// multi-gigabyte weight file once for all seven sub-models instead of
    /// reopening and re-parsing its header per sub-model.
    pub fn from_source<S: WeightSource<R>>(
        source: &mut S,
        cfg: FsqConfig,
        device: &R::Device,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let mut tl = TensorLoader::<R, S> {
            loader: source,
            device,
            prefix: FSQ_LAYER_PREFIX.to_string(),
            dtype,
        };

        // `TensorLoader::linear` keeps a block-quantized weight PACKED, so on
        // a GGUF these two multiply through `quant_matmul`; on safetensors
        // they are the same dense `Linear` they always were. Both are biased,
        // and a packed weight forces that bias to F32 (see
        // `TensorLoader::linear`).
        let in_proj = tl.linear("in_proj", cfg.latent_dim, cfg.lm_hidden, true)?;
        let out_proj = tl.linear("out_proj", cfg.lm_hidden, cfg.latent_dim, true)?;

        Ok(Self::new(in_proj, out_proj, cfg.scale))
    }
}

impl<R: Runtime<DType = DType>> AuxProjections<R>
where
    R::Client: TypeConversionOps<R>,
{
    /// Load the six root-level auxiliary projections from a VoxCPM2
    /// checkpoint. `path` may be the `model.safetensors` file or its
    /// containing directory. `dtype`: cast every loaded tensor to this dtype
    /// (`None` keeps the checkpoint's own).
    pub fn from_safetensors<P: AsRef<Path>>(
        path: P,
        cfg: FsqConfig,
        device: &R::Device,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let mut source = SafeTensorsLoader::open(path)?;
        Self::from_source(&mut source, cfg, device, dtype)
    }

    /// Load from an ALREADY-OPEN checkpoint (safetensors or GGUF — see
    /// [`WeightSource`]), so the VoxCPM2 orchestrator opens its one
    /// multi-gigabyte weight file once for all seven sub-models instead of
    /// reopening and re-parsing its header per sub-model.
    pub fn from_source<S: WeightSource<R>>(
        source: &mut S,
        cfg: FsqConfig,
        device: &R::Device,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let mut tl = TensorLoader::<R, S> {
            loader: source,
            device,
            prefix: String::new(),
            dtype,
        };

        let enc_to_lm_proj =
            biased_linear(&mut tl, "enc_to_lm_proj", cfg.dit_hidden, cfg.lm_hidden)?;
        let lm_to_dit_proj =
            biased_linear(&mut tl, "lm_to_dit_proj", cfg.lm_hidden, cfg.dit_hidden)?;
        let res_to_dit_proj =
            biased_linear(&mut tl, "res_to_dit_proj", cfg.lm_hidden, cfg.dit_hidden)?;
        let fusion_concat_proj = biased_linear(
            &mut tl,
            "fusion_concat_proj",
            2 * cfg.lm_hidden,
            cfg.lm_hidden,
        )?;
        let stop_proj = biased_linear(&mut tl, "stop_proj", cfg.lm_hidden, cfg.lm_hidden)?;

        // `stop_head` is bias-free on this checkpoint — see the module doc
        // for why a bias key is never read here.
        const STOP_CLASSES: usize = 2;
        let stop_head = tl.linear("stop_head", STOP_CLASSES, cfg.lm_hidden, false)?;

        Ok(Self {
            enc_to_lm_proj,
            lm_to_dit_proj,
            res_to_dit_proj,
            fusion_concat_proj,
            stop_proj,
            stop_head,
        })
    }
}

/// Load `{name}.weight[out_dim, in_dim]` + `{name}.bias[out_dim]` as a
/// biased [`MaybeQuantLinear`]. Shared shape for five of the six auxiliary
/// projections (`stop_head` is the bias-free exception, handled inline in
/// [`AuxProjections::from_safetensors`]).
///
/// `TensorLoader::linear` keeps a block-quantized weight PACKED, so on a GGUF
/// each of these multiplies through `quant_matmul`; on safetensors they are
/// the same dense `Linear` they always were.
///
/// This exists only to flip `TensorLoader::linear`'s argument order: the
/// callers below read `(in_dim, out_dim)`, matching how the reference names
/// each bridge (`enc_to_lm`, `lm_to_dit`, ...), while `TensorLoader::linear`
/// takes `(out_features, in_features)` to match the `[out, in]` weight shape.
fn biased_linear<R: Runtime<DType = DType>, S: WeightSource<R>>(
    tl: &mut TensorLoader<'_, R, S>,
    name: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<MaybeQuantLinear<R>>
where
    R::Client: TypeConversionOps<R>,
{
    tl.linear(name, out_dim, in_dim, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn scalar_quantization_rejects_missing_file() {
        let device = <CpuRuntime as Runtime>::default_device();
        assert!(
            ScalarQuantization::<CpuRuntime>::from_safetensors(
                "/nonexistent/model.safetensors",
                FsqConfig::default(),
                &device,
                None
            )
            .is_err()
        );
    }

    #[test]
    fn aux_projections_reject_missing_file() {
        let device = <CpuRuntime as Runtime>::default_device();
        assert!(
            AuxProjections::<CpuRuntime>::from_safetensors(
                "/nonexistent/model.safetensors",
                FsqConfig::default(),
                &device,
                None
            )
            .is_err()
        );
    }
}
