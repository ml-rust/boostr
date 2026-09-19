//! Construction of a [`LoadedModel`] from config and weights.

use super::model::LoadedModel;
use crate::error::{Error, Result};
use crate::model::config::UniversalConfig;
use crate::model::traits::Model;
use crate::nn::VarBuilder;
use numr::dtype::DType;
use numr::ops::{IndexingOps, ReduceOps, ShapeOps};
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> LoadedModel<R>
where
    R::Client: IndexingOps<R>
        + crate::quant::DequantOps<R>
        + numr::ops::TypeConversionOps<R>
        + ReduceOps<R>
        + ShapeOps<R>,
{
    /// Load a model from universal config and weights.
    ///
    /// Uses capability-based dispatch: any model with an attention config
    /// is loaded as a Llama (universal transformer). This means new HF
    /// model types work automatically without code changes as long as they
    /// share the standard transformer structure.
    pub fn load(config: &UniversalConfig, vb: &mut VarBuilder<R>) -> Result<Self> {
        // GGUF weights carry their tensors' `GgmlType`s into `vb`'s `VarMap`
        // at load time (`VarMap::from_gguf`); a SafeTensors-backed `vb`
        // reports none. Fold the distinct formats into the config so every
        // model variant's `quant_formats()` sees them through `m.config()`,
        // with no per-arch `from_varbuilder` change needed.
        let formats = vb.quant_formats();
        let owned_config = if formats.is_empty() {
            None
        } else {
            let mut c = config.clone();
            c.quant_formats = formats.to_vec();
            Some(c)
        };
        let config = owned_config.as_ref().unwrap_or(config);
        match config.model_type.as_str() {
            "mamba1" => {
                let model = crate::model::mamba::Mamba1Model::from_varbuilder(vb, config)?;
                Ok(LoadedModel::Mamba1(Box::new(model)))
            }
            "mamba2" => {
                let model = crate::model::mamba::Mamba2Model::from_varbuilder(vb, config)?;
                Ok(LoadedModel::Mamba2(Box::new(model)))
            }
            "mamba3" => {
                let model = crate::model::mamba::Mamba3Model::from_varbuilder(vb, config)?;
                Ok(LoadedModel::Mamba3(Box::new(model)))
            }
            "hybrid" => {
                let model = crate::model::hybrid::HybridModel::from_varbuilder(vb, config)?;
                Ok(LoadedModel::Hybrid(Box::new(model)))
            }
            "qwen35" => {
                let model = crate::model::qwen35::Qwen35Model::from_varbuilder(vb, config)?;
                Ok(LoadedModel::Qwen35(Box::new(model)))
            }
            // Multimodal: vision/audio encoders + LLM backbone
            _ if config.vision.is_some() || config.audio.is_some() => {
                let model = crate::model::multimodal::MultimodalModel::from_varbuilder(vb, config)?;
                Ok(LoadedModel::Multimodal(Box::new(model)))
            }
            // Everything else with attention config → Llama (the universal transformer)
            _ if config.attention.is_some() => {
                let model = crate::model::llama::Llama::from_varbuilder(vb, config)?;
                Ok(LoadedModel::Llama(Box::new(model)))
            }
            other => Err(Error::ModelError {
                reason: format!(
                    "Unknown model type '{other}' without attention config. \
                     Only pure SSM models (mamba1/mamba2/mamba3) and hybrid models are \
                     supported without attention configuration."
                ),
            }),
        }
    }

    /// Load a tensor-parallel model. Requires a NCCL communicator.
    ///
    /// Tensor parallelism is supported for any model with an attention config
    /// (i.e., transformer architectures loaded via the Llama struct).
    pub fn load_tp(
        config: &UniversalConfig,
        vb: &mut VarBuilder<R>,
        comm: std::sync::Arc<dyn numr::runtime::Communicator>,
    ) -> Result<Self> {
        if config.attention.is_some() {
            let model = crate::model::llama::LlamaTp::from_varbuilder(vb, config, comm)?;
            Ok(LoadedModel::LlamaTp(Box::new(model)))
        } else {
            Err(Error::ModelError {
                reason: format!(
                    "Tensor parallelism not supported for model type '{}' \
                     (requires attention config)",
                    config.model_type
                ),
            })
        }
    }

    /// Load a model from GGUF format
    pub fn load_gguf(config: &UniversalConfig, vb: &mut VarBuilder<R>) -> Result<Self> {
        Self::load(config, vb)
    }
}
