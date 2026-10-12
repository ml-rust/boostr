//! Construction of a [`LoadedModel`] from config and weights.

use super::model::LoadedModel;
use crate::error::{Error, Result};
use crate::format::peft::PeftAdapter;
use crate::model::config::UniversalConfig;
use crate::model::traits::Model;
use crate::nn::VarBuilder;
use numr::dtype::DType;
use numr::ops::{IndexingOps, ReduceOps, ScalarOps, ShapeOps};
use numr::runtime::Runtime;
use std::borrow::Cow;

/// `config` with the distinct quantization formats of `vb`'s tensors
/// folded into `quant_formats`, borrowed unchanged when there are none.
///
/// GGUF weights carry their tensors' `GgmlType`s into `vb`'s `VarMap` at
/// load time (`VarMap::from_gguf`); a SafeTensors-backed `vb` reports none.
/// Folding them into the config lets every model variant's
/// `quant_formats()` see them through `m.config()`, with no per-arch
/// `from_varbuilder` change needed.
fn with_quant_formats<'a, R: Runtime<DType = DType>>(
    config: &'a UniversalConfig,
    vb: &VarBuilder<R>,
) -> Cow<'a, UniversalConfig> {
    let formats = vb.quant_formats();
    if formats.is_empty() {
        Cow::Borrowed(config)
    } else {
        let mut c = config.clone();
        c.quant_formats = formats.to_vec();
        Cow::Owned(c)
    }
}

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
        let folded = with_quant_formats(config, vb);
        let config: &UniversalConfig = &folded;
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

impl<R: Runtime<DType = DType>> LoadedModel<R>
where
    R::Client: IndexingOps<R>
        + crate::quant::DequantOps<R>
        + numr::ops::TypeConversionOps<R>
        + ReduceOps<R>
        + ShapeOps<R>
        + ScalarOps<R>,
{
    /// [`Self::load`] with a PEFT LoRA adapter attached unmerged at load.
    /// `None` is [`Self::load`]. Only `qwen35` models take an adapter, through
    /// [`Qwen35Model::from_varbuilder_with_adapter`](crate::model::qwen35::Qwen35Model::from_varbuilder_with_adapter);
    /// [`Qwen35Model::adapted_modules`](crate::model::qwen35::Qwen35Model::adapted_modules)
    /// lists what it adapted.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] for an adapter on any other model type,
    /// checked before a weight is read. Otherwise as [`Self::load`] and
    /// `from_varbuilder_with_adapter`.
    pub fn load_with_adapter(
        config: &UniversalConfig,
        vb: &mut VarBuilder<R>,
        adapter: Option<&PeftAdapter<R>>,
    ) -> Result<Self> {
        let Some(adapter) = adapter else {
            return Self::load(config, vb);
        };
        if config.model_type != "qwen35" {
            return Err(Error::ModelError {
                reason: format!(
                    "a PEFT adapter attaches to qwen35 models only; this model is '{}'",
                    config.model_type
                ),
            });
        }
        let folded = with_quant_formats(config, vb);
        let model = crate::model::qwen35::Qwen35Model::from_varbuilder_with_adapter(
            vb,
            &folded,
            Some(adapter),
        )?;
        Ok(LoadedModel::Qwen35(Box::new(model)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::hf_qwen35::restructure::tests::maps;
    use crate::model::qwen35::Qwen35Model;
    use crate::model::qwen35::model::{ALL_MODULES, adapter_entries, tiny_config, write_adapter};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn qwen35(loaded: LoadedModel<CpuRuntime>) -> Box<Qwen35Model<CpuRuntime>> {
        match loaded {
            LoadedModel::Qwen35(model) => model,
            other => panic!("expected Qwen35, got {other:?}"),
        }
    }

    #[test]
    fn qwen35_takes_the_adapter() {
        let (client, device) = cpu_setup();
        let config = tiny_config();
        let dir = tempfile::tempdir().unwrap();
        let (hf, mut map) = maps(&client, &device, &config);
        let entries = adapter_entries(&hf, &ALL_MODULES, &device);
        let adapter = write_adapter(dir.path(), &entries, &device);

        let mut vb = VarBuilder::new(&mut map, &device);
        let model = qwen35(
            LoadedModel::<CpuRuntime>::load_with_adapter(&config, &mut vb, Some(&adapter)).unwrap(),
        );
        assert_eq!(model.adapted_modules().len(), ALL_MODULES.len());

        let (_, mut map) = maps(&client, &device, &config);
        let mut vb = VarBuilder::new(&mut map, &device);
        let model =
            qwen35(LoadedModel::<CpuRuntime>::load_with_adapter(&config, &mut vb, None).unwrap());
        assert!(model.adapted_modules().is_empty());
    }

    #[test]
    fn other_models_refuse_an_adapter_before_reading_weights() {
        let (client, device) = cpu_setup();
        let qwen_config = tiny_config();
        let dir = tempfile::tempdir().unwrap();
        let (hf, mut map) = maps(&client, &device, &qwen_config);
        let entries = adapter_entries(&hf, &ALL_MODULES, &device);
        let adapter = write_adapter(dir.path(), &entries, &device);

        let mut config = qwen_config;
        config.model_type = "llama".into();
        let before = map.len();
        let mut vb = VarBuilder::new(&mut map, &device);
        let err = LoadedModel::<CpuRuntime>::load_with_adapter(&config, &mut vb, Some(&adapter))
            .unwrap_err()
            .to_string();
        assert!(err.contains("qwen35 models only"), "{err}");
        assert!(err.contains("'llama'"), "{err}");
        assert_eq!(map.len(), before, "no weight was read");
    }
}
