//! The attach behind [`Qwen35Model::from_varbuilder_with_adapter`]: every
//! adapter module resolved and checked, then attached unmerged.
//!
//! Every module resolves to one projection and passes its shape and device
//! checks before anything attaches (see `adapter_targets`). The factors
//! then move to the loader layout and attach through
//! [`LoraFactors::cast_for`] with the model's activation dtype,
//! [`Qwen35Model::dtype`]: a dense base's factors take its weight dtype, a
//! quantized base's the activation dtype. Both are known at load, so a
//! quantized base needs no cast at forward. The `in_proj_a`/`in_proj_b`
//! adapters of one layer attach together to the fused gate pair.

use super::adapter_targets::{Projection, Target, check_pair, resolve, to_loader_layout};
use super::build::{Qwen35Block, Qwen35Model};
use crate::error::{Error, Result};
use crate::format::peft::PeftAdapter;
use crate::nn::LoraFactors;
use numr::dtype::DType;
use numr::ops::{ScalarOps, ShapeOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};
use std::collections::BTreeMap;

/// The `ssm_alpha` and `ssm_beta` adapters of one layer.
type GateFactors<R> = (Option<LoraFactors<R>>, Option<LoraFactors<R>>);

fn in_module(module: &str, e: Error) -> Error {
    Error::ModelError {
        reason: format!("PEFT adapter module `{module}`: {e}"),
    }
}

impl<R: Runtime<DType = DType>> Qwen35Model<R> {
    /// Attach every module of `adapter`, casting and fusing the factors
    /// on `client`. Resolution and shape checks run for all modules before
    /// the first attach. A failed attach leaves the model partly adapted,
    /// so every caller drops it on error.
    pub(super) fn attach_peft_adapter<C>(
        &mut self,
        client: &C,
        adapter: &PeftAdapter<R>,
    ) -> Result<()>
    where
        C: RuntimeClient<R> + ScalarOps<R> + ShapeOps<R> + TypeConversionOps<R>,
        R::Client: ShapeOps<R> + ScalarOps<R> + TypeConversionOps<R>,
    {
        if !self.adapted_modules.is_empty() {
            return Err(Error::ModelError {
                reason: "qwen35: the model already carries a PEFT adapter".into(),
            });
        }
        if adapter.layers.is_empty() {
            return Err(Error::ModelError {
                reason: "qwen35: the PEFT adapter holds no modules".into(),
            });
        }
        let device = self.norm.weight().tensor().device().clone();
        let mut plan = Vec::with_capacity(adapter.layers.len());
        for (module, pair) in &adapter.layers {
            let target = resolve(module, &self.config)?;
            check_pair(
                module,
                pair,
                target.projection.base_shape(&self.config)?,
                &device,
            )?;
            plan.push((module, pair, target));
        }

        let activation = self.dtype();
        let mut gates: BTreeMap<usize, GateFactors<R>> = BTreeMap::new();
        for (module, pair, target) in plan {
            let factors = to_loader_layout(target.projection, pair, &self.config)
                .map_err(|e| in_module(module, e))?;
            match target.projection {
                Projection::GdnAlpha => gates.entry(target.layer).or_default().0 = Some(factors),
                Projection::GdnBeta => gates.entry(target.layer).or_default().1 = Some(factors),
                _ => self
                    .attach_one(client, target, factors, activation)
                    .map_err(|e| in_module(module, e))?,
            }
        }
        for (layer, (alpha, beta)) in gates {
            let gate_modules = format!("model.layers.{layer}.linear_attn.in_proj_{{a,b}}");
            match self.blocks.get_mut(layer) {
                Some(Qwen35Block::Gdn(block)) => block
                    .mixer
                    .attach_gate_adapters(client, alpha, beta, activation)
                    .map_err(|e| in_module(&gate_modules, e))?,
                _ => {
                    return Err(Error::ModelError {
                        reason: format!("qwen35: layer {layer} is not a Gated DeltaNet layer"),
                    });
                }
            }
        }
        self.adapted_modules = adapter.layers.keys().cloned().collect();
        Ok(())
    }

    /// Attach `factors` to the non-gate projection `target` names.
    fn attach_one<C: TypeConversionOps<R>>(
        &mut self,
        client: &C,
        target: Target,
        factors: LoraFactors<R>,
        activation: DType,
    ) -> Result<()> {
        let layer = target.layer;
        match (self.blocks.get_mut(layer), target.projection) {
            (Some(Qwen35Block::Attention(block)), Projection::Attention(p)) => {
                block.mixer.attach_adapter(client, p, factors, activation)
            }
            (Some(Qwen35Block::Gdn(block)), Projection::Gdn(p)) => {
                block.mixer.attach_adapter(client, p, factors, activation)
            }
            (Some(Qwen35Block::Attention(block)), Projection::Mlp(p)) => {
                block.mlp.attach_adapter(client, p, factors, activation)
            }
            (Some(Qwen35Block::Gdn(block)), Projection::Mlp(p)) => {
                block.mlp.attach_adapter(client, p, factors, activation)
            }
            (_, projection) => Err(Error::ModelError {
                reason: format!("qwen35: layer {layer} has no {projection:?} projection"),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::adapter::tests::{build, factors, write_adapter};
    use super::super::build::tests::rotated_model;
    use super::super::tiny_config;
    use super::*;
    use crate::format::hf_qwen35::restructure::tests::maps;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    const RANK: usize = 2;

    fn load_error(modules: &[(String, Tensor<CpuRuntime>, Tensor<CpuRuntime>)]) -> String {
        let (client, device) = cpu_setup();
        let config = tiny_config();
        let dir = tempfile::tempdir().unwrap();
        let (_, mut gguf) = maps(&client, &device, &config);
        let adapter = write_adapter(dir.path(), modules, &device);
        match build(&mut gguf, &config, &device, Some(&adapter)) {
            Ok(_) => panic!("the adapter must be refused"),
            Err(e) => e.to_string(),
        }
    }

    fn zeros(shape: &[usize]) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::zeros(shape, DType::F32, &CpuDevice::new()).unwrap()
    }

    #[test]
    fn unmatched_modules_are_errors() {
        let h = tiny_config().hidden_size;
        for module in [
            "layers.0.self_attn.q_proj",
            "layers.1.linear_attn.in_proj_z",
            "layers.1.mlp.fc1",
            "layers.5.mlp.up_proj",
        ] {
            let text = load_error(&[(module.to_string(), zeros(&[RANK, h]), zeros(&[4, RANK]))]);
            assert!(text.contains(&format!("model.{module}")), "{text}");
            assert!(text.contains("matches no qwen35 projection"), "{text}");
        }
    }

    #[test]
    fn shape_mismatch_names_the_module_and_both_shapes() {
        let config = tiny_config();
        let h = config.hidden_size;
        let value_dim = config.gdn.as_ref().unwrap().value_dim();
        let text = load_error(&[(
            "layers.0.linear_attn.in_proj_z".to_string(),
            zeros(&[RANK, h]),
            zeros(&[value_dim - 1, RANK]),
        )]);
        assert!(
            text.contains("model.layers.0.linear_attn.in_proj_z"),
            "{text}"
        );
        assert!(
            text.contains(&format!("[{}, {RANK}]", value_dim - 1)),
            "{text}"
        );
        assert!(text.contains(&format!("[{value_dim}, {h}]")), "{text}");
    }

    #[test]
    fn rotated_projection_is_an_error() {
        let (client, device) = cpu_setup();
        let mut model = rotated_model(&device, 0x3535_a001);
        let h = model.config().hidden_size;
        let inter = model.config().intermediate_size();
        let dir = tempfile::tempdir().unwrap();
        let adapter = write_adapter(
            dir.path(),
            &[(
                "layers.0.mlp.gate_proj".to_string(),
                zeros(&[RANK, h]),
                zeros(&[inter, RANK]),
            )],
            &device,
        );
        let text = model
            .attach_peft_adapter(&client, &adapter)
            .unwrap_err()
            .to_string();
        assert!(text.contains("model.layers.0.mlp.gate_proj"), "{text}");
        assert!(text.contains("Hadamard-rotated"), "{text}");
    }

    #[test]
    fn a_second_adapter_is_an_error() {
        let (client, device) = cpu_setup();
        let config = tiny_config();
        let dir = tempfile::tempdir().unwrap();
        let (hf, mut gguf) = maps(&client, &device, &config);
        let entries = factors(&hf, &["layers.0.mlp.up_proj"], &device);
        let adapter = write_adapter(dir.path(), &entries, &device);
        let mut model = build(&mut gguf, &config, &device, Some(&adapter)).unwrap();
        assert_eq!(model.adapted_modules(), ["model.layers.0.mlp.up_proj"]);
        let text = model
            .attach_peft_adapter(&client, &adapter)
            .unwrap_err()
            .to_string();
        assert!(text.contains("already carries"), "{text}");
    }
}
