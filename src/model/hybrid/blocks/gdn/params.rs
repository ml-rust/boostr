//! Parameter access for a [`GdnBlock`] and its gate pair.
//!
//! Names follow the loader layout under `linear_attn` (see
//! `format::gguf::name_map`): `in_proj_qkv`, `in_proj_z`, `alpha_proj`,
//! `beta_proj`, `out_proj`, `conv1d.weight`, `dt_bias` and `norm.weight`.
//! An adapted projection adds `base.*`, `lora_a` and `lora_b`, as
//! [`LoraLinear`](crate::nn::LoraLinear) does.
//!
//! The decay has two names. A frozen block reports the loaded `ssm_a` as
//! `a_neg_exp`. A block made trainable reports `a_log` as `A_log`, the
//! Hugging Face name, and no longer reports `a_neg_exp`.
//!
//! A fused gate pair reports its concatenated base as
//! `alpha_beta_proj.weight`. That weight exists in no checkpoint and
//! stays frozen. Only a LoRA adapter on the fused pair trains, as
//! `alpha_beta_proj.lora_a` and the block-diagonal `alpha_beta_proj.lora_b`.
//! A separate pair reports `alpha_proj.*` and `beta_proj.*`.

use super::gate_proj::GateProjections;
use super::layer::GdnBlock;
use crate::nn::{Module, extend_named};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::TensorId;

/// Name prefix of the fused gate pair.
const FUSED_GATES: &str = "alpha_beta_proj";

impl<R: Runtime<DType = DType>> GateProjections<R> {
    /// All parameters with their stable autograd ids.
    pub(super) fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        match self {
            Self::Fused { linear, .. } => linear.parameters(),
            Self::FusedAdapted { lora, .. } => Module::parameters_with_ids(lora.as_ref()),
            Self::Separate(gates) => {
                let mut params = gates.alpha.parameters();
                params.extend(gates.beta.parameters());
                params
            }
        }
    }

    /// Named parameters, in the order of [`Self::parameters`].
    pub(super) fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        match self {
            Self::Fused { linear, .. } => {
                extend_named(
                    &mut params,
                    FUSED_GATES,
                    Module::named_parameters(linear.as_ref()),
                );
            }
            Self::FusedAdapted { lora, .. } => {
                extend_named(
                    &mut params,
                    FUSED_GATES,
                    Module::named_parameters(lora.as_ref()),
                );
            }
            Self::Separate(gates) => {
                extend_named(&mut params, "alpha_proj", gates.alpha.named_parameters());
                extend_named(&mut params, "beta_proj", gates.beta.named_parameters());
            }
        }
        params
    }
}

impl<R: Runtime<DType = DType>> GdnBlock<R> {
    /// All parameters with their stable autograd ids: projections, gate
    /// pair, conv kernel, decay, `dt_bias` and norm weight.
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        let mut params = self.attn_qkv.parameters();
        params.extend(self.attn_gate.parameters());
        params.extend(self.gates.parameters());
        params.extend(self.ssm_out.parameters());
        let (_, decay) = self.decay_param();
        for var in [&self.conv_weight, decay, &self.ssm_dt_bias] {
            params.push((var.id(), var));
        }
        params.extend(self.norm.parameters());
        params
    }

    /// Parameters that require grad.
    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.parameters()
            .into_iter()
            .filter(|param| param.1.requires_grad())
            .collect()
    }

    /// Named parameters in loader-layout names, in the order of
    /// [`Self::parameters`]. See the module docs for the decay and gate
    /// names.
    pub fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        extend_named(&mut params, "in_proj_qkv", self.attn_qkv.named_parameters());
        extend_named(&mut params, "in_proj_z", self.attn_gate.named_parameters());
        params.extend(self.gates.named_parameters());
        extend_named(&mut params, "out_proj", self.ssm_out.named_parameters());
        params.push(("conv1d.weight".to_string(), &self.conv_weight));
        let (name, decay) = self.decay_param();
        params.push((name.to_string(), decay));
        params.push(("dt_bias".to_string(), &self.ssm_dt_bias));
        extend_named(&mut params, "norm", Module::named_parameters(&self.norm));
        params
    }

    /// The decay parameter and its name: `a_log` once trainable, else the
    /// loaded `ssm_a`.
    fn decay_param(&self) -> (&'static str, &Var<R>) {
        match &self.a_log {
            Some(a_log) => ("A_log", a_log),
            None => ("a_neg_exp", &self.ssm_a),
        }
    }
}

impl<R: Runtime<DType = DType>> Module<R> for GdnBlock<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        GdnBlock::parameters(self)
            .into_iter()
            .map(|param| param.1)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        GdnBlock::named_parameters(self)
    }

    fn parameters_with_ids(&self) -> Vec<(TensorId, &Var<R>)> {
        GdnBlock::parameters(self)
    }

    fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        GdnBlock::trainable_parameters(self)
    }
}

#[cfg(test)]
mod tests {
    use super::super::forward::tests::block;
    use super::super::gate_proj::SeparateGates;
    use super::*;
    use crate::nn::{Linear, LoraFactors, MaybeQuantLinear};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    fn names(params: Vec<(String, &Var<CpuRuntime>)>) -> Vec<String> {
        params.into_iter().map(|(name, _)| name).collect()
    }

    fn named_trainable(block: &GdnBlock<CpuRuntime>) -> Vec<String> {
        block
            .named_parameters()
            .into_iter()
            .filter(|(_, var)| var.requires_grad())
            .map(|(name, _)| name)
            .collect()
    }

    fn zeros(shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::zeros(shape, DType::F32, device).unwrap()
    }

    /// A frozen block lists every weight under its loader name and trains
    /// nothing. Names and ids line up with `parameters`.
    #[test]
    fn frozen_block_names_every_weight() {
        let (_, device) = cpu_setup();
        let block = block(&device, 0x6d4e_0020, false);
        let named = block.named_parameters();
        let ids: Vec<TensorId> = named.iter().map(|(_, var)| var.id()).collect();
        let want_ids: Vec<TensorId> = block.parameters().into_iter().map(|(id, _)| id).collect();
        assert_eq!(ids, want_ids);
        assert_eq!(
            names(named),
            [
                "in_proj_qkv.weight",
                "in_proj_z.weight",
                "alpha_beta_proj.weight",
                "out_proj.weight",
                "conv1d.weight",
                "a_neg_exp",
                "dt_bias",
                "norm.weight",
            ]
        );
        assert!(block.trainable_parameters().is_empty());
        assert_eq!(Module::parameters(&block).len(), 8);
    }

    /// A trainable block trains conv, `A_log`, `dt_bias` and the norm, and
    /// leaves the frozen projections and fused gates alone.
    #[test]
    fn trainable_block_trains_its_own_weights() {
        let (client, device) = cpu_setup();
        let block = block(&device, 0x6d4e_0021, false)
            .with_trainable(&client, true)
            .unwrap();
        assert_eq!(
            named_trainable(&block),
            ["conv1d.weight", "A_log", "dt_bias", "norm.weight"]
        );
        assert!(!names(block.named_parameters()).contains(&"a_neg_exp".to_string()));
        assert_eq!(Module::trainable_parameters(&block).len(), 4);
    }

    /// Trainable gate and projection adapters train their factors only.
    #[test]
    fn trainable_adapters_train_factors_only() {
        let (client, device) = cpu_setup();
        let mut block = block(&device, 0x6d4e_0022, false);
        let cfg = block.config().clone();
        let factors = |out: usize, inp: usize| LoraFactors {
            lora_a: zeros(&[2, inp], &device),
            lora_b: zeros(&[out, 2], &device),
            scaling: 1.0,
        };
        block
            .attach_gate_adapters_trainable(
                &client,
                Some(factors(cfg.value_heads, cfg.hidden_size)),
                None,
                DType::F32,
                true,
            )
            .unwrap();
        block
            .attach_adapter_trainable(
                &client,
                crate::model::hybrid::GdnProjection::SsmOut,
                factors(cfg.hidden_size, cfg.value_dim()),
                DType::F32,
                true,
            )
            .unwrap();
        assert_eq!(
            named_trainable(&block),
            [
                "alpha_beta_proj.lora_a",
                "alpha_beta_proj.lora_b",
                "out_proj.lora_a",
                "out_proj.lora_b",
            ]
        );
        assert!(names(block.named_parameters()).contains(&"alpha_beta_proj.base.weight".into()));

        let mut frozen = super::super::forward::tests::block(&device, 0x6d4e_0023, false);
        frozen
            .attach_gate_adapters(
                &client,
                Some(factors(cfg.value_heads, cfg.hidden_size)),
                None,
                DType::F32,
            )
            .unwrap();
        assert!(frozen.trainable_parameters().is_empty());
    }

    /// A separate pair names each projection and trains a trainable base.
    #[test]
    fn separate_gates_name_each_projection() {
        let (_, device) = cpu_setup();
        let dense = |trainable| {
            MaybeQuantLinear::Standard(Linear::new(zeros(&[4, 8], &device), None, trainable))
        };
        let gates = GateProjections::Separate(Box::new(SeparateGates {
            alpha: dense(true).into(),
            beta: dense(false).into(),
        }));
        let named = gates.named_parameters();
        let trainable: Vec<&str> = named
            .iter()
            .filter(|(_, var)| var.requires_grad())
            .map(|(name, _)| name.as_str())
            .collect();
        assert_eq!(trainable, ["alpha_proj.weight"]);
        assert_eq!(names(named), ["alpha_proj.weight", "beta_proj.weight"]);
        assert_eq!(gates.parameters().len(), 2);
    }
}
