//! Stable-id construction of a [`GdnBlock`], for a trainer that rebuilds
//! the block over fresh tensors after each optimizer step.
//!
//! # Gate pair
//!
//! [`GdnBlock::with_ids`] keeps `ssm_alpha` and `ssm_beta` apart, as
//! `GateProjections::Separate`. Each keeps its own `TensorId` and its own
//! names, `alpha_proj.*` and `beta_proj.*`. The fused pair that
//! [`GdnBlock::new`] builds concatenates both into one weight with a fresh
//! id. That weight exists in no checkpoint and gets a new id on every
//! rebuild, so it can neither train nor keep optimizer state.
//!
//! # Decay
//!
//! The block takes `A_log` and reports it under that name, as a block made
//! trainable by [`GdnBlock::with_trainable`] does. `a_neg_exp` never
//! appears. Every forward reads `-exp(A_log)`.

use super::gate_proj::{GateProjections, SeparateGates};
use super::layer::{GdnBlock, check_linear, check_shape, conv_weight_ncl};
use crate::error::{Error, Result};
use crate::model::config::GdnConfig;
use crate::nn::{GateOrder, GatedRmsNorm, MaybeLoraLinear, MaybeRotatedLinear};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::UnaryOps;
use numr::runtime::Runtime;
use numr::tensor::{Tensor, TensorId};

/// Weights for [`GdnBlock::with_ids`]. Each projection arrives built, so
/// its weight, bias and adapter factors carry the ids and `requires_grad`
/// the caller gave them, for example through `Linear::with_ids` and
/// `LoraLinear::with_ids`. Each raw tensor arrives with its id.
///
/// The field comments give the [`GdnBlock::named_parameters`] name.
pub struct GdnWeightsWithIds<R: Runtime> {
    /// `in_proj_qkv`, `[qkv_dim, hidden_size]`.
    pub attn_qkv: MaybeLoraLinear<R>,
    /// `in_proj_z`, `[value_dim, hidden_size]`.
    pub attn_gate: MaybeLoraLinear<R>,
    /// `alpha_proj`, `[value_heads, hidden_size]`.
    pub ssm_alpha: MaybeLoraLinear<R>,
    /// `beta_proj`, `[value_heads, hidden_size]`.
    pub ssm_beta: MaybeLoraLinear<R>,
    /// `out_proj`, `[hidden_size, value_dim]`.
    pub ssm_out: MaybeLoraLinear<R>,
    /// `conv1d.weight`, `[qkv_dim, 1, conv_kernel]` or `[qkv_dim, conv_kernel]`.
    pub conv1d: (Tensor<R>, TensorId),
    /// `A_log`, `[value_heads]`. The decay is `-exp(A_log)`.
    pub a_log: (Tensor<R>, TensorId),
    /// `dt_bias`, `[value_heads]`.
    pub dt_bias: (Tensor<R>, TensorId),
    /// `norm.weight`, `[state_size]`.
    pub norm: (Tensor<R>, TensorId),
}

impl<R: Runtime<DType = DType>> GdnBlock<R> {
    /// Build from a validated config and weights, keeping every given
    /// `TensorId`.
    ///
    /// The gate pair stays separate; see the module docs. `trainable` sets
    /// `requires_grad` on the conv kernel, `A_log`, `dt_bias` and the norm
    /// weight together, as [`Self::with_trainable`] does. The projections
    /// keep the flags they were built with.
    ///
    /// The block stores `-exp(A_log)` as its frozen `ssm_a`. No forward
    /// reads it while `A_log` exists, and each rebuild recomputes it.
    ///
    /// A rebuild over the tensors of [`Self::named_parameters`], under the
    /// same ids, gives the same forward bit for bit.
    ///
    /// # Errors
    ///
    /// - [`Error::ModelError`] when `cfg.validate()` fails or a shape
    ///   disagrees with `cfg`. The message names the tensor.
    /// - numr errors from `-exp(A_log)` propagate.
    pub fn with_ids<C: UnaryOps<R>>(
        client: &C,
        cfg: GdnConfig,
        weights: GdnWeightsWithIds<R>,
        trainable: bool,
    ) -> Result<Self> {
        cfg.validate()?;
        let (qkv_dim, value_dim) = (cfg.qkv_dim(), cfg.value_dim());
        let (heads, hidden) = (cfg.value_heads, cfg.hidden_size);
        check_linear(weights.attn_qkv.base(), "in_proj_qkv", qkv_dim, hidden)?;
        check_linear(weights.attn_gate.base(), "in_proj_z", value_dim, hidden)?;
        check_linear(weights.ssm_alpha.base(), "alpha_proj", heads, hidden)?;
        check_linear(weights.ssm_beta.base(), "beta_proj", heads, hidden)?;
        check_linear(weights.ssm_out.base(), "out_proj", hidden, value_dim)?;

        let (conv, conv_id) = weights.conv1d;
        let conv = conv_weight_ncl(conv, qkv_dim, cfg.conv_kernel)?;
        let (a_log, a_log_id) = weights.a_log;
        check_shape(&a_log, "A_log", &[heads])?;
        let (dt_bias, dt_bias_id) = weights.dt_bias;
        check_shape(&dt_bias, "dt_bias", &[heads])?;
        let (norm, norm_id) = weights.norm;
        check_shape(&norm, "norm.weight", &[cfg.state_size])?;

        let ssm_a = neg_exp(client, &a_log)?;
        let norm = GatedRmsNorm::with_id(
            norm,
            norm_id,
            cfg.rms_eps,
            GateOrder::NormThenMul,
            trainable,
        );
        let gates = GateProjections::Separate(Box::new(SeparateGates {
            alpha: weights.ssm_alpha,
            beta: weights.ssm_beta,
        }));
        Ok(Self {
            cfg,
            attn_qkv: rotatable(weights.attn_qkv),
            attn_gate: rotatable(weights.attn_gate),
            gates,
            ssm_out: rotatable(weights.ssm_out),
            conv_weight: Var::with_id(conv, conv_id, trainable),
            ssm_a: Var::new(ssm_a, false),
            a_log: Some(Var::with_id(a_log, a_log_id, trainable)),
            ssm_dt_bias: Var::with_id(dt_bias, dt_bias_id, trainable),
            norm,
        })
    }
}

/// The same projection as a [`MaybeRotatedLinear`], without a rotation.
fn rotatable<R: Runtime>(linear: MaybeLoraLinear<R>) -> MaybeRotatedLinear<R> {
    match linear {
        MaybeLoraLinear::Plain(base) => MaybeRotatedLinear::Plain(base),
        MaybeLoraLinear::Lora(lora) => MaybeRotatedLinear::Lora(lora),
    }
}

/// `-exp(a_log)` in `a_log`'s dtype.
fn neg_exp<R: Runtime, C: UnaryOps<R>>(client: &C, a_log: &Tensor<R>) -> Result<Tensor<R>> {
    let a = client.exp(a_log).map_err(Error::Numr)?;
    client.neg(&a).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::super::init::tests::{
        BATCH, SEQ, f64_bits, init_block, input, seeded_weighted_loss,
    };
    use super::*;
    use crate::nn::{Linear, Module};
    use crate::test_utils::cpu_setup;
    use numr::autograd::backward;
    use numr::ops::ScalarOps;
    use numr::runtime::cpu::{CpuClient, CpuRuntime};

    type Block = GdnBlock<CpuRuntime>;

    /// The tensor `value` gives for each named parameter, under that
    /// parameter's id, as a trainer rebuilds a block. Plain projections only.
    fn rebuild(
        client: &CpuClient,
        block: &Block,
        trainable: bool,
        value: impl Fn(&Tensor<CpuRuntime>) -> Tensor<CpuRuntime>,
    ) -> Result<Block> {
        let named = block.named_parameters();
        let pair = |name: &str| {
            let (_, var) = named
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("missing {name}"));
            (value(var.tensor()), var.id())
        };
        let linear = |name: &str| {
            let (_, var) = named.iter().find(|(n, _)| n == name).unwrap();
            let (tensor, id) = pair(name);
            MaybeLoraLinear::from(Linear::with_ids(tensor, id, None, var.requires_grad()))
        };
        let weights = GdnWeightsWithIds {
            attn_qkv: linear("in_proj_qkv.weight"),
            attn_gate: linear("in_proj_z.weight"),
            ssm_alpha: linear("alpha_proj.weight"),
            ssm_beta: linear("beta_proj.weight"),
            ssm_out: linear("out_proj.weight"),
            conv1d: pair("conv1d.weight"),
            a_log: pair("A_log"),
            dt_bias: pair("dt_bias"),
            norm: pair("norm.weight"),
        };
        GdnBlock::with_ids(client, block.config().clone(), weights, trainable)
    }

    fn ids(block: &Block) -> Vec<TensorId> {
        block.parameters().into_iter().map(|(id, _)| id).collect()
    }

    fn names(block: &Block) -> Vec<String> {
        block
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect()
    }

    /// A rebuild over the same values keeps every id and name, and its
    /// training forward equals the original bit for bit.
    #[test]
    fn rebuild_keeps_ids_and_forward() {
        let (client, device) = cpu_setup();
        let block = init_block(&client, &device, DType::F64, 0x6d4e_0100, true);
        let rebuilt = rebuild(&client, &block, true, Clone::clone).unwrap();
        assert_eq!(ids(&rebuilt), ids(&block));
        assert_eq!(names(&rebuilt), names(&block));
        assert_eq!(Module::trainable_parameters(&rebuilt).len(), 9);

        let x = input(&client, &device, DType::F64, 0x6d4e_0101, false);
        let want = block.forward_train(&client, &x, None).unwrap();
        let got = rebuilt.forward_train(&client, &x, None).unwrap();
        assert_eq!(f64_bits(got.tensor()), f64_bits(want.tensor()));
    }

    /// A rebuild over perturbed values reads them, and every gradient
    /// lands under the original block's ids.
    #[test]
    fn rebuild_over_new_values_trains_the_same_ids() {
        let (client, device) = cpu_setup();
        let block = init_block(&client, &device, DType::F64, 0x6d4e_0110, true);
        let nudge = |t: &Tensor<CpuRuntime>| client.add_scalar(t, 0.01).unwrap();
        let rebuilt = rebuild(&client, &block, true, nudge).unwrap();

        let x = input(&client, &device, DType::F64, 0x6d4e_0111, true);
        let before = block.forward_train(&client, &x, None).unwrap();
        let y = rebuilt.forward_train(&client, &x, None).unwrap();
        assert_ne!(
            f64_bits(y.tensor()),
            f64_bits(before.tensor()),
            "new values read"
        );
        assert_eq!(y.shape(), &[BATCH, SEQ, rebuilt.config().hidden_size]);

        let grads = backward(&seeded_weighted_loss(&client, &y, 0x6d4e_0112), &client).unwrap();
        assert!(grads.get(x.id()).is_some(), "x gets a gradient");
        for (name, var) in block.named_parameters() {
            let grad = grads
                .get(var.id())
                .unwrap_or_else(|| panic!("{name}: no gradient under its id"));
            let grad = grad.contiguous().unwrap().to_vec::<f64>();
            assert!(grad.iter().all(|g| g.is_finite()), "{name}");
        }
    }

    /// `trainable = false` freezes the raw tensors only. The projections
    /// keep the flags they were built with.
    #[test]
    fn frozen_rebuild_keeps_projection_flags() {
        let (client, device) = cpu_setup();
        let block = init_block(&client, &device, DType::F64, 0x6d4e_0120, true);
        let rebuilt = rebuild(&client, &block, false, Clone::clone).unwrap();
        let trainable: Vec<String> = rebuilt
            .named_parameters()
            .into_iter()
            .filter(|(_, var)| var.requires_grad())
            .map(|(name, _)| name)
            .collect();
        assert_eq!(
            trainable,
            [
                "in_proj_qkv.weight",
                "in_proj_z.weight",
                "alpha_proj.weight",
                "beta_proj.weight",
                "out_proj.weight",
            ]
        );
        assert_eq!(ids(&rebuilt), ids(&block));
    }

    /// A wrong shape is a `ModelError` that names the tensor.
    #[test]
    fn refuses_a_wrong_shape() {
        let (client, device) = cpu_setup();
        let block = init_block(&client, &device, DType::F64, 0x6d4e_0130, true);
        let heads = block.config().value_heads;
        let wrong = |t: &Tensor<CpuRuntime>| {
            if t.shape() == [heads] {
                Tensor::<CpuRuntime>::zeros(&[heads + 1], DType::F64, &device).unwrap()
            } else {
                t.clone()
            }
        };
        match rebuild(&client, &block, true, wrong) {
            Err(Error::ModelError { reason }) => assert!(reason.contains("A_log"), "{reason}"),
            Err(other) => panic!("expected ModelError, got {other}"),
            Ok(_) => panic!("a [heads + 1] A_log must be refused"),
        }
    }
}
