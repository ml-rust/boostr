//! [`FeedForward`]: the feed-forward half of a hybrid attention layer, a
//! SwiGLU MLP or a Mixture of Experts.

use crate::error::{Error, Result};
use crate::model::config::UniversalConfig;
use crate::model::traits::ModelClient;
use crate::nn::moe::{
    Expert, MOE_ROUTER_TEMPERATURE, MoeLayer, MoeLoadBalanceLossMode, MoeRouter, MoeRouterConfig,
    moe_shared_expert_scale,
};
use crate::nn::{Linear, VarBuilder};
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_mul, var_reshape, var_silu};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;

/// The gated MLP weights: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
pub(in crate::model::hybrid) struct GatedMlp<R: Runtime> {
    pub(in crate::model::hybrid) gate_proj: Linear<R>,
    pub(in crate::model::hybrid) up_proj: Linear<R>,
    pub(in crate::model::hybrid) down_proj: Linear<R>,
}

/// The feed-forward half of a hybrid attention layer.
pub(in crate::model::hybrid) enum FeedForward<R: Runtime> {
    /// The [`GatedMlp`] under `mlp.`.
    Mlp(Box<GatedMlp<R>>),
    /// The [`MoeLayer`] oxidizr trains, under `moe.`: `router.gate`,
    /// `experts.<i>.{gate,up,down}_proj`, and `shared_expert.*` when the
    /// config has one. It routes with [`MOE_ROUTER_TEMPERATURE`] and scales
    /// the shared expert by [`moe_shared_expert_scale`].
    Moe(Box<MoeLayer<R>>),
}

impl<R: Runtime<DType = DType>> FeedForward<R> {
    /// The feed-forward block of one layer: MoE under `moe.` when the config
    /// has a `moe` section, else the MLP under `mlp.`.
    ///
    /// # Errors
    ///
    /// A tensor the block reads is missing, or the shared-expert scale is
    /// not finite.
    pub(in crate::model::hybrid) fn load(
        layer_vb: &mut VarBuilder<R>,
        config: &UniversalConfig,
    ) -> Result<Self> {
        let Some(moe) = config.moe.as_ref() else {
            let mut mlp = layer_vb.pp("mlp");
            return Ok(Self::Mlp(Box::new(GatedMlp {
                gate_proj: Linear::new(mlp.take_tensor("gate_proj.weight")?, None, false),
                up_proj: Linear::new(mlp.take_tensor("up_proj.weight")?, None, false),
                down_proj: Linear::new(mlp.take_tensor("down_proj.weight")?, None, false),
            })));
        };
        let mut vb = layer_vb.pp("moe");
        let mut experts = Vec::with_capacity(moe.num_experts);
        for expert in 0..moe.num_experts {
            experts.push(take_expert(&mut vb.pp(&format!("experts.{expert}")))?);
        }
        let shared = if moe.has_shared_expert() {
            Some(take_expert(&mut vb.pp("shared_expert"))?)
        } else {
            None
        };
        let router_config = MoeRouterConfig {
            router_temperature: MOE_ROUTER_TEMPERATURE,
            load_balance_loss_mode: MoeLoadBalanceLossMode::SwitchPlusSquaredProb,
            ..MoeRouterConfig::new(moe.num_experts, moe.experts_per_tok)
        };
        let router =
            MoeRouter::from_tensor(vb.take_tensor("router.gate.weight")?, router_config, false);
        let layer = MoeLayer::new(router, experts, shared)
            .with_shared_expert_scale(moe_shared_expert_scale(moe.experts_per_tok))?;
        Ok(Self::Moe(Box::new(layer)))
    }

    /// `x` `[B, S, hidden]` → `[B, S, hidden]`.
    pub(in crate::model::hybrid) fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        match self {
            Self::Mlp(mlp) => {
                let gate = mlp.gate_proj.forward(client, x)?;
                let up = mlp.up_proj.forward(client, x)?;
                let gate = var_silu(&gate, client).map_err(Error::Numr)?;
                let hidden = var_mul(&gate, &up, client).map_err(Error::Numr)?;
                mlp.down_proj.forward(client, &hidden)
            }
            Self::Moe(moe) => {
                let shape = x.shape().to_vec();
                let &[batch, seq_len, hidden] = shape.as_slice() else {
                    return Err(Error::InvalidArgument {
                        arg: "x",
                        reason: format!("MoE expects [B, S, hidden], got {shape:?}"),
                    });
                };
                let tokens = var_reshape(x, &[batch * seq_len, hidden]).map_err(Error::Numr)?;
                let out = moe.forward(client, &tokens)?.output;
                var_reshape(&out, &[batch, seq_len, hidden]).map_err(Error::Numr)
            }
        }
    }
}

/// One SwiGLU expert under `vb`: `{gate,up,down}_proj.weight`.
fn take_expert<R: Runtime<DType = DType>>(vb: &mut VarBuilder<R>) -> Result<Expert<R>> {
    Ok(Expert::from_tensors(
        vb.take_tensor("gate_proj.weight")?,
        vb.take_tensor("up_proj.weight")?,
        vb.take_tensor("down_proj.weight")?,
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::VarMap;
    use crate::test_utils::{Lcg, cpu_setup, max_abs_diff};
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    fn config(top_k: usize, shared: bool) -> UniversalConfig {
        let yaml = format!(
            "model_type: hybrid\nvocab_size: 8\nhidden_size: 4\nnum_layers: 1\n\
             max_seq_len: 8\nmoe:\n  num_experts: 3\n  experts_per_tok: {top_k}\n  \
             shared_expert: {shared}\n  intermediate_size: 6\n"
        );
        serde_saphyr::from_str(&yaml).unwrap()
    }

    /// The MoE block loaded through `load`, and the same weights built as
    /// oxidizr builds its training layer.
    fn both(top_k: usize, shared: bool) -> (FeedForward<CpuRuntime>, MoeLayer<CpuRuntime>) {
        let (_, dev) = cpu_setup();
        let mut rng = Lcg(31 + top_k as u64);
        let mut tensors = Vec::new();
        let mut add = |name: String, shape: &[usize]| {
            let t = rng.tensor(&dev, shape, 0.5);
            tensors.push((name, t));
        };
        add("moe.router.gate.weight".into(), &[3, 4]);
        let mut prefixes: Vec<String> = (0..3).map(|e| format!("moe.experts.{e}")).collect();
        if shared {
            prefixes.push("moe.shared_expert".into());
        }
        for p in &prefixes {
            add(format!("{p}.gate_proj.weight"), &[6, 4]);
            add(format!("{p}.up_proj.weight"), &[6, 4]);
            add(format!("{p}.down_proj.weight"), &[4, 6]);
        }
        let get = |name: &str| {
            tensors
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, t)| t.clone())
                .unwrap()
        };
        let expert = |p: &str| {
            Expert::from_tensors(
                get(&format!("{p}.gate_proj.weight")),
                get(&format!("{p}.up_proj.weight")),
                get(&format!("{p}.down_proj.weight")),
                false,
            )
        };
        let router = MoeRouter::from_tensor(
            get("moe.router.gate.weight"),
            MoeRouterConfig {
                router_temperature: 2.0,
                load_balance_loss_mode: MoeLoadBalanceLossMode::SwitchPlusSquaredProb,
                ..MoeRouterConfig::new(3, top_k)
            },
            false,
        );
        let training = MoeLayer::new(
            router,
            (0..3)
                .map(|e| expert(&format!("moe.experts.{e}")))
                .collect(),
            shared.then(|| expert("moe.shared_expert")),
        )
        .with_shared_expert_scale(1.0 / (top_k as f32 + 1.0))
        .unwrap();

        let mut map = VarMap::<CpuRuntime>::new();
        for (name, t) in tensors {
            map.insert(name, t);
        }
        let mut vb = VarBuilder::new(&mut map, &dev);
        let loaded = FeedForward::load(&mut vb, &config(top_k, shared)).unwrap();
        (loaded, training)
    }

    #[test]
    fn moe_block_matches_the_trained_layer() {
        let (client, dev) = cpu_setup();
        for (top_k, shared) in [(1, true), (2, true), (2, false)] {
            let (loaded, training) = both(top_k, shared);
            let x = Lcg(9).tensor(&dev, &[1, 5, 4], 1.0);
            let got = loaded
                .forward(&client, &Var::new(x.clone(), false))
                .unwrap();
            let flat = Var::new(x.reshape(&[5, 4]).unwrap(), false);
            let want = training.forward(&client, &flat).unwrap().output;
            assert_eq!(got.shape(), &[1, 5, 4]);
            let diff = max_abs_diff(
                &got.tensor().to_vec::<f32>(),
                &want.tensor().contiguous().unwrap().to_vec::<f32>(),
            );
            assert!(diff == 0.0, "top_k {top_k} shared {shared}: diff {diff}");
        }
    }

    #[test]
    fn a_missing_expert_is_an_error() {
        let (_, dev) = cpu_setup();
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert(
            "moe.router.gate.weight".into(),
            Tensor::<CpuRuntime>::zeros(&[3, 4], DType::F32, &dev).unwrap(),
        );
        let mut vb = VarBuilder::new(&mut map, &dev);
        assert!(FeedForward::load(&mut vb, &config(1, false)).is_err());
    }
}
