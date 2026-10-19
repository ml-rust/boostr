//! Training constructors of a [`Qwen35AttentionBlock`]: a rebuild that
//! keeps stable autograd ids ([`Qwen35AttentionBlock::with_ids`]), a load
//! from a `VarBuilder` ([`Qwen35AttentionBlock::load`]), and a load that
//! initializes absent weights ([`Qwen35AttentionBlock::init`]).
//!
//! `load` and `init` read the loader names under `self_attn` (see
//! `params`). The weights there carry the checkpoint layout, so `q_proj`
//! interleaves query and gate rows per head. `with_ids` takes the block's
//! own layout, the layout `named_parameters` reports.

use super::layer::{BlockLayout, Qwen35AttentionBlock, Qwen35AttentionWeights};
use crate::error::Result;
use crate::model::config::Qwen35AttentionConfig;
use crate::nn::{Init, Linear, MaybeQuantLinear, MaybeRotatedLinear, QkNorm, VarBuilder};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, RandomOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// Dense weights of a [`Qwen35AttentionBlock`], each with its stable
/// autograd id, in the block's own layout.
pub struct Qwen35AttentionWeightsWithIds<R: Runtime> {
    /// `[num_heads * 2 * head_dim, hidden_size]`, rows in the block's
    /// `[query | gate]` order: the tensor `named_parameters` reports as
    /// `q_proj.weight`.
    pub attn_q: (Tensor<R>, TensorId),
    /// `[num_kv_heads * head_dim, hidden_size]`.
    pub attn_k: (Tensor<R>, TensorId),
    /// `[num_kv_heads * head_dim, hidden_size]`.
    pub attn_v: (Tensor<R>, TensorId),
    /// `[hidden_size, num_heads * head_dim]`.
    pub attn_output: (Tensor<R>, TensorId),
    /// `[head_dim]`.
    pub attn_q_norm: (Tensor<R>, TensorId),
    /// `[head_dim]`.
    pub attn_k_norm: (Tensor<R>, TensorId),
}

impl<R: Runtime<DType = DType>> Qwen35AttentionBlock<R> {
    /// Rebuild from dense weights in the block's layout, keeping each
    /// `TensorId`, with `requires_grad = trainable` on every weight.
    ///
    /// A training loop that writes optimizer-updated tensors back through
    /// this keeps the optimizer state keyed by each id attached.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`](crate::error::Error::ModelError) when
    /// `cfg.validate()` fails or a shape disagrees with `cfg`.
    pub fn with_ids(
        cfg: Qwen35AttentionConfig,
        weights: Qwen35AttentionWeightsWithIds<R>,
        trainable: bool,
    ) -> Result<Self> {
        let linear = |(t, id): (Tensor<R>, TensorId)| -> MaybeRotatedLinear<R> {
            MaybeQuantLinear::Standard(Linear::with_ids(t, id, None, trainable)).into()
        };
        let qk_norm = QkNorm::with_ids(
            weights.attn_q_norm,
            weights.attn_k_norm,
            cfg.head_dim,
            cfg.rms_eps,
            trainable,
        )?;
        Self::assemble(
            cfg,
            BlockLayout {
                attn_q: linear(weights.attn_q),
                attn_k: linear(weights.attn_k),
                attn_v: linear(weights.attn_v),
                attn_output: linear(weights.attn_output),
                qk_norm,
            },
        )
    }

    /// Load from `vb` scoped to a layer's `self_attn`: `q_proj.weight`,
    /// `k_proj.weight`, `v_proj.weight`, `o_proj.weight`, `q_norm.weight`
    /// and `k_norm.weight`.
    ///
    /// `trainable` sets `requires_grad` on every dense weight and both
    /// norms. A block-quantized projection holds no `Var` and stays
    /// frozen. No Hadamard rotation is attached.
    ///
    /// # Errors
    ///
    /// A missing weight, or any error of [`Self::new`].
    pub fn load(
        cfg: &Qwen35AttentionConfig,
        vb: &mut VarBuilder<R>,
        trainable: bool,
    ) -> Result<Self> {
        let mut linear = |name: &str| -> Result<MaybeRotatedLinear<R>> {
            let base = match vb.take_maybe_quant_linear(name, None)? {
                MaybeQuantLinear::Standard(dense) => {
                    let w = dense.weight();
                    let dense = Linear::with_ids(w.tensor().clone(), w.id(), None, trainable);
                    MaybeQuantLinear::Standard(dense)
                }
                other => other,
            };
            Ok(base.into())
        };
        let attn_q = linear("q_proj.weight")?;
        let attn_k = linear("k_proj.weight")?;
        let attn_v = linear("v_proj.weight")?;
        let attn_output = linear("o_proj.weight")?;
        let weights = Qwen35AttentionWeights {
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            attn_q_norm: vb.take_tensor("q_norm.weight")?,
            attn_k_norm: vb.take_tensor("k_norm.weight")?,
        };
        Ok(Self::new(cfg.clone(), weights)?.with_trainable(trainable))
    }

    /// [`Self::load`] for fresh training: a weight absent from `vb` is
    /// initialized at `dtype`. Projections draw from `Init::PyTorchLinear`
    /// and the norm weights start at one. A present weight is taken and
    /// checked against the shape `cfg` gives.
    ///
    /// # Errors
    ///
    /// A shape mismatch on a present weight, an init error, or any error
    /// of [`Self::new`].
    pub fn init<C>(
        cfg: &Qwen35AttentionConfig,
        vb: &mut VarBuilder<R>,
        dtype: DType,
        client: &C,
        trainable: bool,
    ) -> Result<Self>
    where
        C: RuntimeClient<R>
            + RandomOps<R>
            + ScalarOps<R>
            + BinaryOps<R>
            + CompareOps<R>
            + TensorOps<R>,
    {
        let (hidden, hd) = (cfg.hidden_size, cfg.head_dim);
        let mut take = |name: &str, shape: &[usize], init: Init| {
            vb.take_or_init_tensor(name, shape, dtype, init, client)
        };
        let mut linear = |name: &str, shape: &[usize]| -> Result<MaybeRotatedLinear<R>> {
            let w = take(name, shape, Init::PyTorchLinear)?;
            Ok(MaybeQuantLinear::Standard(Linear::new(w, None, trainable)).into())
        };
        let attn_q = linear("q_proj.weight", &[cfg.q_gate_dim(), hidden])?;
        let attn_k = linear("k_proj.weight", &[cfg.kv_dim(), hidden])?;
        let attn_v = linear("v_proj.weight", &[cfg.kv_dim(), hidden])?;
        let attn_output = linear("o_proj.weight", &[hidden, cfg.q_dim()])?;
        let weights = Qwen35AttentionWeights {
            attn_q,
            attn_k,
            attn_v,
            attn_output,
            attn_q_norm: take("q_norm.weight", &[hd], Init::Ones)?,
            attn_k_norm: take("k_norm.weight", &[hd], Init::Ones)?,
        };
        Ok(Self::new(cfg.clone(), weights)?.with_trainable(trainable))
    }
}

#[cfg(test)]
mod tests {
    use super::super::train_fixture::{Fixture, SEQ, cfg32, positions, rope};
    use super::*;
    use crate::nn::VarMap;
    use crate::test_utils::{cpu_setup, to_f64};
    use numr::runtime::cpu::CpuRuntime;

    type Block = Qwen35AttentionBlock<CpuRuntime>;

    fn ids(block: &Block) -> Vec<TensorId> {
        block.parameters().into_iter().map(|(id, _)| id).collect()
    }

    fn output(block: &Block, fixture: &Fixture) -> Vec<f64> {
        let (client, device) = cpu_setup();
        let x = fixture.x(&device, false);
        let y = block
            .forward_train(&client, &x, &rope(&device), &positions(SEQ, &device))
            .unwrap();
        to_f64(y.tensor())
    }

    fn rebuild(block: &Block, trainable: bool) -> Block {
        let p: Vec<(Tensor<CpuRuntime>, TensorId)> = block
            .parameters()
            .into_iter()
            .map(|(id, var)| (var.tensor().clone(), id))
            .collect();
        let mut p = p.into_iter();
        let mut next = || p.next().unwrap();
        let weights = Qwen35AttentionWeightsWithIds {
            attn_q: next(),
            attn_k: next(),
            attn_v: next(),
            attn_output: next(),
            attn_q_norm: next(),
            attn_k_norm: next(),
        };
        Block::with_ids(block.config().clone(), weights, trainable).unwrap()
    }

    /// A rebuild from the block's own parameters keeps every id and the
    /// output, and trains every weight.
    #[test]
    fn with_ids_round_trips_the_block() {
        let (_, device) = cpu_setup();
        let fixture = Fixture::f32(&cfg32(), 0x3a7e_0060);
        let block = fixture.frozen_block(&device);
        let rebuilt = rebuild(&block, true);
        assert_eq!(ids(&rebuilt), ids(&block));
        assert_eq!(rebuilt.trainable_parameters().len(), 6);
        assert_eq!(output(&rebuilt, &fixture), output(&block, &fixture));
    }

    fn varmap(fixture: &Fixture) -> VarMap<CpuRuntime> {
        let (_, device) = cpu_setup();
        let mut map = VarMap::<CpuRuntime>::new();
        let names = [
            (1, "q_proj.weight"),
            (2, "k_proj.weight"),
            (3, "v_proj.weight"),
            (4, "o_proj.weight"),
            (5, "q_norm.weight"),
            (6, "k_norm.weight"),
        ];
        for (i, name) in names {
            map.insert(format!("self_attn.{name}"), fixture.tensor(i, &device));
        }
        map
    }

    /// `load` reads the checkpoint layout and matches `new` on the same
    /// weights; `trainable` reaches every weight.
    #[test]
    fn load_matches_new() {
        let (_, device) = cpu_setup();
        let fixture = Fixture::f32(&cfg32(), 0x3a7e_0061);
        let mut map = varmap(&fixture);
        let mut vb = VarBuilder::new(&mut map, &device);
        let mut attn_vb = vb.pp("self_attn");
        let block = Block::load(&fixture.cfg, &mut attn_vb, true).unwrap();
        assert_eq!(block.trainable_parameters().len(), 6);
        assert_eq!(
            output(&block, &fixture),
            output(&fixture.frozen_block(&device), &fixture)
        );
    }

    /// `init` over an empty map builds the configured shapes, with unit
    /// norms; over a filled map it equals `load`.
    #[test]
    fn init_fills_absent_weights_and_takes_present_ones() {
        let (client, device) = cpu_setup();
        let cfg = cfg32();
        let mut empty = VarMap::<CpuRuntime>::new();
        let mut vb = VarBuilder::new(&mut empty, &device).with_seed(7);
        let mut attn_vb = vb.pp("self_attn");
        let fresh = Block::init(&cfg, &mut attn_vb, DType::F32, &client, true).unwrap();
        let shapes: Vec<Vec<usize>> = fresh
            .parameters()
            .into_iter()
            .map(|(_, var)| var.shape().to_vec())
            .collect();
        let all = Fixture::f32(&cfg, 0).shapes();
        assert_eq!(shapes, all[1..].to_vec());
        let q_norm = fresh.qk_norm().q_norm().weight().tensor().to_vec::<f32>();
        assert!(q_norm.iter().all(|w| *w == 1.0));
        assert!(fresh.is_trainable());

        let fixture = Fixture::f32(&cfg, 0x3a7e_0062);
        let mut map = varmap(&fixture);
        let mut vb = VarBuilder::new(&mut map, &device);
        let mut attn_vb = vb.pp("self_attn");
        let taken = Block::init(&cfg, &mut attn_vb, DType::F32, &client, false).unwrap();
        assert_eq!(
            output(&taken, &fixture),
            output(&fixture.frozen_block(&device), &fixture)
        );
    }
}
