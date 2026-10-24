//! [`GdnBlock::init`] and [`GdnBlock::load`]: a trainable block from a
//! [`VarBuilder`], under the names [`GdnBlock::named_parameters`] reports.
//!
//! Both build through [`GdnBlock::with_ids`]: separate gates, the decay as
//! `A_log`, every projection without a bias. A save of `named_parameters`
//! into a `VarMap` followed by `load` therefore gives the same block.
//!
//! # Init
//!
//! `init` follows `Qwen3NextPreTrainedModel._init_weights` in Hugging Face
//! transformers (`models/qwen3_next/modeling_qwen3_next.py`):
//!
//! - Projections and conv kernel: `N(0, initializer_range)`. The base
//!   `_init_weights` covers `nn.Linear` and `nn.Conv1d` alike.
//! - `dt_bias`: ones.
//! - `A_log`: `ln(U(0, 16))`.
//! - `norm.weight`: ones. The gated norm multiplies its weight in, so the
//!   base rule for an `RMSNorm` class applies. The zero init the reference
//!   gives `Qwen3NextRMSNorm` is for its `1 + weight` norms only.
//!
//! `initializer_range` defaults to 0.02 in `Qwen3NextConfig`.
//! [`GdnBlock::init`] uses that default. [`GdnBlock::init_with_range`]
//! takes another value.
//!
//! The draws are seeded per name when the builder carries
//! [`VarBuilder::with_seed`]. A tensor the builder already holds is taken
//! as is, shape-checked, and never redrawn.

use super::layer::GdnBlock;
use super::with_ids::GdnWeightsWithIds;
use crate::error::{Error, Result};
use crate::model::config::GdnConfig;
use crate::nn::{Init, Linear, MaybeLoraLinear, VarBuilder};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, RandomOps, ScalarOps, TensorOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// `initializer_range` default of `Qwen3NextConfig`.
const INITIALIZER_RANGE: f64 = 0.02;

/// Upper bound of the `U(0, A_MAX)` draw that `A_log = ln(A)` takes.
const A_MAX: f32 = 16.0;

/// The names [`GdnBlock::named_parameters`] reports for a block from
/// `init` or `load`, in its order.
const NAMES: [&str; 9] = [
    "in_proj_qkv.weight",
    "in_proj_z.weight",
    "alpha_proj.weight",
    "beta_proj.weight",
    "out_proj.weight",
    "conv1d.weight",
    "A_log",
    "dt_bias",
    "norm.weight",
];

impl<R: Runtime<DType = DType>> GdnBlock<R> {
    /// [`Self::init_with_range`] with the reference `initializer_range`,
    /// 0.02.
    ///
    /// # Errors
    ///
    /// As [`Self::init_with_range`].
    pub fn init<C>(
        cfg: &GdnConfig,
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
            + TensorOps<R>
            + UnaryOps<R>,
    {
        Self::init_with_range(cfg, vb, dtype, client, trainable, INITIALIZER_RANGE)
    }

    /// Take each tensor `vb` holds and draw the rest in `dtype`, as the
    /// module docs list. `initializer_range` is the standard deviation of
    /// the projection and conv draws. `trainable` sets `requires_grad` on
    /// every tensor.
    ///
    /// # Errors
    ///
    /// - [`Error::ModelError`] when `cfg.validate()` fails or a held
    ///   tensor's shape disagrees with `cfg`.
    /// - numr errors from the draws propagate.
    pub fn init_with_range<C>(
        cfg: &GdnConfig,
        vb: &mut VarBuilder<R>,
        dtype: DType,
        client: &C,
        trainable: bool,
        initializer_range: f64,
    ) -> Result<Self>
    where
        C: RuntimeClient<R>
            + RandomOps<R>
            + ScalarOps<R>
            + BinaryOps<R>
            + CompareOps<R>
            + TensorOps<R>
            + UnaryOps<R>,
    {
        cfg.validate()?;
        let normal = Init::Randn {
            mean: 0.0,
            stdev: initializer_range,
        };
        let (qkv, value, heads) = (cfg.qkv_dim(), cfg.value_dim(), cfg.value_heads);
        let hidden = cfg.hidden_size;
        let shapes: [&[usize]; 6] = [
            &[qkv, hidden],
            &[value, hidden],
            &[heads, hidden],
            &[heads, hidden],
            &[hidden, value],
            &[qkv, 1, cfg.conv_kernel],
        ];
        let mut drawn = Vec::with_capacity(NAMES.len());
        for (name, shape) in NAMES.iter().zip(shapes) {
            drawn.push(vb.take_or_init_tensor(name, shape, dtype, normal, client)?);
        }
        drawn.push(take_or_init_a_log(vb, heads, dtype, client)?);
        drawn.push(vb.take_or_init_tensor("dt_bias", &[heads], dtype, Init::Ones, client)?);
        let norm_shape = [cfg.state_size];
        drawn.push(vb.take_or_init_tensor(
            "norm.weight",
            &norm_shape,
            dtype,
            Init::Ones,
            client,
        )?);
        Self::from_tensors(client, cfg, drawn, trainable)
    }

    /// Take every tensor under the names of [`Self::named_parameters`] for
    /// a block from [`Self::init`] or [`Self::with_ids`]. Each tensor keeps
    /// its stored dtype. `trainable` sets `requires_grad` on every tensor.
    ///
    /// A frozen inference block's names differ: `a_neg_exp` and the fused
    /// `alpha_beta_proj`. `load` reads neither.
    ///
    /// # Errors
    ///
    /// - The `VarBuilder` error for a missing or quantized tensor.
    /// - [`Error::ModelError`] when `cfg.validate()` fails or a shape
    ///   disagrees with `cfg`.
    pub fn load<C: UnaryOps<R>>(
        cfg: &GdnConfig,
        vb: &mut VarBuilder<R>,
        client: &C,
        trainable: bool,
    ) -> Result<Self> {
        let mut taken = Vec::with_capacity(NAMES.len());
        for name in NAMES {
            taken.push(vb.take_tensor(name)?);
        }
        Self::from_tensors(client, cfg, taken, trainable)
    }

    /// A block over `tensors`, given in [`NAMES`] order, each with a
    /// fresh id.
    fn from_tensors<C: UnaryOps<R>>(
        client: &C,
        cfg: &GdnConfig,
        tensors: Vec<Tensor<R>>,
        trainable: bool,
    ) -> Result<Self> {
        let count = tensors.len();
        let Ok([qkv, gate, alpha, beta, out, conv, a_log, dt_bias, norm]) =
            <[Tensor<R>; 9]>::try_from(tensors)
        else {
            return Err(Error::ModelError {
                reason: format!("gdn: expected {} tensors, got {count}", NAMES.len()),
            });
        };
        let linear = |w: Tensor<R>| MaybeLoraLinear::from(Linear::new(w, None, trainable));
        let weights = GdnWeightsWithIds {
            attn_qkv: linear(qkv),
            attn_gate: linear(gate),
            ssm_alpha: linear(alpha),
            ssm_beta: linear(beta),
            ssm_out: linear(out),
            conv1d: (conv, TensorId::new()),
            a_log: (a_log, TensorId::new()),
            dt_bias: (dt_bias, TensorId::new()),
            norm: (norm, TensorId::new()),
        };
        Self::with_ids(client, cfg.clone(), weights, trainable)
    }
}

/// `A_log` from `vb`, or `ln(U(0, A_MAX))` drawn under the name `A_log`.
/// `U(0, A_MAX)` is the `U(-A_MAX / 2, A_MAX / 2)` draw shifted up.
fn take_or_init_a_log<R, C>(
    vb: &mut VarBuilder<R>,
    heads: usize,
    dtype: DType,
    client: &C,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + RandomOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + TensorOps<R>
        + UnaryOps<R>,
{
    if vb.contains("A_log") {
        return vb.take_or_init_tensor("A_log", &[heads], dtype, Init::Zeros, client);
    }
    let half = A_MAX / 2.0;
    let centered = vb.take_or_init_tensor("A_log", &[heads], dtype, Init::Uniform(half), client)?;
    let a = client
        .add_scalar(&centered, f64::from(half))
        .map_err(Error::Numr)?;
    client.log(&a).map_err(Error::Numr)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::nn::{Module, VarMap};
    use crate::test_utils::cpu_setup;
    use numr::autograd::{Var, var_mul, var_sum};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    pub(in super::super) const BATCH: usize = 2;
    /// 9 tokens in chunks of 4: the last chunk is padded.
    pub(in super::super) const SEQ: usize = 9;

    type Block = GdnBlock<CpuRuntime>;
    type T = Tensor<CpuRuntime>;

    /// Two key heads, four value heads: head repeat 2.
    pub(in super::super) fn cfg() -> GdnConfig {
        GdnConfig {
            hidden_size: 8,
            conv_kernel: 4,
            state_size: 4,
            key_heads: 2,
            value_heads: 4,
            inner_size: 16,
            rms_eps: 1e-6,
            chunk_size: 4,
            v_grouped: false,
        }
    }

    pub(in super::super) fn init_block(
        client: &CpuClient,
        device: &CpuDevice,
        dtype: DType,
        seed: u64,
        trainable: bool,
    ) -> Block {
        let mut map = VarMap::<CpuRuntime>::new();
        let mut vb = VarBuilder::new(&mut map, device).with_seed(seed);
        GdnBlock::init(&cfg(), &mut vb, dtype, client, trainable).unwrap()
    }

    /// `U(-1, 1)` of `shape`, seeded.
    pub(in super::super) fn uniform(
        client: &CpuClient,
        device: &CpuDevice,
        dtype: DType,
        shape: &[usize],
        seed: u64,
    ) -> T {
        Init::Uniform(1.0)
            .init_tensor_seeded(shape, dtype, device, client, seed)
            .unwrap()
    }

    /// `x` `[BATCH, SEQ, hidden_size]`.
    pub(in super::super) fn input(
        client: &CpuClient,
        device: &CpuDevice,
        dtype: DType,
        seed: u64,
        requires_grad: bool,
    ) -> Var<CpuRuntime> {
        let shape = [BATCH, SEQ, cfg().hidden_size];
        Var::new(uniform(client, device, dtype, &shape, seed), requires_grad)
    }

    /// `sum(y * w)` with a seeded `w`, so every output gets its own
    /// upstream gradient.
    pub(in super::super) fn seeded_weighted_loss(
        client: &CpuClient,
        y: &Var<CpuRuntime>,
        seed: u64,
    ) -> Var<CpuRuntime> {
        let t = y.tensor();
        let w = uniform(client, t.device(), t.dtype(), t.shape(), seed);
        let prod = var_mul(y, &Var::new(w, false), client).unwrap();
        var_sum(&prod, &[0, 1, 2], false, client).unwrap()
    }

    /// The bits of `t` as f64. F32 widens exactly, so equal bits mean
    /// equal source bits.
    pub(in super::super) fn f64_bits(t: &T) -> Vec<u64> {
        let t = t.contiguous().unwrap();
        match t.dtype() {
            DType::F64 => t.to_vec::<f64>().iter().map(|v| v.to_bits()).collect(),
            DType::F32 => t
                .to_vec::<f32>()
                .iter()
                .map(|v| f64::from(*v).to_bits())
                .collect(),
            other => panic!("f64_bits: unsupported dtype {other:?}"),
        }
    }

    fn named_bits(block: &Block) -> Vec<(String, Vec<u64>)> {
        block
            .named_parameters()
            .into_iter()
            .map(|(name, var)| (name, f64_bits(var.tensor())))
            .collect()
    }

    fn values(block: &Block, name: &str) -> Vec<f64> {
        let (_, var) = block
            .named_parameters()
            .into_iter()
            .find(|(n, _)| n == name)
            .unwrap();
        var.tensor().contiguous().unwrap().to_vec::<f64>()
    }

    /// Same seed, same tensors. Another seed moves every drawn tensor.
    #[test]
    fn init_is_deterministic_per_seed() {
        let (client, device) = cpu_setup();
        let a = init_block(&client, &device, DType::F64, 7, true);
        let b = init_block(&client, &device, DType::F64, 7, true);
        let c = init_block(&client, &device, DType::F64, 8, true);
        let names: Vec<String> = named_bits(&a).into_iter().map(|(n, _)| n).collect();
        assert_eq!(names, NAMES);
        assert_eq!(named_bits(&a), named_bits(&b));
        for ((name, x), (_, y)) in named_bits(&a).into_iter().zip(named_bits(&c)) {
            let constant = name == "dt_bias" || name == "norm.weight";
            assert_eq!(x == y, constant, "{name}: seed dependence");
        }
        assert_eq!(Module::trainable_parameters(&a).len(), NAMES.len());
    }

    /// The reference init: `N(0, 0.02)` projections and conv, `dt_bias`
    /// and norm ones, `exp(A_log)` in `[0, 16)`.
    #[test]
    fn init_follows_the_reference() {
        let (client, device) = cpu_setup();
        let block = init_block(&client, &device, DType::F64, 11, true);
        assert!(values(&block, "dt_bias").iter().all(|v| *v == 1.0));
        assert!(values(&block, "norm.weight").iter().all(|v| *v == 1.0));
        for a in values(&block, "A_log") {
            assert!(a.exp() < 16.0, "exp(A_log) = {}", a.exp());
        }
        let mut drawn = Vec::new();
        for name in &NAMES[..6] {
            drawn.extend(values(&block, name));
        }
        let n = drawn.len() as f64;
        let mean = drawn.iter().sum::<f64>() / n;
        let std = (drawn.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
        assert!(mean.abs() < 0.003, "mean {mean}");
        assert!((std - 0.02).abs() < 0.002, "std {std}");
    }

    /// A tensor the builder holds is taken as is: `A_log` is not
    /// transformed, `dt_bias` not reset.
    #[test]
    fn init_takes_held_tensors() {
        let (client, device) = cpu_setup();
        let heads = cfg().value_heads;
        let held = [0.25, -0.5, 1.0, 2.0];
        let mut map = VarMap::<CpuRuntime>::new();
        map.insert(
            "A_log".into(),
            T::from_slice(&held, &[heads], &device).unwrap(),
        );
        map.insert(
            "dt_bias".into(),
            T::from_slice(&held, &[heads], &device).unwrap(),
        );
        let mut vb = VarBuilder::new(&mut map, &device).with_seed(3);
        let block = GdnBlock::init(&cfg(), &mut vb, DType::F64, &client, true).unwrap();
        assert_eq!(values(&block, "A_log"), held);
        assert_eq!(values(&block, "dt_bias"), held);
    }

    /// `init`, save every named tensor into a `VarMap`, `load`: the same
    /// names, values and training forward. A missing tensor is an error.
    #[test]
    fn load_round_trips_named_parameters() {
        let (client, device) = cpu_setup();
        let block = init_block(&client, &device, DType::F64, 0x6d4e_0200, true);
        let save = || {
            let mut map = VarMap::<CpuRuntime>::new();
            for (name, var) in block.named_parameters() {
                map.insert(name, var.tensor().clone());
            }
            map
        };
        let mut map = save();
        let mut vb = VarBuilder::new(&mut map, &device);
        let loaded = GdnBlock::load(&cfg(), &mut vb, &client, true).unwrap();
        assert_eq!(named_bits(&loaded), named_bits(&block));
        assert_eq!(Module::trainable_parameters(&loaded).len(), NAMES.len());

        let x = input(&client, &device, DType::F64, 0x6d4e_0201, false);
        let want = block.forward_train(&client, &x, None).unwrap();
        let got = loaded.forward_train(&client, &x, None).unwrap();
        assert_eq!(f64_bits(got.tensor()), f64_bits(want.tensor()));

        let mut partial = save();
        partial.remove("A_log");
        let mut vb = VarBuilder::new(&mut partial, &device);
        assert!(GdnBlock::load(&cfg(), &mut vb, &client, true).is_err());
    }
}
