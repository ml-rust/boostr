//! A linear layer that may carry a LoRA adapter.

use crate::error::Result;
use crate::nn::linear::Linear;
use crate::nn::lora::LoraLinear;
use crate::nn::module::Module;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::TensorId;

/// A linear projection that is either plain or LoRA-adapted.
///
/// Mirrors [`MaybeQuantLinear`](crate::nn::MaybeQuantLinear): model structs use a
/// single field type, and the variant decides how `forward` and parameter
/// enumeration behave. This is what lets a module (e.g. an MoE `Expert`) be
/// LoRA-adapted per projection without duplicating the module for each case.
// `Lora` is boxed because `LoraLinear` carries the base `Linear` plus both
// adapter factors; unboxed it made every `Plain` projection pay that footprint.
// `Plain` stays inline deliberately: it is the common variant (only targeted
// projections are adapted), and boxing it would add a heap allocation per
// projection per layer to buy back a size difference that no longer matters.
#[allow(clippy::large_enum_variant)]
pub enum MaybeLoraLinear<R: Runtime> {
    Plain(Linear<R>),
    Lora(Box<LoraLinear<R>>),
}

// `LoraLinear`'s inherent methods are bounded on `DType = DType`, so this
// block carries the same bound.
impl<R: Runtime<DType = DType>> MaybeLoraLinear<R> {
    /// Forward pass: plain linear, or base + scaled low-rank path.
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + TensorOps<R> + BinaryOps<R> + ScalarOps<R>,
        R::Client: TensorOps<R> + BinaryOps<R> + ScalarOps<R>,
    {
        match self {
            Self::Plain(linear) => linear.forward(client, input),
            Self::Lora(lora) => lora.forward(client, input),
        }
    }

    /// The underlying base linear layer, adapter aside.
    pub fn base(&self) -> &Linear<R> {
        match self {
            Self::Plain(linear) => linear,
            Self::Lora(lora) => lora.base(),
        }
    }

    /// The base weight `[out_features, in_features]`.
    pub fn weight(&self) -> &Var<R> {
        self.base().weight()
    }

    /// The base bias, if any.
    pub fn bias(&self) -> Option<&Var<R>> {
        self.base().bias()
    }

    /// The adapter factors `(lora_a, lora_b)`, or `None` when unadapted.
    pub fn adapters(&self) -> Option<(&Var<R>, &Var<R>)> {
        match self {
            Self::Plain(_) => None,
            Self::Lora(lora) => Some((lora.lora_a(), lora.lora_b())),
        }
    }

    /// `true` when a LoRA adapter is attached.
    pub fn is_adapted(&self) -> bool {
        matches!(self, Self::Lora(_))
    }

    /// All parameters with their stable autograd IDs — base plus adapters.
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        match self {
            Self::Plain(linear) => linear.parameters(),
            // `Linear` also has an INHERENT `parameters()` returning `(TensorId, &Var)`
            // pairs, which shadows the trait method — go through `Module` explicitly.
            Self::Lora(lora) => Module::parameters(lora.as_ref())
                .into_iter()
                .map(|var| (var.id(), var))
                .collect(),
        }
    }

    /// Trainable parameters with their stable autograd IDs.
    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.parameters()
            .into_iter()
            .filter(|param| param.1.requires_grad())
            .collect()
    }

    /// Named parameters for checkpoint traversal.
    pub fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        match self {
            Self::Plain(linear) => linear.named_parameters(),
            Self::Lora(lora) => lora.named_parameters(),
        }
    }

    /// Fold any adapter into the base weight, producing a plain `Linear`.
    ///
    /// The plain variant is rebuilt with [`Linear::with_ids`] so a merge keeps
    /// the base parameter identity intact; the adapted variant delegates to
    /// [`LoraLinear::merge_into_base`].
    pub fn merge_into_base<C>(&self, client: &C) -> Result<Linear<R>>
    where
        C: RuntimeClient<R> + TensorOps<R> + BinaryOps<R> + ScalarOps<R>,
        R::Client: TensorOps<R> + BinaryOps<R> + ScalarOps<R>,
    {
        match self {
            Self::Plain(linear) => {
                let weight = linear.weight();
                let bias = linear.bias().map(|b| (b.tensor().clone(), b.id()));
                Ok(Linear::with_ids(
                    weight.tensor().clone(),
                    weight.id(),
                    bias,
                    weight.requires_grad(),
                ))
            }
            Self::Lora(lora) => lora.merge_into_base(client),
        }
    }
}

impl<R: Runtime> From<Linear<R>> for MaybeLoraLinear<R> {
    fn from(linear: Linear<R>) -> Self {
        Self::Plain(linear)
    }
}

impl<R: Runtime> From<LoraLinear<R>> for MaybeLoraLinear<R> {
    fn from(lora: LoraLinear<R>) -> Self {
        Self::Lora(Box::new(lora))
    }
}

impl<R: Runtime<DType = DType>> Module<R> for MaybeLoraLinear<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        MaybeLoraLinear::parameters(self)
            .into_iter()
            .map(|param| param.1)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        MaybeLoraLinear::named_parameters(self)
    }

    fn parameters_with_ids(&self) -> Vec<(TensorId, &Var<R>)> {
        MaybeLoraLinear::parameters(self)
    }

    fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        MaybeLoraLinear::trainable_parameters(self)
    }
}
