//! A linear layer that may carry a LoRA adapter.

use crate::error::{Error, Result};
use crate::nn::linear::{Linear, MaybeQuantLinear};
use crate::nn::lora::LoraLinear;
use crate::nn::module::Module;
use crate::quant::traits::QuantMatmulOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::TensorId;

/// A linear projection that is either plain or LoRA-adapted.
///
/// `Plain` carries a [`MaybeQuantLinear`] rather than a bare [`Linear`] so a
/// projection can be dense, block-quantized, or decomposed with or without an
/// adapter — this is what lets a module (e.g. an MoE `Expert`) be LoRA-adapted
/// per projection, dense or QUANTIZED, without duplicating the module for each
/// case.
// `Lora` is boxed because `LoraLinear` carries the base `MaybeQuantLinear`
// plus both adapter factors; unboxed it made every `Plain` projection pay
// that footprint. `Plain` stays inline deliberately: it is the common variant
// (only targeted projections are adapted), and boxing it would add a heap
// allocation per projection per layer to buy back a size difference that no
// longer matters.
#[allow(clippy::large_enum_variant)]
pub enum MaybeLoraLinear<R: Runtime> {
    Plain(MaybeQuantLinear<R>),
    Lora(Box<LoraLinear<R>>),
}

// `LoraLinear`'s inherent methods are bounded on `DType = DType`, so this
// block carries the same bound.
impl<R: Runtime<DType = DType>> MaybeLoraLinear<R> {
    /// Forward pass: plain base, or base + scaled low-rank path.
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + QuantMatmulOps<R>
            + TypeConversionOps<R>,
        R::Client: TensorOps<R> + BinaryOps<R> + ScalarOps<R>,
    {
        match self {
            Self::Plain(base) => base.forward(client, input),
            Self::Lora(lora) => lora.forward(client, input),
        }
    }

    /// The underlying base linear layer, adapter aside.
    pub fn base(&self) -> &MaybeQuantLinear<R> {
        match self {
            Self::Plain(base) => base,
            Self::Lora(lora) => lora.base(),
        }
    }

    /// The base weight `[out_features, in_features]`, if it is `Var`-wrapped.
    ///
    /// `None` when the base is quantized: block-quantized and decomposed
    /// storage carry no trainable `Var<R>` weight.
    pub fn weight(&self) -> Option<&Var<R>> {
        self.base().weight()
    }

    /// The base bias, if any and if it is `Var`-wrapped. Mirrors [`Self::weight`].
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

    /// All parameters with their stable autograd IDs.
    ///
    /// `Plain` enumerates whatever the base itself exposes (all of it, since
    /// nothing here is frozen by construction — dense or quantized).
    /// `Lora`'s base IS frozen by construction, so only the adapter factors
    /// come back; see [`LoraLinear`]'s `Module` impl.
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        match self {
            // `MaybeQuantLinear` also has an INHERENT `parameters()` returning
            // `(TensorId, &Var)` pairs, which method syntax resolves to directly.
            Self::Plain(base) => base.parameters(),
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
            Self::Plain(base) => base.named_parameters(),
            Self::Lora(lora) => lora.named_parameters(),
        }
    }

    /// Fold any adapter into the base weight, producing a plain `Linear`.
    ///
    /// The plain dense variant is rebuilt with [`Linear::with_ids`] so a merge
    /// keeps the base parameter identity intact; the adapted variant delegates
    /// to [`LoraLinear::merge_into_base`]. A plain QUANTIZED base (no adapter
    /// attached) has no `Var<R>` weight to rebuild a `Linear` from, so that
    /// case errors too — there is nothing to "merge" and no dense result to
    /// hand back without requantizing.
    pub fn merge_into_base<C>(&self, client: &C) -> Result<Linear<R>>
    where
        C: RuntimeClient<R> + TensorOps<R> + BinaryOps<R> + ScalarOps<R>,
        R::Client: TensorOps<R> + BinaryOps<R> + ScalarOps<R>,
    {
        match self {
            Self::Plain(MaybeQuantLinear::Standard(linear)) => {
                let weight = linear.weight();
                let bias = linear.bias().map(|b| (b.tensor().clone(), b.id()));
                Ok(Linear::with_ids(
                    weight.tensor().clone(),
                    weight.id(),
                    bias,
                    weight.requires_grad(),
                ))
            }
            Self::Plain(MaybeQuantLinear::Quantized(_) | MaybeQuantLinear::DecomposedQuant(_)) => {
                Err(Error::ModelError {
                    reason: "cannot produce a plain Linear from a quantized base with no LoRA \
                             adapter attached — merging would require requantizing the result; \
                             use the quantized base as-is instead of merging"
                        .into(),
                })
            }
            Self::Lora(lora) => lora.merge_into_base(client),
        }
    }
}

impl<R: Runtime> From<Linear<R>> for MaybeLoraLinear<R> {
    fn from(linear: Linear<R>) -> Self {
        Self::Plain(linear.into())
    }
}

impl<R: Runtime> From<MaybeQuantLinear<R>> for MaybeLoraLinear<R> {
    fn from(base: MaybeQuantLinear<R>) -> Self {
        Self::Plain(base)
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
