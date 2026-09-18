//! A linear layer whose stored weight was quantized in a Hadamard-rotated
//! basis (the PrismML llama.cpp fork's activation-rotation contract — see
//! `crate::format::gguf::prism_hadamard`), plus the `Plain`/`Rotated`
//! dispatch enum mirroring [`crate::nn::maybe_lora::MaybeLoraLinear`].

use super::maybe_quant_linear::MaybeQuantLinear;
use crate::error::{Error, Result};
use crate::nn::hadamard::HadamardRotation;
use crate::nn::module::Module;
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{BinaryOps, FwhtOps, MatmulOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::TensorId;

/// A linear layer whose stored weight expects a Hadamard-rotated input.
///
/// The activation is rotated (sign-flip then transform, via
/// [`HadamardRotation::forward`]) before the underlying [`MaybeQuantLinear`]
/// runs. `fwht` carries no autograd, so the rotated activation is always a
/// detached leaf `Var` — matching how [`MaybeQuantLinear`] itself detaches a
/// quantized forward's output when its input needs no gradient.
pub struct RotatedLinear<R: Runtime> {
    inner: MaybeQuantLinear<R>,
    rotation: HadamardRotation<R>,
}

impl<R: Runtime<DType = DType>> RotatedLinear<R> {
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when `rotation` carries a sign
    /// vector whose width does not match `inner`'s `in_features`, or when
    /// `in_features` is not a multiple of `rotation.block_size()`.
    pub fn new(inner: MaybeQuantLinear<R>, rotation: HadamardRotation<R>) -> Result<Self> {
        let in_features = inner.shape()[1];
        if let Some(width) = rotation.width()
            && width != in_features
        {
            return Err(Error::InvalidArgument {
                arg: "rotation",
                reason: format!(
                    "rotation sign width {width} does not match the linear layer's \
                     in_features {in_features}"
                ),
            });
        }
        if !in_features.is_multiple_of(rotation.block_size()) {
            return Err(Error::InvalidArgument {
                arg: "rotation",
                reason: format!(
                    "in_features {in_features} is not a multiple of block_size {}",
                    rotation.block_size()
                ),
            });
        }
        Ok(Self { inner, rotation })
    }

    /// Rotates `input`, then runs the base linear layer.
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>
            + FwhtOps<R>,
        R::Client: TensorOps<R> + DequantOps<R> + MatmulOps<R>,
    {
        let rotated = self.rotation.forward(client, input.tensor())?;
        let rotated = Var::new(rotated, false);
        self.inner.forward(client, &rotated)
    }

    /// The wrapped base layer, rotation aside.
    pub fn base(&self) -> &MaybeQuantLinear<R> {
        &self.inner
    }

    pub fn in_features(&self) -> usize {
        self.inner.shape()[1]
    }

    pub fn out_features(&self) -> usize {
        self.inner.shape()[0]
    }

    pub fn weight(&self) -> Option<&Var<R>> {
        self.inner.weight()
    }

    pub fn bias(&self) -> Option<&Var<R>> {
        self.inner.bias()
    }

    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.inner.parameters()
    }

    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.inner.trainable_parameters()
    }

    pub fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        self.inner.named_parameters()
    }
}

/// A linear projection that is either plain or Hadamard-rotated, mirroring
/// [`crate::nn::maybe_lora::MaybeLoraLinear`]'s `Plain`/adapted dispatch.
// `Rotated` is boxed for the same reason `MaybeLoraLinear::Lora` is: `Plain`
// is the common variant and stays inline, so the size gap is accepted.
#[allow(clippy::large_enum_variant)]
pub enum MaybeRotatedLinear<R: Runtime> {
    Plain(MaybeQuantLinear<R>),
    Rotated(Box<RotatedLinear<R>>),
}

impl<R: Runtime<DType = DType>> MaybeRotatedLinear<R> {
    /// Forward pass: plain base, or rotate-then-base.
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>
            + FwhtOps<R>,
        R::Client: TensorOps<R> + DequantOps<R> + MatmulOps<R>,
    {
        match self {
            Self::Plain(base) => base.forward(client, input),
            Self::Rotated(rotated) => rotated.forward(client, input),
        }
    }

    /// Forward several projections that share one input.
    ///
    /// Unrotated layers go through [`MaybeQuantLinear::forward_batch`]. A
    /// rotated layer anywhere in the batch sends every layer down its own
    /// `forward` instead — a batch half rotated and half not is not one
    /// operation. A rotated layer's own `forward` still rotates the
    /// WHOLE (possibly batched) input in a single `fwht` call: `fwht`
    /// applies per `block_size` segment of the last axis regardless of how
    /// many leading (batch) dimensions the tensor carries.
    pub fn forward_batch<C>(layers: &[&Self], client: &C, input: &Var<R>) -> Result<Vec<Var<R>>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>
            + FwhtOps<R>,
        R::Client: TensorOps<R> + DequantOps<R> + MatmulOps<R>,
    {
        let plain: Option<Vec<&MaybeQuantLinear<R>>> = layers
            .iter()
            .map(|layer| match layer {
                Self::Plain(base) => Some(base),
                Self::Rotated(_) => None,
            })
            .collect();
        match plain {
            Some(bases) => MaybeQuantLinear::forward_batch(&bases, client, input),
            None => layers
                .iter()
                .map(|layer| layer.forward(client, input))
                .collect(),
        }
    }

    /// The underlying base linear layer, rotation aside.
    pub fn base(&self) -> &MaybeQuantLinear<R> {
        match self {
            Self::Plain(base) => base,
            Self::Rotated(rotated) => rotated.base(),
        }
    }

    pub fn weight(&self) -> Option<&Var<R>> {
        self.base().weight()
    }

    pub fn bias(&self) -> Option<&Var<R>> {
        self.base().bias()
    }

    /// `true` when a Hadamard rotation is attached.
    pub fn is_rotated(&self) -> bool {
        matches!(self, Self::Rotated(_))
    }

    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.base().parameters()
    }

    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.base().trainable_parameters()
    }

    pub fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        self.base().named_parameters()
    }
}

impl<R: Runtime<DType = DType>> From<MaybeQuantLinear<R>> for MaybeRotatedLinear<R> {
    fn from(base: MaybeQuantLinear<R>) -> Self {
        Self::Plain(base)
    }
}

impl<R: Runtime<DType = DType>> From<RotatedLinear<R>> for MaybeRotatedLinear<R> {
    fn from(rotated: RotatedLinear<R>) -> Self {
        Self::Rotated(Box::new(rotated))
    }
}

impl<R: Runtime<DType = DType>> Module<R> for MaybeRotatedLinear<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        MaybeRotatedLinear::parameters(self)
            .into_iter()
            .map(|param| param.1)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        MaybeRotatedLinear::named_parameters(self)
    }

    fn parameters_with_ids(&self) -> Vec<(TensorId, &Var<R>)> {
        MaybeRotatedLinear::parameters(self)
    }

    fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        MaybeRotatedLinear::trainable_parameters(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::linear::dense::Linear;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    fn build_linear_rotation(
        device: &<CpuRuntime as Runtime>::Device,
    ) -> (MaybeQuantLinear<CpuRuntime>, HadamardRotation<CpuRuntime>) {
        // in=16, out=4, block=8
        let weight_data: Vec<f32> = (0..4 * 16).map(|i| ((i as f32) * 0.037).sin()).collect();
        let weight = Tensor::<CpuRuntime>::from_slice(&weight_data, &[4, 16], device).unwrap();
        let linear = MaybeQuantLinear::Standard(Linear::new(weight, None, false));

        let signs: Vec<i8> = (0..16).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
        let rotation =
            HadamardRotation::<CpuRuntime>::new(8, Some(&signs), DType::F32, device).unwrap();
        (linear, rotation)
    }

    #[test]
    fn rotated_linear_matches_manual_rotate_then_forward() {
        let (client, device) = cpu_setup();
        let (linear, rotation) = build_linear_rotation(&device);

        let x_data: Vec<f32> = (0..2 * 16).map(|i| (i as f32) * 0.05 - 0.3).collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_data, &[2, 16], &device).unwrap(),
            false,
        );

        // Rebuild an identical base + rotation for the manual reference, since
        // `RotatedLinear::new` below takes `linear`/`rotation` by value.
        let (linear_ref, rotation_ref) = build_linear_rotation(&device);
        let manual_rotated = rotation_ref.forward(&client, x.tensor()).unwrap();
        let manual_out = match &linear_ref {
            MaybeQuantLinear::Standard(dense) => dense
                .forward(&client, &Var::new(manual_rotated, false))
                .unwrap(),
            _ => unreachable!(),
        };

        let rotated_linear = RotatedLinear::new(linear, rotation).unwrap();
        let out = rotated_linear.forward(&client, &x).unwrap();

        let got: Vec<f32> = out.tensor().to_vec();
        let expected: Vec<f32> = manual_out.tensor().to_vec();
        assert_eq!(got, expected);
    }

    #[test]
    fn width_mismatch_errors() {
        let (_client, device) = cpu_setup();
        let weight =
            Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4 * 16], &[4, 16], &device).unwrap();
        let linear = MaybeQuantLinear::Standard(Linear::new(weight, None, false));

        // Sign width 8, but in_features is 16 — mismatch.
        let signs: Vec<i8> = vec![1; 8];
        let rotation =
            HadamardRotation::<CpuRuntime>::new(8, Some(&signs), DType::F32, &device).unwrap();

        assert!(RotatedLinear::new(linear, rotation).is_err());
    }

    #[test]
    fn maybe_rotated_plain_forwards_identically_to_bare_maybe_quant_linear() {
        let (client, device) = cpu_setup();
        let weight_data: Vec<f32> = (0..4 * 16).map(|i| (i as f32) * 0.02).collect();
        let weight = Tensor::<CpuRuntime>::from_slice(&weight_data, &[4, 16], &device).unwrap();
        let bare = MaybeQuantLinear::Standard(Linear::new(
            Tensor::<CpuRuntime>::from_slice(&weight_data, &[4, 16], &device).unwrap(),
            None,
            false,
        ));
        let plain: MaybeRotatedLinear<CpuRuntime> =
            MaybeQuantLinear::Standard(Linear::new(weight, None, false)).into();

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; 16], &[1, 16], &device).unwrap(),
            false,
        );

        let bare_out = bare.forward(&client, &x).unwrap();
        let plain_out = plain.forward(&client, &x).unwrap();
        assert_eq!(
            bare_out.tensor().to_vec::<f32>(),
            plain_out.tensor().to_vec::<f32>()
        );
        assert!(!plain.is_rotated());
    }
}
