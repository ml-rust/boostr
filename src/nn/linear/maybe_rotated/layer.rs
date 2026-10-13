//! The [`MaybeRotatedLinear`] type: a projection that is plain,
//! Hadamard-rotated, or carries an unmerged LoRA adapter. Holds the
//! single-projection forward, the accessors, and parameter enumeration.

use crate::error::Result;
use crate::nn::hadamard::HadamardRotation;
use crate::nn::linear::maybe_quant_linear::MaybeQuantLinear;
use crate::nn::linear::rotated_linear::RotatedLinear;
use crate::nn::lora::LoraLinear;
use crate::nn::module::Module;
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{BinaryOps, FwhtOps, MatmulOps, ScalarOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::TensorId;

/// A linear projection that is plain, Hadamard-rotated, or LoRA-adapted.
///
/// `Lora` holds a dense or quantized base plus an unmerged adapter. It is
/// never rotated: [`Self::attach_adapter`] refuses a `Rotated` projection.
// `Rotated` and `Lora` are boxed for the same reason `MaybeLoraLinear::Lora`
// is: `Plain` is the common variant and stays inline, so the size gap is
// accepted.
#[allow(clippy::large_enum_variant)]
pub enum MaybeRotatedLinear<R: Runtime> {
    Plain(MaybeQuantLinear<R>),
    Rotated(Box<RotatedLinear<R>>),
    Lora(Box<LoraLinear<R>>),
}

impl<R: Runtime<DType = DType>> MaybeRotatedLinear<R> {
    /// Forward pass: plain base, rotate-then-base, or base plus the scaled
    /// low-rank path `scaling * (x A^T) B^T`.
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>
            + FwhtOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + DequantOps<R> + MatmulOps<R>,
    {
        match self {
            Self::Plain(base) => base.forward(client, input),
            Self::Rotated(rotated) => rotated.forward(client, input),
            Self::Lora(lora) => lora.forward(client, input),
        }
    }

    /// The underlying base linear layer, rotation and adapter aside.
    pub fn base(&self) -> &MaybeQuantLinear<R> {
        match self {
            Self::Plain(base) => base,
            Self::Rotated(rotated) => rotated.base(),
            Self::Lora(lora) => lora.base(),
        }
    }

    /// The Hadamard rotation attached to this layer, if any.
    pub fn rotation(&self) -> Option<&HadamardRotation<R>> {
        match self {
            Self::Rotated(rotated) => Some(rotated.rotation()),
            Self::Plain(_) | Self::Lora(_) => None,
        }
    }

    /// The attached adapter, if any.
    pub fn adapter(&self) -> Option<&LoraLinear<R>> {
        match self {
            Self::Lora(lora) => Some(lora),
            Self::Plain(_) | Self::Rotated(_) => None,
        }
    }

    /// The base weight, adapter aside. `None` for a quantized base.
    pub fn weight(&self) -> Option<&Var<R>> {
        self.base().weight()
    }

    /// The base bias, adapter aside. `None` for a quantized base.
    pub fn bias(&self) -> Option<&Var<R>> {
        self.base().bias()
    }

    /// `true` when a Hadamard rotation is attached.
    pub fn is_rotated(&self) -> bool {
        matches!(self, Self::Rotated(_))
    }

    /// `true` when a LoRA adapter is attached.
    pub fn is_adapted(&self) -> bool {
        matches!(self, Self::Lora(_))
    }

    /// All parameters with their stable autograd ids: the base's, plus the
    /// adapter factors when adapted.
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        match self {
            Self::Plain(base) => base.parameters(),
            Self::Rotated(rotated) => rotated.base().parameters(),
            Self::Lora(lora) => Module::parameters(lora.as_ref())
                .into_iter()
                .map(|var| (var.id(), var))
                .collect(),
        }
    }

    /// Parameters that require grad.
    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.parameters()
            .into_iter()
            .filter(|param| param.1.requires_grad())
            .collect()
    }

    /// Named parameters for checkpoint traversal. An adapted projection
    /// names its base `base.*` and its factors `lora_a`/`lora_b`, as
    /// [`LoraLinear`] does.
    pub fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        match self {
            Self::Plain(base) => base.named_parameters(),
            Self::Rotated(rotated) => rotated.base().named_parameters(),
            Self::Lora(lora) => lora.named_parameters(),
        }
    }

    /// Cheap duplicate over the same storage, keeping every `TensorId`:
    /// base weight, rotation signs, and adapter factors when adapted.
    pub fn alias(&self) -> Self {
        match self {
            Self::Plain(base) => Self::Plain(base.alias()),
            Self::Rotated(rotated) => Self::Rotated(Box::new(rotated.alias())),
            Self::Lora(lora) => Self::Lora(Box::new(lora.alias())),
        }
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

impl<R: Runtime<DType = DType>> From<LoraLinear<R>> for MaybeRotatedLinear<R> {
    fn from(lora: LoraLinear<R>) -> Self {
        Self::Lora(Box::new(lora))
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
    use crate::nn::linear::quant_linear::QuantLinear;
    use crate::quant::QuantFormat;
    use crate::quant::traits::QuantizeOps;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    const IN: usize = 32;
    const OUT: usize = 6;
    const RANK: usize = 3;
    const SCALING: f32 = 1.75;

    fn tensor(device: &CpuDevice, shape: &[usize], seed: f32) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 + 1.0) * seed).sin() * 0.2)
            .collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
    }

    fn dense(w: &Tensor<CpuRuntime>) -> MaybeQuantLinear<CpuRuntime> {
        MaybeQuantLinear::Standard(Linear::new(w.clone(), None, false))
    }

    /// `scaling * (x A^T) B^T` with plain numr ops.
    fn delta(
        client: &CpuClient,
        x: &Tensor<CpuRuntime>,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Tensor<CpuRuntime> {
        let a_t = a.t().unwrap().contiguous().unwrap();
        let b_t = b.t().unwrap().contiguous().unwrap();
        let mid = client.matmul(x, &a_t).unwrap();
        let low = client.matmul(&mid, &b_t).unwrap();
        client.mul_scalar(&low, f64::from(SCALING)).unwrap()
    }

    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < 1e-5, "element {i}: {g} vs {w}");
        }
    }

    #[test]
    fn plain_forwards_identically_to_bare_maybe_quant_linear() {
        let (client, device) = cpu_setup();
        let w = tensor(&device, &[OUT, IN], 0.02);
        let bare = dense(&w);
        let plain: MaybeRotatedLinear<CpuRuntime> = dense(&w).into();
        let x = Var::new(tensor(&device, &[1, IN], 0.1), false);

        let bare_out = bare.forward(&client, &x).unwrap();
        let plain_out = plain.forward(&client, &x).unwrap();
        assert_eq!(
            bare_out.tensor().to_vec::<f32>(),
            plain_out.tensor().to_vec::<f32>()
        );
        assert!(!plain.is_rotated());
        assert!(!plain.is_adapted());
    }

    /// The adapted forward equals `x (W + scaling B A)^T`, the merged weight
    /// built with plain numr ops.
    #[test]
    fn lora_forward_equals_merged_weight_forward() {
        let (client, device) = cpu_setup();
        let w = tensor(&device, &[OUT, IN], 0.13);
        let a = tensor(&device, &[RANK, IN], 0.29);
        let b = tensor(&device, &[OUT, RANK], 0.47);
        let mut layer: MaybeRotatedLinear<CpuRuntime> = dense(&w).into();
        layer.attach_adapter(a.clone(), b.clone(), SCALING).unwrap();
        assert!(layer.is_adapted());
        assert!(!layer.is_rotated());
        assert!(layer.rotation().is_none());

        let ba = client.matmul(&b, &a).unwrap();
        let merged = client
            .add(&w, &client.mul_scalar(&ba, f64::from(SCALING)).unwrap())
            .unwrap();
        let merged_t = merged.t().unwrap().contiguous().unwrap();
        let x = tensor(&device, &[4, IN], 0.61);
        let want: Vec<f32> = client.matmul(&x, &merged_t).unwrap().to_vec();

        let got = layer.forward(&client, &Var::new(x, false)).unwrap();
        assert!(
            !got.requires_grad(),
            "an attached adapter is frozen and must not record a graph"
        );
        assert_close(&got.tensor().to_vec::<f32>(), &want);
    }

    /// Over a block-quantized base the adapter adds its delta to the
    /// quantized base's own output.
    #[test]
    fn lora_forward_over_quantized_base_adds_delta_to_base_output() {
        let (client, device) = cpu_setup();
        let w = tensor(&device, &[OUT, IN], 0.17);
        let quant = client.quantize(&w, QuantFormat::Q8_0).unwrap();
        let base = MaybeQuantLinear::Quantized(QuantLinear::new(quant, None));
        let a = tensor(&device, &[RANK, IN], 0.31);
        let b = tensor(&device, &[OUT, RANK], 0.43);
        let x = tensor(&device, &[3, IN], 0.71);

        let base_out = base.forward(&client, &Var::new(x.clone(), false)).unwrap();
        let want: Vec<f32> = client
            .add(base_out.tensor(), &delta(&client, &x, &a, &b))
            .unwrap()
            .to_vec();

        let mut layer: MaybeRotatedLinear<CpuRuntime> = base.into();
        layer.attach_adapter(a, b, SCALING).unwrap();
        assert!(layer.weight().is_none());
        let got = layer.forward(&client, &Var::new(x, false)).unwrap();
        assert_close(&got.tensor().to_vec::<f32>(), &want);
    }

    /// An alias holds the same storage and ids; an adapter attached to it
    /// leaves the original plain.
    #[test]
    fn alias_shares_storage_and_adapts_alone() {
        let (_, device) = cpu_setup();
        let w = tensor(&device, &[OUT, IN], 0.19);
        let plain: MaybeRotatedLinear<CpuRuntime> = dense(&w).into();
        let mut alias = plain.alias();
        let (base, copy) = (plain.weight().unwrap(), alias.weight().unwrap());
        assert_eq!(copy.tensor().storage().ptr(), base.tensor().storage().ptr());
        assert_eq!(copy.id(), base.id());

        let a = tensor(&device, &[RANK, IN], 0.23);
        alias
            .attach_adapter(a, tensor(&device, &[OUT, RANK], 0.37), SCALING)
            .unwrap();
        assert!(alias.is_adapted() && !plain.is_adapted());
        let again = alias.alias();
        let factors = |l: &MaybeRotatedLinear<CpuRuntime>| {
            let lora = l.adapter().unwrap();
            (lora.lora_a().tensor().storage().ptr(), lora.lora_a().id())
        };
        assert_eq!(factors(&again), factors(&alias));
        assert_eq!(
            again.weight().unwrap().tensor().storage().ptr(),
            plain.weight().unwrap().tensor().storage().ptr()
        );
    }
}
