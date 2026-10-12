//! [`MaybeRotatedLinear::attach_adapter`]: turning a plain projection into
//! a LoRA-adapted one in place, and [`MaybeRotatedLinear::attach_factors`],
//! the same after casting the factors to the projection's compute dtype.

use super::layer::MaybeRotatedLinear;
use crate::error::{Error, Result};
use crate::nn::lora::{LoraFactors, LoraLinear};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> MaybeRotatedLinear<R> {
    /// Attach an unmerged adapter, in place: `lora_a` `[rank, in_features]`,
    /// `lora_b` `[out_features, rank]`, `scaling` as computed for this
    /// module. The factors are frozen (`requires_grad = false`). A refused
    /// attach leaves the projection unchanged.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when this projection is `Rotated` or already
    /// adapted, or when a check of [`LoraLinear::from_scaled_weights`]
    /// fails.
    pub fn attach_adapter(
        &mut self,
        lora_a: Tensor<R>,
        lora_b: Tensor<R>,
        scaling: f32,
    ) -> Result<()> {
        let base = match self {
            Self::Plain(base) => base.alias(),
            Self::Rotated(rotated) => {
                return Err(Error::ModelError {
                    reason: format!(
                        "attach_adapter: projection {:?} is Hadamard-rotated. A rotated \
                         checkpoint is re-laid-out for llama.cpp's activation-rotation contract \
                         (rotated weights, and inputs such as ssm_out's in grouped value-head \
                         order), so an adapter trained on the original checkpoint does not \
                         apply to it. Load the adapter onto an unrotated checkpoint",
                        rotated.base().shape()
                    ),
                });
            }
            Self::Lora(_) => {
                return Err(Error::ModelError {
                    reason: "attach_adapter: projection already carries a LoRA adapter; attach \
                             at most one adapter per projection"
                        .to_string(),
                });
            }
        };
        let lora = LoraLinear::from_scaled_weights(base, lora_a, lora_b, scaling, false)?;
        *self = Self::Lora(Box::new(lora));
        Ok(())
    }

    /// [`Self::attach_adapter`] after casting `factors` with
    /// [`LoraFactors::cast_for`]: to a dense base's weight dtype, else to
    /// `activation_dtype`, the dtype the projection's input arrives in.
    ///
    /// # Errors
    ///
    /// As [`Self::attach_adapter`]; numr errors from the cast propagate.
    pub fn attach_factors<C: TypeConversionOps<R>>(
        &mut self,
        client: &C,
        factors: LoraFactors<R>,
        activation_dtype: DType,
    ) -> Result<()> {
        let factors = factors.cast_for(client, self.base(), activation_dtype)?;
        self.attach_adapter(factors.lora_a, factors.lora_b, factors.scaling)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::hadamard::HadamardRotation;
    use crate::nn::linear::dense::Linear;
    use crate::nn::linear::maybe_quant_linear::MaybeQuantLinear;
    use crate::nn::linear::rotated_linear::RotatedLinear;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

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

    #[test]
    fn rotated_projection_refuses_an_adapter() {
        let (_, device) = cpu_setup();
        let signs: Vec<i8> = (0..IN).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
        let rotation =
            HadamardRotation::<CpuRuntime>::new(8, Some(signs.as_slice()), DType::F32, &device)
                .unwrap();
        let w = tensor(&device, &[OUT, IN], 0.05);
        let mut layer: MaybeRotatedLinear<CpuRuntime> =
            RotatedLinear::new(dense(&w), rotation).unwrap().into();
        let err = layer
            .attach_adapter(
                tensor(&device, &[RANK, IN], 0.3),
                tensor(&device, &[OUT, RANK], 0.4),
                SCALING,
            )
            .unwrap_err();
        assert!(err.to_string().contains("Hadamard-rotated"), "{err}");
        assert!(layer.is_rotated());
        assert!(!layer.is_adapted());
    }

    #[test]
    fn second_adapter_and_bad_shapes_are_refused() {
        let (_, device) = cpu_setup();
        let w = tensor(&device, &[OUT, IN], 0.05);
        let mut layer: MaybeRotatedLinear<CpuRuntime> = dense(&w).into();
        assert!(
            layer
                .attach_adapter(
                    tensor(&device, &[RANK, IN + 1], 0.3),
                    tensor(&device, &[OUT, RANK], 0.4),
                    SCALING,
                )
                .is_err()
        );
        assert!(
            !layer.is_adapted(),
            "a refused attach leaves the layer plain"
        );
        layer
            .attach_adapter(
                tensor(&device, &[RANK, IN], 0.3),
                tensor(&device, &[OUT, RANK], 0.4),
                SCALING,
            )
            .unwrap();
        assert!(layer.is_adapted());
        assert!(
            layer
                .attach_adapter(
                    tensor(&device, &[RANK, IN], 0.3),
                    tensor(&device, &[OUT, RANK], 0.4),
                    SCALING,
                )
                .is_err()
        );
    }

    /// F32 factors over an F64 dense base attach as F64.
    #[test]
    fn attach_factors_casts_to_the_dense_base_dtype() {
        let (client, device) = cpu_setup();
        let w = client
            .cast(&tensor(&device, &[OUT, IN], 0.05), DType::F64)
            .unwrap();
        let mut layer: MaybeRotatedLinear<CpuRuntime> = dense(&w).into();
        let factors = LoraFactors {
            lora_a: tensor(&device, &[RANK, IN], 0.3),
            lora_b: tensor(&device, &[OUT, RANK], 0.4),
            scaling: SCALING,
        };
        assert!(
            layer
                .attach_adapter(factors.lora_a.clone(), factors.lora_b.clone(), SCALING)
                .is_err(),
            "uncast F32 factors over an F64 base are refused"
        );
        layer.attach_factors(&client, factors, DType::F32).unwrap();
        let adapter = layer.adapter().expect("attached");
        assert_eq!(adapter.lora_a().tensor().dtype(), DType::F64);
        assert_eq!(adapter.lora_b().tensor().dtype(), DType::F64);
    }

    #[test]
    fn adapted_parameters_list_base_and_factors() {
        let (_, device) = cpu_setup();
        let w = tensor(&device, &[OUT, IN], 0.05);
        let mut layer: MaybeRotatedLinear<CpuRuntime> = dense(&w).into();
        layer
            .attach_adapter(
                tensor(&device, &[RANK, IN], 0.3),
                tensor(&device, &[OUT, RANK], 0.4),
                SCALING,
            )
            .unwrap();
        let names: Vec<String> = layer
            .named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert_eq!(names, ["base.weight", "lora_a", "lora_b"]);
        assert_eq!(layer.parameters().len(), 3);
        assert!(layer.trainable_parameters().is_empty());
        let adapter = layer.adapter().expect("adapted layer exposes its adapter");
        assert_eq!(adapter.scaling(), SCALING);
        assert_eq!(adapter.rank(), RANK);
    }
}
