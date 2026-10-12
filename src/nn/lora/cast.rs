//! [`LoraFactors`] cast to the dtype an adapter computes in.

use super::LoraFactors;
use crate::error::Result;
use crate::nn::MaybeQuantLinear;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> LoraFactors<R> {
    /// The factors cast to `dtype`. A factor already in `dtype` is kept
    /// without a copy.
    ///
    /// # Errors
    ///
    /// numr errors from the cast propagate.
    pub fn cast_to<C: TypeConversionOps<R>>(self, client: &C, dtype: DType) -> Result<Self> {
        let cast = |t: Tensor<R>| -> Result<Tensor<R>> {
            if t.dtype() == dtype {
                Ok(t)
            } else {
                Ok(client.cast(&t, dtype)?)
            }
        };
        Ok(Self {
            lora_a: cast(self.lora_a)?,
            lora_b: cast(self.lora_b)?,
            scaling: self.scaling,
        })
    }

    /// The factors cast to the dtype an adapter over `base` computes in: a
    /// dense base's weight dtype, else `activation_dtype`. A quantized
    /// base holds no float weight, and its adapter reads the activation
    /// the base reads.
    ///
    /// # Errors
    ///
    /// numr errors from the cast propagate.
    pub fn cast_for<C: TypeConversionOps<R>>(
        self,
        client: &C,
        base: &MaybeQuantLinear<R>,
        activation_dtype: DType,
    ) -> Result<Self> {
        let dtype = base
            .weight()
            .map_or(activation_dtype, |weight| weight.tensor().dtype());
        self.cast_to(client, dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn tensor(device: &CpuDevice, shape: &[usize], seed: f32) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 + 1.0) * seed).sin() * 0.2)
            .collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
    }

    /// A dense base sets the factor dtype; a quantized base takes the
    /// activation dtype.
    #[test]
    fn cast_for_follows_the_dense_base_else_the_activation() {
        use crate::quant::QuantFormat;
        use crate::quant::traits::QuantizeOps;
        let (client, device) = cpu_setup();
        let factors = || LoraFactors {
            lora_a: tensor(&device, &[2, 32], 0.2),
            lora_b: tensor(&device, &[3, 2], 0.3),
            scaling: 0.5,
        };
        let w = tensor(&device, &[3, 32], 0.1);
        let wide = client.cast(&w, DType::F64).unwrap();
        let dense = MaybeQuantLinear::Standard(Linear::new(wide, None, false));
        let cast = factors().cast_for(&client, &dense, DType::F32).unwrap();
        assert_eq!(cast.lora_a.dtype(), DType::F64);
        assert_eq!(cast.lora_b.dtype(), DType::F64);
        assert_eq!(cast.scaling, 0.5);
        let back: Vec<f64> = cast.lora_a.to_vec();
        let want: Vec<f32> = factors().lora_a.to_vec();
        for (b, w) in back.iter().zip(&want) {
            assert_eq!(*b, f64::from(*w));
        }

        let quant = client.quantize(&w, QuantFormat::Q8_0).unwrap();
        let quantized = MaybeQuantLinear::Quantized(crate::nn::QuantLinear::new(quant, None));
        let cast = factors().cast_for(&client, &quantized, DType::F64).unwrap();
        assert_eq!(cast.lora_a.dtype(), DType::F64);
        let kept = factors().cast_for(&client, &quantized, DType::F32).unwrap();
        assert_eq!(kept.lora_b.dtype(), DType::F32);
    }
}
