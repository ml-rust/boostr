//! [`MaybeRotatedLinear::check_tracked`]: refuse a projection whose
//! forward gives its input no gradient, before a training forward runs.

use super::layer::MaybeRotatedLinear;
use crate::error::{Error, Result};
use crate::nn::linear::maybe_quant_linear::MaybeQuantLinear;
use numr::dtype::DType;
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> MaybeRotatedLinear<R> {
    /// Check that a training forward carries a gradient through this
    /// projection to its input.
    ///
    /// A Hadamard-rotated projection detaches its input. An AWQ/GPTQ
    /// decomposed base gives no input gradient. A dense or GGUF
    /// block-quantized base passes, plain or with a LoRA adapter.
    /// `label` names the projection in the error, for example
    /// `qwen35_attention.attn_q`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] with `arg = label` for a rotated
    /// projection or a decomposed base.
    pub fn check_tracked(&self, label: &'static str) -> Result<()> {
        if self.is_rotated() {
            return Err(Error::InvalidArgument {
                arg: label,
                reason: format!(
                    "{label} is Hadamard-rotated, and the rotation detaches its input. \
                     Load the model without the Hadamard attach to train it"
                ),
            });
        }
        match self.base() {
            MaybeQuantLinear::DecomposedQuant(_) => Err(Error::InvalidArgument {
                arg: label,
                reason: format!(
                    "{label} has a decomposed AWQ/GPTQ base, which gives no input gradient. \
                     Load it as a dense or GGUF block-quantized weight to train through it"
                ),
            }),
            MaybeQuantLinear::Standard(_) | MaybeQuantLinear::Quantized(_) => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::hadamard::HadamardRotation;
    use crate::nn::linear::dense::Linear;
    use crate::nn::linear::rotated_linear::RotatedLinear;
    use crate::quant::decomposed::{
        DecomposedQuantLinear, DecomposedQuantMethod, DecomposedQuantTensor,
    };
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    fn zeros(shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::zeros(shape, DType::F32, device).unwrap()
    }

    fn dense(device: &CpuDevice) -> MaybeQuantLinear<CpuRuntime> {
        MaybeQuantLinear::Standard(Linear::new(zeros(&[4, 8], device), None, false))
    }

    fn expect_refusal(linear: &MaybeRotatedLinear<CpuRuntime>, want: &str) {
        match linear.check_tracked("block.proj") {
            Err(Error::InvalidArgument { arg, reason }) => {
                assert_eq!(arg, "block.proj");
                assert!(reason.contains(want), "{reason}");
            }
            other => panic!("expected InvalidArgument, got {:?}", other.err()),
        }
    }

    #[test]
    fn plain_and_adapted_pass() {
        let (_, device) = cpu_setup();
        let mut linear: MaybeRotatedLinear<CpuRuntime> = dense(&device).into();
        assert!(linear.check_tracked("block.proj").is_ok());
        linear
            .attach_adapter(zeros(&[2, 8], &device), zeros(&[4, 2], &device), 1.0)
            .unwrap();
        assert!(linear.check_tracked("block.proj").is_ok());
    }

    #[test]
    fn rotated_is_refused() {
        let (_, device) = cpu_setup();
        let signs: Vec<i8> = (0..8).map(|i| if i % 2 == 0 { -1 } else { 1 }).collect();
        let rotation =
            HadamardRotation::<CpuRuntime>::new(8, Some(&signs), DType::F32, &device).unwrap();
        let linear: MaybeRotatedLinear<CpuRuntime> =
            RotatedLinear::new(dense(&device), rotation).unwrap().into();
        expect_refusal(&linear, "Hadamard-rotated");
    }

    #[test]
    fn decomposed_base_is_refused() {
        let (_, device) = cpu_setup();
        let weight = DecomposedQuantTensor::new(
            zeros(&[8, 1], &device),
            zeros(&[1, 4], &device),
            zeros(&[1, 4], &device),
            None,
            DecomposedQuantMethod::Awq { group_size: 128 },
            vec![4, 8],
        );
        let base =
            MaybeQuantLinear::DecomposedQuant(Box::new(DecomposedQuantLinear::new(weight, None)));
        let linear: MaybeRotatedLinear<CpuRuntime> = base.into();
        expect_refusal(&linear, "decomposed AWQ/GPTQ");
    }
}
