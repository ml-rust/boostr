//! Carrying an activation contract on the weight it governs, and the check a
//! dispatch site runs before it launches.
//!
//! The contract lives on [`QuantTensor`] because that is what reaches a
//! kernel: the router sees a weight and a shape, never the file the weight
//! came from. It is optional, and the two states are NOT "checked" and
//! "unchecked by accident":
//!
//! - `Some(..)` — the weight came from a TCF file, which declares a contract
//!   for every tensor (Section 3). Dispatch is checked against it.
//! - `None` — the source format cannot express a contract. A GGUF block
//!   format has no field for one, so a GGUF weight carries none and
//!   dispatches exactly as it did before this check existed. Absence is "the
//!   format cannot say", never "any kernel will do".
//!
//! A block-encoded TCF weight declares `InputRepresentation::GgmlReference`:
//! the kernel family `ggml-quants.c` defines for its block type, whose
//! activation path differs per backend. Every block matmul entry point checks
//! that at entry with its backend's `BLOCK_CONTRACT`, so a file that pins a
//! more specific representation is refused rather than run with another.

use numr::dtype::DType;
use numr::runtime::Runtime;

use crate::error::{Error, Result};
use crate::quant::QuantTensor;

use super::declared::ActivationContract;
use super::kernel::KernelContract;
use super::mismatch::ActivationContractMismatchDetail;

impl<R: Runtime<DType = DType>> QuantTensor<R> {
    /// The activation contract this weight's source format declared, if it
    /// can declare one.
    pub fn activation_contract(&self) -> Option<&ActivationContract> {
        self.contract.as_ref()
    }

    /// Attach the contract a TCF file declares for this weight.
    ///
    /// Consuming and returning `self` keeps both constructors' signatures
    /// unchanged, so every existing caller — every GGUF caller included —
    /// builds a contract-free tensor exactly as before.
    #[must_use]
    pub fn with_activation_contract(mut self, contract: ActivationContract) -> Self {
        self.contract = Some(contract);
        self
    }

    /// Refuse this weight if `kernel` does not compute what its declared
    /// contract requires.
    ///
    /// Call this at the point a kernel is SELECTED, with that kernel's own
    /// declared [`KernelContract`]. A weight carrying no contract passes: see
    /// the module docs for why that is not a hole.
    ///
    /// # Errors
    ///
    /// [`Error::ActivationContractMismatch`] when the weight declares a
    /// contract the kernel does not satisfy. TCF Section 9 defines no float
    /// fallback for that case, so this is a refusal and the caller must not
    /// reroute to another kernel behind it.
    pub fn check_activation_contract(&self, kernel: &KernelContract) -> Result<()> {
        let Some(declared) = self.contract.as_ref() else {
            return Ok(());
        };
        if kernel.satisfies(declared) {
            return Ok(());
        }
        Err(Error::ActivationContractMismatch(Box::new(
            ActivationContractMismatchDetail {
                tensor: declared.tensor.clone(),
                encoding: self.format().name().to_string(),
                declared: declared.clone(),
                kernel: *kernel,
            },
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::QuantFormat;
    use crate::tcf::{
        ContractFlags, ContractRecord, DotAccumulator, ExecutionRole, InputRepresentation,
        MathMode, OutputDtype, QuantAxis, RoundingMode, ScaleComputeDtype,
    };
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn contract(
        tensor: &str,
        input: InputRepresentation,
        dot: DotAccumulator,
    ) -> ActivationContract {
        ActivationContract::from_record(
            tensor,
            ExecutionRole::Matmul,
            &ContractRecord {
                contract_id: 1,
                input_representation: input,
                quant_group: 0,
                quant_axis: QuantAxis::Last,
                rounding_mode: RoundingMode::RnEven,
                qmin: 0,
                qmax: 0,
                scale_compute_dtype: ScaleComputeDtype::F32,
                dot_accumulator: dot,
                output_dtype: OutputDtype::F32,
                math_mode: MathMode::ReassociationAllowed,
                calibration_id: 0,
                flags: ContractFlags::NONE,
                contract_digest: [0x11; 16],
            },
        )
    }

    fn exact_f32_contract(tensor: &str) -> ActivationContract {
        contract(tensor, InputRepresentation::F32, DotAccumulator::F32)
    }

    fn ggml_reference_contract(tensor: &str) -> ActivationContract {
        contract(
            tensor,
            InputRepresentation::GgmlReference,
            DotAccumulator::GgmlReference,
        )
    }

    fn q4_0_weight() -> QuantTensor<CpuRuntime> {
        let device = CpuDevice::new();
        QuantTensor::<CpuRuntime>::from_bytes(&[0u8; 18], QuantFormat::Q4_0, &[1, 32], &device)
            .expect("one block")
    }

    /// A GGUF weight has no contract to check, and the check must not invent
    /// one: its dispatch is unchanged by this feature.
    #[test]
    fn a_gguf_weight_carries_no_contract_and_passes_every_kernel() {
        let weight = q4_0_weight();
        assert!(weight.activation_contract().is_none());
        weight
            .check_activation_contract(&KernelContract::f32_activation("cpu_quant_matmul_f32"))
            .expect("no contract, no check");
        weight
            .check_activation_contract(&KernelContract::dynamic_int8_activation(
                "quant_gemv_q4_k_q8_1_mwr",
            ))
            .expect("no contract, no check");
        weight
            .check_activation_contract(&KernelContract::ggml_reference("cpu ggml block matmul"))
            .expect("no contract, no check");
    }

    /// The contract a block-encoded TCF file declares: the kernel family
    /// `ggml-quants.c` defines for the block type. Every backend's block
    /// kernel satisfies it, and nothing more specific does.
    #[test]
    fn a_ggml_reference_contract_admits_only_the_ggml_family() {
        let weight = q4_0_weight().with_activation_contract(ggml_reference_contract("layer.w"));
        weight
            .check_activation_contract(&KernelContract::ggml_reference("cuda ggml block matmul"))
            .expect("the family the file named");
        weight
            .check_activation_contract(&KernelContract::f32_activation("cpu_quant_matmul_f32"))
            .expect_err("an f32 kernel promises a representation the file never asked for");
        weight
            .check_activation_contract(&KernelContract::dynamic_int8_activation("tcf_mmq"))
            .expect_err("a group-32 kernel promises a representation the file never asked for");
    }

    /// The converse: a file that pins exact f32 activations is not served by
    /// the ggml family, whose activation path depends on the backend.
    #[test]
    fn the_ggml_family_is_refused_for_a_weight_declaring_f32_activations() {
        let weight = q4_0_weight().with_activation_contract(exact_f32_contract("layer.w"));
        weight
            .check_activation_contract(&KernelContract::ggml_reference("cpu ggml block matmul"))
            .expect_err("the family does not promise f32 activations");
    }

    #[test]
    fn an_f32_kernel_runs_a_weight_declaring_f32_activations() {
        let weight = q4_0_weight().with_activation_contract(exact_f32_contract("layer.w"));
        weight
            .check_activation_contract(&KernelContract::f32_activation("cpu_quant_matmul_f32"))
            .expect("the kernel computes what the contract declares");
    }

    /// The refusal this feature exists for, and the error names everything
    /// needed to act on it: the tensor, its encoding, what was declared, and
    /// what the selected kernel would have computed instead.
    #[test]
    fn an_int8_kernel_is_refused_for_a_weight_declaring_f32_activations() {
        let weight = q4_0_weight().with_activation_contract(exact_f32_contract("layer.w"));
        let kernel = KernelContract::dynamic_int8_activation("tcf_mmq_feat_major");
        let err = weight
            .check_activation_contract(&kernel)
            .expect_err("quantizing the activation violates the contract");

        match &err {
            Error::ActivationContractMismatch(detail) => {
                assert_eq!(detail.tensor, "layer.w");
                assert!(detail.encoding.contains("Q4"), "{}", detail.encoding);
                assert_eq!(
                    detail.declared.input_representation,
                    InputRepresentation::F32
                );
                assert_eq!(detail.kernel.kernel, "tcf_mmq_feat_major");
            }
            other => panic!("expected a contract mismatch, got {other:?}"),
        }

        let text = err.to_string();
        assert!(text.contains("E_ACTIVATION_CONTRACT_MISMATCH"), "{text}");
        assert!(text.contains("layer.w"), "{text}");
    }

    #[test]
    fn a_cloned_weight_keeps_its_contract() {
        let weight = q4_0_weight().with_activation_contract(exact_f32_contract("layer.w"));
        let cloned = weight.clone();
        assert_eq!(
            cloned.activation_contract().map(|c| c.digest),
            Some([0x11; 16])
        );
    }
}
