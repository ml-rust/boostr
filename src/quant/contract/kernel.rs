//! What a quantized-matmul kernel's arithmetic actually satisfies.
//!
//! A kernel is not a free choice among weights: it fixes how the activation
//! reaches the dot product and how the dot product accumulates. TCF Section 9
//! turns that into a dispatch key, so every kernel that can receive a
//! TCF-sourced weight declares a [`KernelContract`] beside itself, and the
//! router checks the weight's declared contract against it before launching.
//!
//! Declaring, not branching: adding a kernel means writing one constant next
//! to it and passing that constant to the check. Nothing here enumerates
//! kernels, so nothing here has to be edited when one is added.
//!
//! # No fallback
//!
//! Section 9 defines no float fallback for a weight whose contract no kernel
//! satisfies. A mismatch is [`crate::error::Error::ActivationContractMismatch`]
//! and the operation stops; it is never quietly rerouted to another kernel.

use core::fmt;

use tcf_core::{DotAccumulator, ExecutionRole, InputRepresentation, MathMode, OutputDtype};

use super::declared::{
    ActivationContract, dot_accumulator_name, input_representation_name, output_dtype_name,
    role_name,
};

/// Values per activation quantization group in the 8-bit dynamic activation
/// record every dp4a-class kernel consumes: 32 along K, which is the width of
/// one Q8_1 activation block and of one Q8_K sub-block alike.
pub const DYNAMIC_INT8_GROUP: u16 = 32;

/// Inclusive code range an activation quantizer that scales by `amax / 127`
/// produces.
pub const DYNAMIC_INT8_RANGE: (i16, i16) = (-127, 127);

/// The activation contract one kernel satisfies.
///
/// A field left `None` is one the kernel does not constrain, so a declared
/// contract carrying any value for it still matches. Every other field must
/// agree exactly: the kernel either computes what the contract says or it
/// does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelContract {
    /// The kernel or kernel family this describes, as it appears in source.
    pub kernel: &'static str,
    /// How the kernel represents the activation it reads.
    pub input_representation: InputRepresentation,
    /// How the kernel accumulates the dot product.
    pub dot_accumulator: DotAccumulator,
    /// The element type the kernel writes.
    pub output_dtype: OutputDtype,
    /// The execution role the kernel serves.
    pub role: ExecutionRole,
    /// Whether the kernel reorders the sum or fuses multiply-add. Never a
    /// declared permission — this states what the kernel's arithmetic
    /// actually does, checked against `math_mode` in [`Self::satisfies`].
    pub reassociates: bool,
    /// Values per activation quantization group, for a kernel that quantizes
    /// the activation. `None` for a kernel that does not.
    pub quant_group: Option<u16>,
    /// Inclusive code range the kernel's activation quantizer produces.
    /// `None` for a kernel that does not quantize the activation.
    pub quant_range: Option<(i16, i16)>,
}

impl KernelContract {
    /// A matmul kernel that reads the activation as f32 and accumulates in
    /// f32 — no activation quantization anywhere in the dot product.
    ///
    /// `reassociates` is always `true` here. Every kernel built on this
    /// constructor blocks or lane-splits the reduction (CPU: 8-lane AVX2 FMA
    /// plus a horizontal reduction, `cpu/kernels/tcf/matmul.rs`. CUDA GEMM:
    /// register-tile blocking with a split-K fixup. WGPU: unverified from
    /// source, so it takes the value that REFUSES rather than the one that
    /// permits), so none reproduces a fixed left-to-right sum.
    #[must_use]
    pub const fn f32_activation(kernel: &'static str) -> Self {
        Self {
            kernel,
            input_representation: InputRepresentation::F32,
            dot_accumulator: DotAccumulator::F32,
            output_dtype: OutputDtype::F32,
            role: ExecutionRole::Matmul,
            reassociates: true,
            quant_group: None,
            quant_range: None,
        }
    }

    /// A matmul kernel that quantizes the activation to 8-bit codes per
    /// group of `DYNAMIC_INT8_GROUP` values along K and runs the dot product
    /// on integers, rescaling to f32 afterwards.
    ///
    /// This is the contract of the whole dp4a and integer-MMA kernel family:
    /// the activation the caller handed in is NOT the activation the dot
    /// product sees.
    ///
    /// `reassociates` is always `true`. dp4a accumulates int32 partials
    /// across lanes and reduces them with a multi-warp tree; the MMA family
    /// reduces inside `mma.sync` and again across K tiles. Neither is a
    /// fixed-order sum.
    #[must_use]
    pub const fn dynamic_int8_activation(kernel: &'static str) -> Self {
        Self {
            kernel,
            input_representation: InputRepresentation::A8S32Dynamic,
            dot_accumulator: DotAccumulator::I32ThenF32Scale,
            output_dtype: OutputDtype::F32,
            role: ExecutionRole::Matmul,
            reassociates: true,
            quant_group: Some(DYNAMIC_INT8_GROUP),
            quant_range: Some(DYNAMIC_INT8_RANGE),
        }
    }

    /// Whether this kernel computes what `declared` requires.
    ///
    /// A kernel that reassociates never satisfies `ReassociationForbidden`:
    /// it reorders the sum regardless of what the file asked for. A kernel
    /// that does not reassociate satisfies either mode, since a fixed-order
    /// sum is a special case of "reassociation allowed".
    #[must_use]
    pub fn satisfies(&self, declared: &ActivationContract) -> bool {
        self.input_representation == declared.input_representation
            && self.dot_accumulator == declared.dot_accumulator
            && self.output_dtype == declared.output_dtype
            && self.role == declared.role
            && (!self.reassociates || declared.math_mode != MathMode::ReassociationForbidden)
            && self
                .quant_group
                .is_none_or(|group| group == declared.quant_group)
            && self
                .quant_range
                .is_none_or(|range| range == declared.quant_range)
    }
}

impl fmt::Display for KernelContract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: input {}, dot {}, output {}",
            self.kernel,
            input_representation_name(self.input_representation),
            dot_accumulator_name(self.dot_accumulator),
            output_dtype_name(self.output_dtype),
        )?;
        if let Some(group) = self.quant_group {
            write!(f, ", group {group}")?;
        }
        if let Some((qmin, qmax)) = self.quant_range {
            write!(f, ", range [{qmin}, {qmax}]")?;
        }
        write!(
            f,
            ", reassociates {}, role {}",
            self.reassociates,
            role_name(self.role)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tcf_core::{
        ContractFlags, ContractRecord, MathMode, QuantAxis, RoundingMode, ScaleComputeDtype,
    };

    fn record(
        input_representation: InputRepresentation,
        dot_accumulator: DotAccumulator,
        quant_group: u16,
        quant_range: (i16, i16),
    ) -> ContractRecord {
        ContractRecord {
            contract_id: 1,
            input_representation,
            quant_group,
            quant_axis: QuantAxis::Last,
            rounding_mode: RoundingMode::RnEven,
            qmin: quant_range.0,
            qmax: quant_range.1,
            scale_compute_dtype: ScaleComputeDtype::F32,
            dot_accumulator,
            output_dtype: OutputDtype::F32,
            math_mode: MathMode::ReassociationAllowed,
            kernel_semantics_id: 0,
            calibration_id: 0,
            flags: ContractFlags::NONE,
            contract_digest: [0u8; 16],
        }
    }

    fn declared(
        input_representation: InputRepresentation,
        dot_accumulator: DotAccumulator,
        quant_group: u16,
        quant_range: (i16, i16),
    ) -> ActivationContract {
        ActivationContract::from_record(
            "w",
            ExecutionRole::Matmul,
            &record(
                input_representation,
                dot_accumulator,
                quant_group,
                quant_range,
            ),
        )
    }

    #[test]
    fn an_f32_kernel_satisfies_an_f32_contract() {
        let kernel = KernelContract::f32_activation("tcf_gemv_f32");
        let contract = declared(InputRepresentation::F32, DotAccumulator::F32, 0, (0, 0));
        assert!(kernel.satisfies(&contract));
    }

    /// The whole point of the rule: a kernel that quantizes the activation
    /// does NOT satisfy a contract declaring exact f32 activations, however
    /// close its numbers come.
    #[test]
    fn an_int8_kernel_does_not_satisfy_an_f32_contract() {
        let kernel = KernelContract::dynamic_int8_activation("tcf_mmq_feat_major");
        let contract = declared(InputRepresentation::F32, DotAccumulator::F32, 0, (0, 0));
        assert!(!kernel.satisfies(&contract));
    }

    #[test]
    fn an_int8_kernel_satisfies_a_matching_dynamic_contract() {
        let kernel = KernelContract::dynamic_int8_activation("tcf_mmq_feat_major");
        let contract = declared(
            InputRepresentation::A8S32Dynamic,
            DotAccumulator::I32ThenF32Scale,
            DYNAMIC_INT8_GROUP,
            DYNAMIC_INT8_RANGE,
        );
        assert!(kernel.satisfies(&contract));
    }

    /// A group width the kernel does not compile for is a different grid, so
    /// the codes it would read are not the codes the contract describes.
    #[test]
    fn a_differing_group_width_is_a_mismatch() {
        let kernel = KernelContract::dynamic_int8_activation("tcf_mmq_feat_major");
        let contract = declared(
            InputRepresentation::A8S32Dynamic,
            DotAccumulator::I32ThenF32Scale,
            64,
            DYNAMIC_INT8_RANGE,
        );
        assert!(!kernel.satisfies(&contract));
    }

    /// A matmul kernel does not serve a weight declared for embedding lookup,
    /// even when the arithmetic matches: the role is the third component of
    /// the dispatch key.
    #[test]
    fn a_differing_role_is_a_mismatch() {
        let kernel = KernelContract::f32_activation("tcf_gemv_f32");
        let mut contract = declared(InputRepresentation::F32, DotAccumulator::F32, 0, (0, 0));
        contract.role = ExecutionRole::Lookup;
        assert!(!kernel.satisfies(&contract));
    }

    /// The defect this threading fixes: a kernel that reassociates ran
    /// unrefused against a `REASSOCIATION_FORBIDDEN` file before this check
    /// existed. It must refuse now, whatever else the contract agrees on.
    #[test]
    fn a_reassociating_kernel_does_not_satisfy_reassociation_forbidden() {
        let kernel = KernelContract::f32_activation("tcf_gemv_f32");
        let mut contract = declared(InputRepresentation::F32, DotAccumulator::F32, 0, (0, 0));
        contract.math_mode = MathMode::ReassociationForbidden;
        assert!(!kernel.satisfies(&contract));
    }

    #[test]
    fn a_reassociating_kernel_satisfies_reassociation_allowed() {
        let kernel = KernelContract::f32_activation("tcf_gemv_f32");
        let mut contract = declared(InputRepresentation::F32, DotAccumulator::F32, 0, (0, 0));
        contract.math_mode = MathMode::ReassociationAllowed;
        assert!(kernel.satisfies(&contract));
    }

    /// Every shipped kernel contract is declared `true`. A future kernel
    /// added with the wrong value fails here, not in the field.
    #[test]
    fn every_shipped_kernel_contract_declares_it_reassociates() {
        assert!(KernelContract::f32_activation("f32").reassociates);
        assert!(KernelContract::dynamic_int8_activation("int8").reassociates);
    }

    #[test]
    fn mismatch_error_text_names_the_math_mode() {
        let kernel = KernelContract::f32_activation("tcf_gemv_f32");
        let mut contract = declared(InputRepresentation::F32, DotAccumulator::F32, 0, (0, 0));
        contract.math_mode = MathMode::ReassociationForbidden;
        let declared_text = contract.to_string();
        let kernel_text = kernel.to_string();
        assert!(
            declared_text.contains("REASSOCIATION_FORBIDDEN"),
            "{declared_text}"
        );
        assert!(kernel_text.contains("reassociates true"), "{kernel_text}");
    }

    #[test]
    fn display_names_the_kernel_and_what_it_computes() {
        let text = KernelContract::dynamic_int8_activation("tcf_mmq_feat_major").to_string();
        assert!(text.contains("tcf_mmq_feat_major"), "{text}");
        assert!(text.contains("A8S32_DYNAMIC"), "{text}");
        assert!(text.contains("group 32"), "{text}");
    }
}
