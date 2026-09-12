//! The activation contract a TCF weight carries out of its file.
//!
//! TCF Section 9 makes the contract part of the dispatch key: a kernel is
//! resolved on `weight_encoding`, `execution_role`, and the contract's typed
//! semantic fields (`input_representation`, `dot_accumulator`, `math_mode`,
//! and the rest `KernelContract::satisfies` checks), never on
//! `contract_digest` — that field is an integrity check, not a dispatch key.
//! The format defines no float fallback for a weight whose contract no
//! kernel satisfies. [`ActivationContract`] is the runtime form of
//! `ContractRecord`, attached to the weight it governs so a dispatch site
//! can check it without reopening the file.
//!
//! Only a TCF weight has one. A GGUF block format cannot express a contract,
//! so a GGUF weight carries none and dispatches unchecked — the absence is
//! "the format cannot say", never "the contract is satisfied".

use core::fmt;
use core::fmt::Write as _;

use crate::tcf::{
    ContractRecord, DotAccumulator, ExecutionRole, InputRepresentation, MathMode, OutputDtype,
    QuantAxis, RoundingMode, ScaleComputeDtype,
};

/// The activation contract one weight declares, with the provenance needed to
/// name it in an error.
///
/// `tensor` is provenance, never identity (Section 6): dispatch resolves on
/// `role`, the encoding, and the typed semantic fields below, and the name
/// only labels the failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationContract {
    /// Provenance name of the weight this contract governs.
    pub tensor: String,
    /// The execution role the weight is declared for. Section 8.6.1.
    pub role: ExecutionRole,
    /// `contract_digest`, BLAKE3-128. An integrity identity, not a dispatch
    /// key. Section 9.
    pub digest: [u8; 16],
    /// How the kernel must represent the activation it reads.
    pub input_representation: InputRepresentation,
    /// How the kernel must accumulate the dot product.
    pub dot_accumulator: DotAccumulator,
    /// The element type the kernel must produce.
    pub output_dtype: OutputDtype,
    /// Whether the kernel may reorder the sum. Section 9.1. Dropping this
    /// field is what let a `REASSOCIATION_FORBIDDEN` file run on a
    /// reassociating kernel with no refusal — never omit it again.
    pub math_mode: MathMode,
    /// Which axis activation quantization groups run along. Section 9.1.
    /// v1 defines only `LAST`, but a future variant must still be handled
    /// explicitly wherever a kernel states the axis it actually groups on.
    pub quant_axis: QuantAxis,
    /// How an activation value is rounded to its integer code. Section 9.1.
    /// v1 defines only `RN_EVEN`.
    pub rounding_mode: RoundingMode,
    /// The dtype the activation scale `d_a` is computed in. Section 9.1.
    /// v1 defines only `F32`.
    pub scale_compute_dtype: ScaleComputeDtype,
    /// Values per activation quantization group along the quantization axis.
    /// Section 9.2 defines it for `A8S32_DYNAMIC` alone; every other
    /// representation writes zero.
    pub quant_group: u16,
    /// Inclusive code range the activation quantizer must produce. Zero for a
    /// representation with no quantization grid.
    pub quant_range: (i16, i16),
}

impl ActivationContract {
    /// Read a decoded `ContractRecord` into the runtime form, labelled with
    /// the tensor's provenance name and the role it declares.
    #[must_use]
    pub fn from_record(tensor: &str, role: ExecutionRole, record: &ContractRecord) -> Self {
        Self {
            tensor: tensor.to_string(),
            role,
            digest: record.contract_digest,
            input_representation: record.input_representation,
            dot_accumulator: record.dot_accumulator,
            output_dtype: record.output_dtype,
            math_mode: record.math_mode,
            quant_axis: record.quant_axis,
            rounding_mode: record.rounding_mode,
            scale_compute_dtype: record.scale_compute_dtype,
            quant_group: record.quant_group,
            quant_range: (record.qmin, record.qmax),
        }
    }

    /// `contract_digest` as lowercase hex — the integrity identity, in the
    /// form an error message can be matched against.
    #[must_use]
    pub fn digest_hex(&self) -> String {
        let mut out = String::with_capacity(32);
        for byte in self.digest {
            // Writing into the buffer, not `format!` per byte: the digest is
            // rendered on every error path that names a contract.
            let _ = write!(out, "{byte:02x}");
        }
        out
    }
}

impl fmt::Display for ActivationContract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "input {}, dot {}, output {}, math {}, axis {}, rounding {}, scale dtype {}",
            input_representation_name(self.input_representation),
            dot_accumulator_name(self.dot_accumulator),
            output_dtype_name(self.output_dtype),
            math_mode_name(self.math_mode),
            quant_axis_name(self.quant_axis),
            rounding_mode_name(self.rounding_mode),
            scale_compute_dtype_name(self.scale_compute_dtype),
        )?;
        if self.quant_group != 0 {
            write!(f, ", group {}", self.quant_group)?;
        }
        if self.quant_range != (0, 0) {
            write!(
                f,
                ", range [{}, {}]",
                self.quant_range.0, self.quant_range.1
            )?;
        }
        write!(
            f,
            ", role {}, digest {}",
            role_name(self.role),
            self.digest_hex()
        )
    }
}

/// Wire-field spelling of `input_representation`. Section 9.1.
#[must_use]
pub fn input_representation_name(value: InputRepresentation) -> &'static str {
    match value {
        InputRepresentation::F32 => "F32",
        InputRepresentation::F16 => "F16",
        InputRepresentation::Bf16 => "BF16",
        InputRepresentation::A8S32Dynamic => "A8S32_DYNAMIC",
        InputRepresentation::GgmlReference => "GGML_REFERENCE",
    }
}

/// Wire-field spelling of `dot_accumulator`. Section 9.1.
#[must_use]
pub fn dot_accumulator_name(value: DotAccumulator) -> &'static str {
    match value {
        DotAccumulator::F32 => "F32",
        DotAccumulator::I32ThenF32Scale => "I32_THEN_F32_SCALE",
        DotAccumulator::GgmlReference => "GGML_REFERENCE",
    }
}

/// Wire-field spelling of `output_dtype`. Section 9.1.
#[must_use]
pub fn output_dtype_name(value: OutputDtype) -> &'static str {
    match value {
        OutputDtype::F32 => "F32",
        OutputDtype::F16 => "F16",
        OutputDtype::Bf16 => "BF16",
    }
}

/// Wire-field spelling of `math_mode`. Section 9.1.
#[must_use]
pub fn math_mode_name(value: MathMode) -> &'static str {
    match value {
        MathMode::ReassociationAllowed => "REASSOCIATION_ALLOWED",
        MathMode::ReassociationForbidden => "REASSOCIATION_FORBIDDEN",
    }
}

/// Wire-field spelling of `quant_axis`. Section 9.1.
#[must_use]
pub fn quant_axis_name(value: QuantAxis) -> &'static str {
    match value {
        QuantAxis::Last => "LAST",
    }
}

/// Wire-field spelling of `rounding_mode`. Section 9.1.
#[must_use]
pub fn rounding_mode_name(value: RoundingMode) -> &'static str {
    match value {
        RoundingMode::RnEven => "RN_EVEN",
    }
}

/// Wire-field spelling of `scale_compute_dtype`. Section 9.1.
#[must_use]
pub fn scale_compute_dtype_name(value: ScaleComputeDtype) -> &'static str {
    match value {
        ScaleComputeDtype::F32 => "F32",
    }
}

/// Wire-field spelling of `execution_role`. Section 8.6.1.
#[must_use]
pub fn role_name(value: ExecutionRole) -> &'static str {
    match value {
        ExecutionRole::Matmul => "MATMUL",
        ExecutionRole::Lookup => "LOOKUP",
        ExecutionRole::Conv1d => "CONV1D",
        ExecutionRole::Indexed => "INDEXED",
        ExecutionRole::StateUpdate => "STATE_UPDATE",
        ExecutionRole::Elementwise => "ELEMENTWISE",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tcf::{ContractFlags, MathMode, QuantAxis, RoundingMode, ScaleComputeDtype};

    fn f32_record() -> ContractRecord {
        ContractRecord {
            contract_id: 1,
            input_representation: InputRepresentation::F32,
            quant_group: 0,
            quant_axis: QuantAxis::Last,
            rounding_mode: RoundingMode::RnEven,
            qmin: 0,
            qmax: 0,
            scale_compute_dtype: ScaleComputeDtype::F32,
            dot_accumulator: DotAccumulator::F32,
            output_dtype: OutputDtype::F32,
            math_mode: MathMode::ReassociationAllowed,
            calibration_id: 0,
            flags: ContractFlags::NONE,
            contract_digest: [0xab; 16],
        }
    }

    #[test]
    fn every_dispatch_field_survives_the_record() {
        let declared = ActivationContract::from_record(
            "blk.0.attn_q.weight",
            ExecutionRole::Matmul,
            &f32_record(),
        );
        assert_eq!(declared.input_representation, InputRepresentation::F32);
        assert_eq!(declared.dot_accumulator, DotAccumulator::F32);
        assert_eq!(declared.output_dtype, OutputDtype::F32);
        assert_eq!(declared.role, ExecutionRole::Matmul);
        assert_eq!(declared.digest_hex(), "ab".repeat(16));
    }

    /// v1 defines exactly one variant for each of these three, so there is no
    /// mismatched value to round-trip — only that the record's value survives
    /// into the runtime form instead of being dropped.
    #[test]
    fn quant_axis_rounding_mode_and_scale_dtype_survive_the_record() {
        let declared = ActivationContract::from_record("w", ExecutionRole::Matmul, &f32_record());
        assert_eq!(declared.quant_axis, QuantAxis::Last);
        assert_eq!(declared.rounding_mode, RoundingMode::RnEven);
        assert_eq!(declared.scale_compute_dtype, ScaleComputeDtype::F32);
    }

    /// The defect this threading fixes: a file declaring
    /// `REASSOCIATION_FORBIDDEN` must keep that value in the runtime form, or
    /// dispatch has nothing left to refuse a reassociating kernel with.
    #[test]
    fn reassociation_forbidden_survives_the_record() {
        let mut record = f32_record();
        record.math_mode = MathMode::ReassociationForbidden;
        let declared = ActivationContract::from_record("w", ExecutionRole::Matmul, &record);
        assert_eq!(declared.math_mode, MathMode::ReassociationForbidden);
    }

    #[test]
    fn display_names_the_fields_a_reader_dispatches_on() {
        let declared = ActivationContract::from_record("w", ExecutionRole::Matmul, &f32_record());
        let text = declared.to_string();
        assert!(text.contains("input F32"), "{text}");
        assert!(text.contains("dot F32"), "{text}");
        assert!(text.contains("math REASSOCIATION_ALLOWED"), "{text}");
        assert!(text.contains("axis LAST"), "{text}");
        assert!(text.contains("rounding RN_EVEN"), "{text}");
        assert!(text.contains("scale dtype F32"), "{text}");
        assert!(text.contains("role MATMUL"), "{text}");
        assert!(text.contains(&declared.digest_hex()), "{text}");
    }
}
