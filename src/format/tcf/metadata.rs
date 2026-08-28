//! Owned copies of a TCF file's directory metadata.
//!
//! [`super::loader::TcfLoader`] decodes the directory once and keeps these
//! values, so a caller answers "why is this tensor at this precision" without
//! reopening the file.

use tcf_core::{
    Encoding, FallbackReason, Header, ModuleRecord, ModuleRole, NativeEncoding, PolicyFlags,
    ResidencyClass, Role, TensorFlags, TensorRecord,
};

/// The spec's name for an encoding identifier. Section 12.
///
/// An identifier outside the v1 registry never reaches this function: the
/// record decoder rejects it first. The catch-all arm exists because
/// `Encoding` is `#[non_exhaustive]`.
pub fn encoding_name(encoding: Encoding) -> String {
    match encoding {
        Encoding::Native(NativeEncoding::Q4S32T64) => "Q4S32_T64".to_string(),
        Encoding::Native(NativeEncoding::Q4AS32T64) => "Q4AS32_T64".to_string(),
        Encoding::Native(NativeEncoding::Q4AS64T64) => "Q4AS64_T64".to_string(),
        Encoding::Native(NativeEncoding::Q6S32T64) => "Q6S32_T64".to_string(),
        Encoding::Native(NativeEncoding::Q8S32T64) => "Q8S32_T64".to_string(),
        Encoding::Raw(raw) => format!("{raw:?}").to_uppercase(),
        other => format!("0x{:04x}", other.to_u16()),
    }
}

/// The header fields a caller plans against. Section 5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TcfHeaderInfo {
    /// Format major version. `1` in v1.
    pub major: u16,
    /// Format minor version.
    pub minor: u16,
    /// Layout schema identifier.
    pub schema_id: u32,
    /// Number of `TensorRecord`s.
    pub tensor_count: u32,
    /// Number of `ModuleRecord`s.
    pub module_count: u32,
    /// Number of `ContractRecord`s.
    pub contract_count: u32,
    /// Number of `CalibrationRecord`s.
    pub calibration_count: u32,
    /// Number of `RelationRecord`s.
    pub relation_count: u32,
    /// Number of `WorkloadProfileRecord`s.
    pub workload_count: u32,
    /// First byte of the tensor data section. Section 4.1.
    pub data_off: u64,
    /// Total file length in bytes.
    pub file_len: u64,
}

impl From<&Header> for TcfHeaderInfo {
    fn from(h: &Header) -> Self {
        Self {
            major: h.major,
            minor: h.minor,
            schema_id: h.schema_id,
            tensor_count: h.tensor_count,
            module_count: h.module_count,
            contract_count: h.contract_count,
            calibration_count: h.calibration_count,
            relation_count: h.relation_count,
            workload_count: h.workload_count,
            data_off: h.data_off,
            file_len: h.file_len,
        }
    }
}

/// One module's policy, with its name resolved. Section 7.
#[derive(Debug, Clone)]
pub struct TcfModuleInfo {
    /// Module identifier, unique within the file.
    pub module_id: u32,
    /// Parent module, or `tcf_core::ROOT_PARENT_ID` for a root.
    pub parent_id: u32,
    /// Resolved UTF-8 name. Empty when the record carries none.
    pub name: String,
    /// What the module is.
    pub module_role: ModuleRole,
    /// Producer preference, highest first. Section 7.
    pub preferred_encoding: [Option<Encoding>; 4],
    /// Declared fallback encoding, if any.
    pub fallback_encoding: Option<Encoding>,
    /// Producer policy bits, e.g. `FORBID_REQUANT`.
    pub policy_flags: PolicyFlags,
    /// Default residency for the module's tensors.
    pub default_residency: ResidencyClass,
}

impl TcfModuleInfo {
    /// Build from a decoded record and its resolved name.
    pub fn new(record: &ModuleRecord, name: String) -> Self {
        Self {
            module_id: record.module_id,
            parent_id: record.parent_id,
            name,
            module_role: record.module_role,
            preferred_encoding: record.preferred_encoding,
            fallback_encoding: record.fallback_encoding,
            policy_flags: record.policy_flags,
            default_residency: record.default_residency,
        }
    }

    /// The highest-ranked preferred encoding, if the module declares one.
    pub fn top_preferred_encoding(&self) -> Option<Encoding> {
        self.preferred_encoding[0]
    }
}

/// One tensor's directory entry, with its name resolved. Section 8.
///
/// The full [`TensorRecord`] is kept so nothing the file states is lost.
#[derive(Debug, Clone)]
pub struct TcfTensorInfo {
    /// Resolved UTF-8 name. Empty when the record carries none. Section 6.
    pub name: String,
    /// The decoded record, verbatim.
    pub record: TensorRecord,
}

impl TcfTensorInfo {
    /// Build from a decoded record and its resolved name.
    pub fn new(record: TensorRecord, name: String) -> Self {
        Self { name, record }
    }

    /// This tensor's on-disk encoding. Section 12.
    pub fn encoding(&self) -> Encoding {
        self.record.encoding
    }

    /// Why this tensor is not at the module's preferred encoding. Section 8.6.
    ///
    /// [`FallbackReason::None`] means the tensor is at the preferred encoding.
    pub fn fallback_reason(&self) -> FallbackReason {
        self.record.fallback_reason
    }

    /// True when the producer recorded a reason for departing from preference.
    pub fn is_fallback(&self) -> bool {
        self.record.fallback_reason != FallbackReason::None
    }

    /// Honest bits per weight, scales and minima included. `None` for a raw
    /// encoding, whose width is fixed by its element size. Section 12.2.
    pub fn bits_per_weight(&self) -> Option<f64> {
        self.record.encoding.geometry().map(|g| g.bits_per_weight())
    }

    /// Row-major logical shape, trailing zero dimensions excluded. Section 8.
    ///
    /// A dimension larger than `usize` saturates. The load path then rejects
    /// the tensor, because the decoded value count cannot match the shape.
    pub fn shape(&self) -> Vec<usize> {
        self.record
            .shape()
            .iter()
            .map(|d| usize::try_from(*d).unwrap_or(usize::MAX))
            .collect()
    }

    /// Owning module identifier. Section 7.
    pub fn module_id(&self) -> u32 {
        self.record.module_id
    }

    /// Semantic role. Never a dispatch key. Section 8.7.
    pub fn role(&self) -> Role {
        self.record.role
    }

    /// Placement intent. Section 8.5.
    pub fn residency_class(&self) -> ResidencyClass {
        self.record.residency_class
    }

    /// Task sensitivity delta and its 95% interval, valid only when
    /// `SENSITIVITY_VALID` is set. Section 8.2.
    pub fn sensitivity(&self) -> Option<(f32, f32)> {
        self.record
            .flags
            .contains(TensorFlags::SENSITIVITY_VALID)
            .then_some((self.record.sensitivity_delta, self.record.sensitivity_ci95))
    }

    /// Stored payload bytes, alignment padding excluded. Section 8.0.1.
    pub fn payload_bytes(&self) -> u64 {
        self.record.logical_payload_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tcf_core::RawEncoding;

    #[test]
    fn encoding_names_match_section_12() {
        assert_eq!(
            encoding_name(Encoding::Native(NativeEncoding::Q4AS64T64)),
            "Q4AS64_T64"
        );
        assert_eq!(encoding_name(Encoding::Raw(RawEncoding::Bf16)), "BF16");
        assert_eq!(encoding_name(Encoding::Raw(RawEncoding::F8E4M3)), "F8E4M3");
    }
}
