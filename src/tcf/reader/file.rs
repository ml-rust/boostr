//! The [`TcfFile`] type: [`TcfFile::open`] and the directory accessors.
//! See the module documentation for the Section 16 guarantee `open` makes
//! true by construction.

use std::collections::{HashMap, HashSet};

use crate::tcf::consts::{HEADER_BYTES, SCHEMA_ID};
use crate::tcf::error::TcfError;
use crate::tcf::record::field::{StringRef, expect_zero};
use crate::tcf::record::{
    CalibrationRecord, ContractRecord, Header, ModuleRecord, RelationRecord, TensorRecord,
    WorkloadProfileRecord,
};

use super::checks::{
    check_contract_digests, check_contract_record_digests, check_policy_digests,
    check_relation_digests, check_tensor,
};
use super::header_checks::{
    check_directory_digest, check_header_digest, check_header_flags, check_magic, check_major,
    check_required_feature_bits, check_required_feature_counts, check_two_level_feature_bit,
};
use super::sections::{bounds, decode_all, string_bytes, validate_sections};

/// A tensor's `data_offset`, relative to `Header.data_off`. Shared by
/// [`TcfFile::payload`] and `TcfFile::padding`, which both start from this
/// offset into `self.data`. Section 8.
pub(super) fn rel_data_offset(t: &TensorRecord, header: &Header) -> Result<u64, TcfError> {
    t.data_offset
        .checked_sub(header.data_off)
        .ok_or(bounds("tensor payload"))
}

/// A parsed, validated TCF v1 file, borrowing the caller's bytes.
///
/// The header and the whole directory are validated by [`TcfFile::open`].
/// Payload bytes are read only by [`TcfFile::payload`] and
/// [`TcfFile::verify_tensor`] — see the module documentation for the
/// Section 16 guarantee this structure makes true by construction.
///
/// Nothing here owns file bytes. The decoded records are fixed-size values
/// copied out of the directory at their spec offsets (Section 4 forbids
/// reinterpreting mapped bytes as a struct).
#[derive(Debug, Clone)]
pub struct TcfFile<'a> {
    pub(super) header: Header,
    /// Bytes `[0, data_off)`: everything `open` is allowed to read. Section 4.1.
    pub(super) directory: &'a [u8],
    /// Bytes `[data_off, file_len)`: tensor payloads, untouched by `open`.
    pub(super) data: &'a [u8],
    modules: Vec<ModuleRecord>,
    tensors: Vec<TensorRecord>,
    contracts: Vec<ContractRecord>,
    calibrations: Vec<CalibrationRecord>,
    relations: Vec<RelationRecord>,
    workload_profiles: Vec<WorkloadProfileRecord>,
}

impl<'a> TcfFile<'a> {
    /// Parse and validate the header and the directory. Section 4, Section 5.
    ///
    /// Validation runs in the Section 4 order, failing at the first problem,
    /// and completes every range check before any allocation sized from a
    /// file-supplied number:
    ///
    /// 1. length, magic, `major`,
    /// 2. unknown and mandated `required_features` bits,
    /// 3. header `flags` and the reserved header range,
    /// 4. `header_digest`,
    /// 5. every section range: `count * record_size` overflow, file bounds,
    ///    mutual overlap, 64-byte alignment,
    /// 6. `directory_digest`,
    /// 7. every record array, then the three derived record digests
    ///    (`policy_digest`, `contract_digest`, `relation_digest`), then the
    ///    cross-record invariants.
    ///
    /// No payload page is read. Section 16.
    ///
    /// # Errors
    /// - [`TcfError::SectionBounds`]: fewer than 192 bytes, a slice shorter
    ///   than `file_len`, `header_bytes != 192`, a section range that
    ///   overflows, overlaps another, or falls outside `[192, data_off)`.
    ///   Also a dangling record reference: a `module_id` (Section 7) or an
    ///   `activation_contract_id` (Section 3) naming no record in the file.
    /// - [`TcfError::BadMagic`], [`TcfError::UnsupportedMajor`].
    /// - [`TcfError::UnknownRequiredFeature`]: an unknown bit is set, or a
    ///   bit Section 5.2 mandates disagrees with the record counts.
    /// - [`TcfError::NonzeroReserved`]: header `flags` bit 0 clear, an
    ///   unknown flag bit, or a non-zero reserved byte.
    /// - [`TcfError::BadSchemaId`]: `schema_id` is not `1`.
    /// - [`TcfError::UnknownEnumValue`]: a record field carries a value v1
    ///   does not define.
    /// - [`TcfError::HeaderDigestMismatch`],
    ///   [`TcfError::DirectoryDigestMismatch`].
    /// - [`TcfError::MisalignedSection`]: a section start, `data_off`, or a
    ///   tensor's `data_offset` is not a multiple of 64.
    /// - [`TcfError::ContractDigestMismatch`],
    ///   [`TcfError::PolicyDigestMismatch`],
    ///   [`TcfError::RelationDigestMismatch`]: a recomputed record digest
    ///   differs from the stored one (Section 7, Section 9, Section 11).
    /// - [`TcfError::MissingFallbackReason`],
    ///   [`TcfError::MissingWorkloadProfile`],
    ///   [`TcfError::ResidentBytesViolation`],
    ///   [`TcfError::InvalidQuantShape`], plus every error a record's own
    ///   `decode` raises.
    pub fn open(bytes: &'a [u8]) -> Result<Self, TcfError> {
        let head = bytes
            .get(..HEADER_BYTES as usize)
            .ok_or(TcfError::SectionBounds { section: "Header" })?;

        // 1. Identity, before anything else is believed.
        check_magic(head)?;
        check_major(head)?;

        // 2. Capability claims, before any other flag field (Section 8.1.5:
        //    an unknown bit here is a capability claim, not a malformed byte).
        check_required_feature_bits(head)?;

        // 3. Header flags and the reserved tail.
        check_header_flags(head)?;
        expect_zero(head, 184, 8, "Header.reserved@184")?;

        let header = Header::decode(head)?;
        if header.header_bytes != HEADER_BYTES {
            return Err(TcfError::SectionBounds {
                section: "Header.header_bytes",
            });
        }
        if header.schema_id != SCHEMA_ID {
            return Err(TcfError::BadSchemaId {
                schema_id: header.schema_id,
            });
        }
        check_required_feature_counts(&header)?;

        // 4. The header's own digest.
        check_header_digest(head, &header)?;

        // 5. Every section range, before a single allocation sized from a
        //    file-supplied count (Section 4).
        let available = u64::try_from(bytes.len()).map_err(|_| TcfError::SectionBounds {
            section: "file_len",
        })?;
        let sections = validate_sections(&header, available)?;

        let data_off = usize::try_from(header.data_off).map_err(|_| TcfError::SectionBounds {
            section: "data_off",
        })?;
        let file_len = usize::try_from(header.file_len).map_err(|_| TcfError::SectionBounds {
            section: "file_len",
        })?;
        // The one split. Everything below reads `directory`; `data` is
        // reachable only from `payload` and `verify_tensor` (Section 16).
        let directory = bytes.get(..data_off).ok_or(TcfError::SectionBounds {
            section: "data_off",
        })?;
        let data = bytes
            .get(data_off..file_len)
            .ok_or(TcfError::SectionBounds {
                section: "file_len",
            })?;

        // 6. The directory's digest, over `[192, data_off)` exactly.
        check_directory_digest(directory, &header)?;

        // 7. Record arrays. Every range above is proven, so a count-sized
        //    allocation is now bounded by the file's real length.
        let modules: Vec<ModuleRecord> = decode_all(directory, &sections, 0, header.module_count)?;
        let tensors: Vec<TensorRecord> = decode_all(directory, &sections, 1, header.tensor_count)?;
        let contracts: Vec<ContractRecord> =
            decode_all(directory, &sections, 2, header.contract_count)?;
        let calibrations: Vec<CalibrationRecord> =
            decode_all(directory, &sections, 3, header.calibration_count)?;
        let relations: Vec<RelationRecord> =
            decode_all(directory, &sections, 4, header.relation_count)?;
        let workload_profiles: Vec<WorkloadProfileRecord> =
            decode_all(directory, &sections, 5, header.workload_count)?;

        // Section 9: all three derived record digests are recomputed here
        // and a mismatch is rejected. Every one of these records is
        // directory, never payload, so this reads no page Section 16
        // protects.
        check_policy_digests(directory, &sections, &header, &modules)?;
        check_contract_record_digests(directory, &sections, &contracts)?;
        check_relation_digests(directory, &sections, &relations)?;
        check_contract_digests(&contracts)?;
        // Built once so per-tensor lookup is O(1) rather than a linear scan
        // of `modules`/`workload_profiles` for every tensor. `entry` keeps
        // the first record for a repeated id, matching what `.find()` would
        // have returned.
        let mut modules_by_id: HashMap<u32, &ModuleRecord> = HashMap::with_capacity(modules.len());
        for m in &modules {
            modules_by_id.entry(m.module_id).or_insert(m);
        }
        let workload_ids: HashSet<u32> = workload_profiles.iter().map(|w| w.workload_id).collect();
        let contract_ids: HashSet<u32> = contracts.iter().map(|c| c.contract_id).collect();
        for tensor in &tensors {
            check_tensor(
                tensor,
                &header,
                &modules_by_id,
                &workload_ids,
                &contract_ids,
            )?;
        }
        check_two_level_feature_bit(&header)?;

        Ok(Self {
            header,
            directory,
            data,
            modules,
            tensors,
            contracts,
            calibrations,
            relations,
            workload_profiles,
        })
    }

    /// The validated file header. Section 5.
    #[must_use]
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Every `ModuleRecord`, in file order. Section 7.
    #[must_use]
    pub fn modules(&self) -> &[ModuleRecord] {
        &self.modules
    }

    /// Every `TensorRecord`, in file order. Section 8.
    #[must_use]
    pub fn tensors(&self) -> &[TensorRecord] {
        &self.tensors
    }

    /// Every `ContractRecord`, in file order. Section 9.
    #[must_use]
    pub fn contracts(&self) -> &[ContractRecord] {
        &self.contracts
    }

    /// Every `CalibrationRecord`, in file order. Section 10.
    #[must_use]
    pub fn calibrations(&self) -> &[CalibrationRecord] {
        &self.calibrations
    }

    /// Every `RelationRecord`, in file order. Section 11.
    #[must_use]
    pub fn relations(&self) -> &[RelationRecord] {
        &self.relations
    }

    /// Every `WorkloadProfileRecord`, in file order. Section 10.5.
    #[must_use]
    pub fn workload_profiles(&self) -> &[WorkloadProfileRecord] {
        &self.workload_profiles
    }

    /// Resolve a [`StringRef`] against the string table. Section 6.
    ///
    /// `r.off` is relative to `Header.string_off`. Names are provenance,
    /// never identity: nothing in dispatch depends on this call succeeding
    /// for a tensor to be usable.
    ///
    /// # Errors
    /// - [`TcfError::SectionBounds`]: the pair runs past `string_len`.
    /// - [`TcfError::InvalidUtf8Name`]: the bytes are not UTF-8 (Section 6).
    pub fn string(&self, r: StringRef) -> Result<&'a str, TcfError> {
        let raw = string_bytes(self.directory, &self.header, r)?;
        core::str::from_utf8(raw).map_err(|_| TcfError::InvalidUtf8Name {
            off: r.off(),
            len: r.len(),
        })
    }

    /// The `ContractRecord` a tensor's `activation_contract_id` names.
    /// Section 3, Section 9.
    ///
    /// A kernel is selected on `encoding`, `execution_role`, and the
    /// contract's semantic fields (Section 8.6.1, Section 9.1); the record's
    /// `contract_digest` is an integrity identity, not part of that key.
    /// Which kernels exist is no concern of this crate — the file states
    /// what the tensor was produced for, and nothing here decides what can
    /// run it.
    ///
    /// A repeated `contract_id` resolves to the first record in file order,
    /// which is what [`TcfFile::open`] has already proven carries the same
    /// digest as every later one (Section 17).
    ///
    /// # Errors
    /// [`TcfError::SectionBounds`] if `activation_contract_id` names no
    /// `ContractRecord` in this file. [`TcfFile::open`] has already proven
    /// that cannot happen for a tensor of this file; the check stands so a
    /// record from elsewhere cannot resolve to nothing unnoticed.
    pub fn contract(&self, t: &TensorRecord) -> Result<&ContractRecord, TcfError> {
        self.contracts
            .iter()
            .find(|c| c.contract_id == t.activation_contract_id)
            .ok_or(bounds("TensorRecord.activation_contract_id"))
    }

    /// The payload bytes of one tensor: exactly `logical_payload_bytes`,
    /// excluding the trailing alignment padding `physical_span_bytes`
    /// counts. Section 8, Section 14.4, Section 15.1.
    ///
    /// This is the first call that reads a payload page. Section 16.
    ///
    /// # Errors
    /// [`TcfError::SectionBounds`] if `data_offset` precedes `data_off` or
    /// the span runs past `file_len`. [`TcfFile::open`] has already proven
    /// neither happens for a tensor of this file; the check stands so a
    /// record from elsewhere cannot read out of bounds.
    pub fn payload(&self, t: &TensorRecord) -> Result<&'a [u8], TcfError> {
        let rel = rel_data_offset(t, &self.header)?;
        let start = usize::try_from(rel).map_err(|_| bounds("tensor payload"))?;
        let len = usize::try_from(t.logical_payload_bytes).map_err(|_| bounds("tensor payload"))?;
        let end = start.checked_add(len).ok_or(bounds("tensor payload"))?;
        self.data.get(start..end).ok_or(bounds("tensor payload"))
    }
}
