//! Section layout: the Section 4.1 bounds and alignment checks, record
//! decoding, and string-table access. Everything here reads directory bytes.

use crate::tcf::consts::{
    CALIBRATION_RECORD_BYTES, CONTRACT_RECORD_BYTES, HEADER_BYTES, MODULE_RECORD_BYTES,
    RELATION_RECORD_BYTES, SECTION_ALIGN, TENSOR_RECORD_BYTES, WORKLOAD_PROFILE_RECORD_BYTES,
};
use crate::tcf::error::TcfError;
use crate::tcf::record::field::StringRef;
use crate::tcf::record::{Header, Record};

/// One validated top-level section range, in absolute file bytes. Section 4.1.
pub(super) struct SectionSpan {
    pub(super) name: &'static str,
    pub(super) start: u64,
    pub(super) len: u64,
}

impl SectionSpan {
    /// End offset, or `None` on overflow.
    pub(super) fn end(&self) -> Option<u64> {
        self.start.checked_add(self.len)
    }
}

/// `E_SECTION_BOUNDS` naming one section. Section 17.
pub(super) const fn bounds(section: &'static str) -> TcfError {
    TcfError::SectionBounds { section }
}

/// `count * record_size`, rejecting the overflow Section 4 names explicitly.
pub(super) fn record_span(count: u32, size: usize, name: &'static str) -> Result<u64, TcfError> {
    let size = u64::try_from(size).map_err(|_| TcfError::SectionBounds { section: name })?;
    u64::from(count)
        .checked_mul(size)
        .ok_or(TcfError::SectionBounds { section: name })
}

/// Every top-level section range, validated per Section 4: no arithmetic
/// overflow, no mutual overlap, 64-byte alignment, and containment in the
/// directory `[192, data_off)`.
///
/// Containment is the directory range rather than `file_len` because
/// Section 4.1 places every section before `Tensor data` and Section 5.3
/// digests exactly `[192, data_off)`. A section reaching past `data_off`
/// would also put a record on a payload page, which Section 16 forbids a
/// planner from touching.
///
/// A zero-length section is skipped: an absent array carries no meaningful
/// offset, and producers write `0` for one.
pub(super) fn validate_sections(
    header: &Header,
    available: u64,
) -> Result<Vec<SectionSpan>, TcfError> {
    if header.file_len < u64::from(HEADER_BYTES) || header.file_len > available {
        return Err(TcfError::SectionBounds {
            section: "file_len",
        });
    }
    if header.data_off < u64::from(HEADER_BYTES) || header.data_off > header.file_len {
        return Err(TcfError::SectionBounds {
            section: "data_off",
        });
    }
    if !header.data_off.is_multiple_of(SECTION_ALIGN) {
        return Err(TcfError::MisalignedSection {
            section: "data_off",
        });
    }

    // Order matches Section 4.1; `decode_all` indexes this list by position.
    let spans = [
        (
            "ModuleRecord[]",
            header.module_off,
            record_span(header.module_count, MODULE_RECORD_BYTES, "ModuleRecord[]")?,
        ),
        (
            "TensorRecord[]",
            header.tensor_off,
            record_span(header.tensor_count, TENSOR_RECORD_BYTES, "TensorRecord[]")?,
        ),
        (
            "ContractRecord[]",
            header.contract_off,
            record_span(
                header.contract_count,
                CONTRACT_RECORD_BYTES,
                "ContractRecord[]",
            )?,
        ),
        (
            "CalibrationRecord[]",
            header.calibration_off,
            record_span(
                header.calibration_count,
                CALIBRATION_RECORD_BYTES,
                "CalibrationRecord[]",
            )?,
        ),
        (
            "RelationRecord[]",
            header.relation_off,
            record_span(
                header.relation_count,
                RELATION_RECORD_BYTES,
                "RelationRecord[]",
            )?,
        ),
        (
            "WorkloadProfileRecord[]",
            header.workload_off,
            record_span(
                header.workload_count,
                WORKLOAD_PROFILE_RECORD_BYTES,
                "WorkloadProfileRecord[]",
            )?,
        ),
        ("String table", header.string_off, header.string_len),
        ("Proof section", header.proof_off, header.proof_len),
    ];

    let mut sections = Vec::with_capacity(spans.len());
    for (name, start, len) in spans {
        sections.push(SectionSpan { name, start, len });
    }

    for section in &sections {
        if section.len == 0 {
            continue;
        }
        if !section.start.is_multiple_of(SECTION_ALIGN) {
            return Err(TcfError::MisalignedSection {
                section: section.name,
            });
        }
        let end = section.end().ok_or(TcfError::SectionBounds {
            section: section.name,
        })?;
        if section.start < u64::from(HEADER_BYTES) || end > header.data_off {
            return Err(TcfError::SectionBounds {
                section: section.name,
            });
        }
    }

    for (i, a) in sections.iter().enumerate() {
        if a.len == 0 {
            continue;
        }
        let a_end = a.end().ok_or(TcfError::SectionBounds { section: a.name })?;
        for b in sections.iter().skip(i.saturating_add(1)) {
            if b.len == 0 {
                continue;
            }
            let b_end = b.end().ok_or(TcfError::SectionBounds { section: b.name })?;
            if a.start < b_end && b.start < a_end {
                return Err(TcfError::SectionBounds { section: b.name });
            }
        }
    }

    Ok(sections)
}

/// Decode one record array out of the directory. Section 4.1.
///
/// `index` selects the section by its Section 4.1 position in the list
/// [`validate_sections`] built, whose ranges are already proven to lie in
/// the directory — so the `count`-sized allocation here is bounded by the
/// file's real length (Section 4).
pub(super) fn decode_all<R: Record>(
    directory: &[u8],
    sections: &[SectionSpan],
    index: usize,
    count: u32,
) -> Result<Vec<R>, TcfError> {
    let section = sections.get(index).ok_or(TcfError::SectionBounds {
        section: "directory",
    })?;
    let count = usize::try_from(count).map_err(|_| bounds(section.name))?;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        out.push(R::decode(record_slice::<R>(
            directory, sections, index, i,
        )?)?);
    }
    Ok(out)
}

/// The exact stored bytes of record `i` of the array at section `index`.
/// Section 4.1.
///
/// The three derived digests are defined over a record's stored bytes, so
/// they are recomputed from this slice rather than from a re-encode of the
/// decoded value: what the file holds is what the digest must cover.
pub(super) fn record_slice<'d, R: Record>(
    directory: &'d [u8],
    sections: &[SectionSpan],
    index: usize,
    i: usize,
) -> Result<&'d [u8], TcfError> {
    let section = sections.get(index).ok_or(TcfError::SectionBounds {
        section: "directory",
    })?;
    let name = section.name;
    let base = usize::try_from(section.start).map_err(|_| bounds(name))?;
    let offset = i
        .checked_mul(R::SIZE)
        .and_then(|o| o.checked_add(base))
        .ok_or(bounds(name))?;
    let end = offset.checked_add(R::SIZE).ok_or(bounds(name))?;
    directory.get(offset..end).ok_or(bounds(name))
}

/// The string-table bytes a [`StringRef`] names, unvalidated. Section 6.
///
/// Used both by [`TcfFile::string`], which then checks UTF-8, and by
/// `policy_digest`, which hashes the bytes as they stand (Section 7).
pub(super) fn string_bytes<'d>(
    directory: &'d [u8],
    header: &Header,
    r: StringRef,
) -> Result<&'d [u8], TcfError> {
    let len = u64::from(r.len());
    let rel_end = r.off().checked_add(len).ok_or(bounds("string table"))?;
    if rel_end > header.string_len {
        return Err(bounds("string table"));
    }
    let start = header
        .string_off
        .checked_add(r.off())
        .ok_or(bounds("string table"))?;
    let end = start.checked_add(len).ok_or(bounds("string table"))?;
    let start = usize::try_from(start).map_err(|_| bounds("string table"))?;
    let end = usize::try_from(end).map_err(|_| bounds("string table"))?;
    directory.get(start..end).ok_or(bounds("string table"))
}
