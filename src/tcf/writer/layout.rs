//! Pass 1 and pass 3 of the writer: the layout arithmetic, the directory, and
//! the two header digests. FORMAT.md Section 4.1, Section 5,
//! Section 5.2, Section 5.3; MIGRATION.md Section 4.5.1.
//!
//! Pass 1 is `plan` then `emit_directory`: every `physical_span_bytes`,
//! `data_offset`, and `proof_rel_off`, then the header, the six record
//! arrays, and the string table. It reads no tensor data: `physical_span_bytes`
//! is a function of shape and encoding, and `data_offset` is a running
//! 64-aligned sum.
//!
//! Pass 3 is `finalize`: `header_digest`, then `directory_digest`.
//!
//! Payload bytes belong to the sibling `payload` module. Nothing here
//! touches one.

use crate::tcf::encoding::Encoding;
use std::io::{Seek, SeekFrom, Write};

use crate::tcf::consts::{
    CALIBRATION_RECORD_BYTES, CONTRACT_RECORD_BYTES, HEADER_BYTES, MAGIC, MAJOR,
    MODULE_RECORD_BYTES, PROOF_COUNT, RELATION_RECORD_BYTES, SCHEMA_ID, SECTION_ALIGN,
    TENSOR_RECORD_BYTES, WORKLOAD_PROFILE_RECORD_BYTES,
};
use crate::tcf::digest::hash_128;
use crate::tcf::enums::ProofFormat;
use crate::tcf::error::TcfError;
use crate::tcf::flags::{HeaderFlags, RequiredFeatures};
use crate::tcf::proof::PROOF_BYTES;
use crate::tcf::record::{HEADER_DIGEST_RANGE, Header, Record};

use super::{LAYOUT_BOUNDS, Payload, TcfWriter};

/// Every section offset and length, computed in pass 1 from headers alone.
/// Section 4.1, MIGRATION.md Section 4.5.1.
pub(crate) struct Layout {
    module_off: u64,
    pub(super) tensor_off: u64,
    contract_off: u64,
    calibration_off: u64,
    relation_off: u64,
    workload_off: u64,
    string_off: u64,
    string_len: u64,
    pub(super) proof_off: u64,
    proof_len: u64,
    /// `writer.rs` sizes the directory buffer from this; see `prepare`.
    pub(super) data_off: u64,
    file_len: u64,
}

/// Round `value` up to the next multiple of 64. Section 4.1.
fn align_up(value: u64) -> Result<u64, TcfError> {
    value
        .checked_next_multiple_of(SECTION_ALIGN)
        .ok_or(LAYOUT_BOUNDS)
}

/// `count * record_size` for a record array, rejecting the overflow Section 4
/// names explicitly.
fn array_bytes(count: usize, record_size: usize) -> Result<u64, TcfError> {
    let count = u64::try_from(count).map_err(|_| LAYOUT_BOUNDS)?;
    let size = u64::try_from(record_size).map_err(|_| LAYOUT_BOUNDS)?;
    count.checked_mul(size).ok_or(LAYOUT_BOUNDS)
}

/// A record count as the `u32` the header stores. Section 5.
pub(super) fn header_count(count: usize, section: &'static str) -> Result<u32, TcfError> {
    u32::try_from(count).map_err(|_| TcfError::SectionBounds { section })
}

/// Place a section of `len` bytes at the next 64-aligned offset at or after
/// `cursor`, advancing it. Section 4.1.
///
/// A zero-length section is **absent**: it takes offset `0` and length `0`,
/// and the cursor does not move. Section 4.1 requires a reader to ignore an
/// absent section's offset entirely, and `0` is the value producers write.
fn place(cursor: &mut u64, len: u64) -> Result<(u64, u64), TcfError> {
    if len == 0 {
        return Ok((0, 0));
    }
    let start = align_up(*cursor)?;
    *cursor = start.checked_add(len).ok_or(LAYOUT_BOUNDS)?;
    Ok((start, len))
}

/// A bounds-checked mutable window `[off, off + len)` into the file buffer.
fn window(buf: &mut [u8], off: u64, len: usize) -> Result<&mut [u8], TcfError> {
    let start = usize::try_from(off).map_err(|_| LAYOUT_BOUNDS)?;
    let end = start.checked_add(len).ok_or(LAYOUT_BOUNDS)?;
    buf.get_mut(start..end).ok_or(LAYOUT_BOUNDS)
}

/// Write `src` at absolute offset `off`.
pub(super) fn put(buf: &mut [u8], off: u64, src: &[u8]) -> Result<(), TcfError> {
    window(buf, off, src.len())?.copy_from_slice(src);
    Ok(())
}

/// A sink's `std::io::Error` as the `Clone + Eq + Hash` form `TcfError`
/// carries. Used at every seek and write call site.
fn io_error(e: std::io::Error) -> TcfError {
    TcfError::Io {
        kind: e.kind(),
        message: e.to_string(),
    }
}

/// Write `bytes` at absolute offset `off` in `sink`.
///
/// This is the streaming counterpart of [`put`]: the same absolute offsets
/// pass 1 computed, addressed by seeking instead of by indexing a buffer
/// that scales with the file. On a sink that is a real file, an `off` past
/// the current end leaves a hole; Section 14.4 padding is still written as
/// explicit zeros rather than left to the sink.
pub(super) fn seek_write<W: Write + Seek>(
    sink: &mut W,
    off: u64,
    bytes: &[u8],
) -> Result<(), TcfError> {
    sink.seek(SeekFrom::Start(off)).map_err(io_error)?;
    sink.write_all(bytes).map_err(io_error)
}

/// Encode record `index` of an array based at `base`, then decode it back
/// out of the buffer.
///
/// The read-back is the writer's own conformance check: a record the reader
/// would reject fails in the producer. It costs one decode per record and
/// touches no payload page.
pub(super) fn emit_record<R: Record>(
    buf: &mut [u8],
    base: u64,
    index: usize,
    record: &R,
) -> Result<(), TcfError> {
    let offset = array_bytes(index, R::SIZE)?
        .checked_add(base)
        .ok_or(LAYOUT_BOUNDS)?;
    let slot = window(buf, offset, R::SIZE)?;
    record.encode(slot)?;
    R::decode(slot)?;
    Ok(())
}

/// Pass 3: `header_digest`, then `directory_digest`. Section 5.3.
///
/// Order matters. `directory_digest` covers `[192, data_off)` and lands in
/// the header at `[160,176)`; `header_digest` covers `[0,192)` with
/// `[144,160)` treated as zero, so it must be computed after the directory
/// digest is already in place.
pub(super) fn finalize(buf: &mut [u8], layout: &Layout) -> Result<(), TcfError> {
    let end = usize::try_from(layout.data_off).map_err(|_| LAYOUT_BOUNDS)?;
    let body = buf.get(HEADER_BYTES as usize..end).ok_or(LAYOUT_BOUNDS)?;
    let directory = hash_128(body);
    put(buf, 160, directory.as_bytes())?;

    let mut image = [0u8; HEADER_BYTES as usize];
    let head = buf
        .get(..HEADER_BYTES as usize)
        .ok_or(TcfError::SectionBounds { section: "Header" })?;
    image.copy_from_slice(head);
    let zeroed = image
        .get_mut(HEADER_DIGEST_RANGE)
        .ok_or(TcfError::SectionBounds { section: "Header" })?;
    zeroed.fill(0);
    let header = hash_128(&image);
    put(buf, 144, header.as_bytes())?;

    let head = buf
        .get(..HEADER_BYTES as usize)
        .ok_or(TcfError::SectionBounds { section: "Header" })?;
    Header::decode(head)?;
    Ok(())
}

impl TcfWriter {
    /// Pass 1: every section offset, and every writer-owned size and proof
    /// field on every tensor. MIGRATION.md Section 4.5.1.
    ///
    /// Reads no tensor data. `physical_span_bytes` is a function of shape and
    /// encoding; `data_offset` is a running 64-aligned sum.
    pub(super) fn plan(&mut self) -> Result<Layout, TcfError> {
        let mut cursor = u64::from(HEADER_BYTES);
        let (module_off, _) = place(
            &mut cursor,
            array_bytes(self.modules.len(), MODULE_RECORD_BYTES)?,
        )?;
        let (tensor_off, _) = place(
            &mut cursor,
            array_bytes(self.tensors.len(), TENSOR_RECORD_BYTES)?,
        )?;
        let (contract_off, _) = place(
            &mut cursor,
            array_bytes(self.contracts.len(), CONTRACT_RECORD_BYTES)?,
        )?;
        let (calibration_off, _) = place(
            &mut cursor,
            array_bytes(self.calibrations.len(), CALIBRATION_RECORD_BYTES)?,
        )?;
        let (relation_off, _) = place(
            &mut cursor,
            array_bytes(self.relations.len(), RELATION_RECORD_BYTES)?,
        )?;
        let (workload_off, _) = place(
            &mut cursor,
            array_bytes(self.workloads.len(), WORKLOAD_PROFILE_RECORD_BYTES)?,
        )?;
        let (string_off, string_len) = place(
            &mut cursor,
            u64::try_from(self.strings.len()).map_err(|_| LAYOUT_BOUNDS)?,
        )?;

        let quantized = self
            .tensors
            .iter()
            .filter(|t| t.encoding.is_quantized())
            .count();
        let (proof_off, proof_len) = place(&mut cursor, array_bytes(quantized, PROOF_BYTES)?)?;

        let data_off = align_up(cursor)?;
        let mut data_cursor = data_off;
        let mut proof_cursor: u64 = 0;
        // `self.tensors` and `self.payloads` are pushed together by
        // `push_tensor`, so they stay index-aligned.
        for (tensor, payload) in self.tensors.iter_mut().zip(self.payloads.iter()) {
            // A registered tensor carries no payload yet, and its kind was
            // already fixed by `register_block_tensor` or
            // `register_raw_tensor`; the produced payload is checked against
            // the record in `finish_streaming`.
            match (tensor.encoding, payload) {
                (_, None)
                | (Encoding::Block(_), Some(Payload::Block { .. }))
                | (Encoding::Raw(_), Some(Payload::Raw(_))) => {}
                _ => {
                    return Err(TcfError::UnsupportedEncoding {
                        raw: tensor.encoding.to_u16(),
                    });
                }
            }
            // Section 8.0.1: determined from the shape and the encoding for
            // both kinds. A raw tensor's length is NOT the caller's byte
            // count — a caller's vector of another length is rejected in
            // pass 2, where the two are compared.
            let logical = tensor.determined_payload_bytes()?;
            let span = align_up(logical)?;

            tensor.data_offset = data_cursor;
            tensor.logical_payload_bytes = logical;
            tensor.physical_span_bytes = span;
            // Section 8.1: the canonical packed resident cost, and the
            // number a scheduler uses. Both equal the span in v1.
            tensor.resident_bytes = span;
            tensor.transfer_bytes = span;

            // Section 15.3: 64 values, format 1, for a block tensor; all
            // three fields zero for a raw one.
            if tensor.encoding.is_quantized() {
                tensor.proof_rel_off = proof_cursor;
                tensor.proof_count = PROOF_COUNT;
                tensor.proof_format = ProofFormat::DequantF16;
                proof_cursor = proof_cursor
                    .checked_add(u64::try_from(PROOF_BYTES).map_err(|_| LAYOUT_BOUNDS)?)
                    .ok_or(LAYOUT_BOUNDS)?;
            }

            data_cursor = data_cursor.checked_add(span).ok_or(LAYOUT_BOUNDS)?;
        }

        Ok(Layout {
            module_off,
            tensor_off,
            contract_off,
            calibration_off,
            relation_off,
            workload_off,
            string_off,
            string_len,
            proof_off,
            proof_len,
            data_off,
            file_len: data_cursor,
        })
    }

    /// Pass 1, second half: the header, all six record arrays, and the
    /// string table. Section 4.1.
    ///
    /// The header's two digest fields stay zero until pass 3.
    pub(super) fn emit_directory(&self, buf: &mut [u8], layout: &Layout) -> Result<(), TcfError> {
        let header = self.header(layout)?;
        let head = window(buf, 0, HEADER_BYTES as usize)?;
        header.encode(head)?;

        for (i, record) in self.modules.iter().enumerate() {
            emit_record(buf, layout.module_off, i, record)?;
        }
        for (i, record) in self.tensors.iter().enumerate() {
            emit_record(buf, layout.tensor_off, i, record)?;
        }
        for (i, record) in self.contracts.iter().enumerate() {
            emit_record(buf, layout.contract_off, i, record)?;
        }
        for (i, record) in self.calibrations.iter().enumerate() {
            emit_record(buf, layout.calibration_off, i, record)?;
        }
        for (i, record) in self.relations.iter().enumerate() {
            emit_record(buf, layout.relation_off, i, record)?;
        }
        for (i, record) in self.workloads.iter().enumerate() {
            emit_record(buf, layout.workload_off, i, record)?;
        }
        if layout.string_len != 0 {
            put(buf, layout.string_off, &self.strings)?;
        }
        Ok(())
    }

    /// The header, with both digests zero. Section 5, Section 5.2.
    fn header(&self, layout: &Layout) -> Result<Header, TcfError> {
        let relation_count = header_count(self.relations.len(), "RelationRecord[]")?;
        let workload_count = header_count(self.workloads.len(), "WorkloadProfileRecord[]")?;

        // Section 5.2: bits 0 through 3 in every conforming v1 file; bit 4
        // exactly when there are relations, bit 5 exactly when there are
        // workload profiles.
        let mut required = RequiredFeatures::ACTIVATION_CONTRACTS
            .union(RequiredFeatures::PLACEMENT_METADATA)
            .union(RequiredFeatures::SEMANTIC_DIGESTS)
            .union(RequiredFeatures::SOURCE_PROOFS);
        if relation_count > 0 {
            required = required.union(RequiredFeatures::RELATIONS);
        }
        if workload_count > 0 {
            required = required.union(RequiredFeatures::WORKLOAD_PROFILES);
        }
        // Section 5.2: bit 6 named the retired two-level tile encodings. No
        // encoding this writer emits uses it, so it is never set.

        Ok(Header {
            magic: MAGIC,
            major: MAJOR,
            minor: 0,
            header_bytes: HEADER_BYTES,
            schema_id: SCHEMA_ID,
            flags: HeaderFlags::LITTLE_ENDIAN,
            tensor_count: header_count(self.tensors.len(), "TensorRecord[]")?,
            module_count: header_count(self.modules.len(), "ModuleRecord[]")?,
            contract_count: header_count(self.contracts.len(), "ContractRecord[]")?,
            calibration_count: header_count(self.calibrations.len(), "CalibrationRecord[]")?,
            relation_count,
            workload_count,
            required_features: required,
            module_off: layout.module_off,
            tensor_off: layout.tensor_off,
            contract_off: layout.contract_off,
            calibration_off: layout.calibration_off,
            relation_off: layout.relation_off,
            string_off: layout.string_off,
            string_len: layout.string_len,
            proof_off: layout.proof_off,
            proof_len: layout.proof_len,
            data_off: layout.data_off,
            file_len: layout.file_len,
            header_digest: [0u8; 16],
            directory_digest: [0u8; 16],
            workload_off: layout.workload_off,
        })
    }
}
