//! `TcfWriter`: the TCF file writer. `FORMAT.md` Section 4, Section 4.1,
//! Section 5, Section 5.3, Section 8, Section 14.4, Section 15.
//!
//! # The three passes
//!
//! The property that makes the order work is that **the directory is computable from headers alone**:
//! `physical_span_bytes` is a function of shape and encoding, and
//! `data_offset` is a running 64-aligned sum. Neither needs tensor data.
//!
//! ```text
//! pass 1   `plan` + `emit_directory`
//!          every physical_span_bytes, data_offset, proof_rel_off
//!          header, all six record arrays, string table
//!
//! pass 2   `emit_payloads`, in directory order, one tensor at a time:
//!          payload_digest, semantic_digest and the 64 proof values
//!          payload written at its precomputed data_offset, zero-filled to
//!          physical_span_bytes (Section 14.4), then the payload dropped
//!
//! pass 3   `finalize`
//!          header_digest, then directory_digest (Section 5.3)
//! ```
//!
//! Pass 2 moves each payload out of the writer, writes it at its final
//! offset, and drops it before the next tensor is touched. The proof vector
//! each tensor produces is written straight into the proof section at its
//! final offset rather than buffered, which leaves the section complete
//! before pass 3 computes the digest covering it.
//!
//! # Memory: the two entry points differ, and only one is streaming
//!
//! Neither [`TcfWriter::finish`] nor [`TcfWriter::finish_into`] holds a
//! second copy of a payload, and neither is one-tensor-in-flight either:
//! [`TcfWriter::add_block_tensor`] and [`TcfWriter::add_raw_tensor`]
//! store their payload in the writer, so every tensor's payload is resident
//! at once by the time pass 2 starts. What pass 2 bounds is the *extra* cost
//! on top of that set, not the set itself.
//!
//! [`TcfWriter::finish_streaming`] is the path that holds one payload at a
//! time. A caller registers each tensor's record with
//! [`TcfWriter::register_block_tensor`] or
//! [`TcfWriter::register_raw_tensor`], supplying no payload, and passes a
//! callback the writer pulls from once per tensor in directory order. That
//! works because the directory is computable from headers alone: only the
//! per-tensor digests need data, and the directory is written last. Peak
//! memory is then the directory plus one tensor.
//!
//! All three write byte-identical files for the same tensors in the same
//! order.
//!
//! Every path buffers `[0, data_off)` — the header, the six record arrays,
//! the string table, and the proof section. That is a function of record
//! **count**, not of payload size. Payload bytes are written to the sink at
//! their precomputed absolute offsets and never enter a buffer that scales
//! with the file. [`TcfWriter::finish`] is [`TcfWriter::finish_into`] over
//! an in-memory sink, and returns the whole file.
//!
//! # What the writer owns
//!
//! A caller supplies policy — encoding, `fallback_reason`, residency,
//! telemetry — and never the layout arithmetic. The writer fills in
//! `data_offset`, `logical_payload_bytes`, `physical_span_bytes`,
//! `resident_bytes`, `transfer_bytes`, `semantic_digest`, `payload_digest`,
//! `proof_rel_off`, `proof_count`, and `proof_format` on every tensor. A
//! caller-set non-zero value in any of them is rejected by name rather than
//! silently overwritten: a producer that computed an offset itself and got
//! it wrong must hear about it.
//!
//! The three derived record digests are owned the same way: Section 9
//! requires a producer to compute `policy_digest` (Section 7),
//! `contract_digest` (Section 9), and `relation_digest` (Section 11), so
//! [`TcfWriter::finish`] computes each from the record's own encoded bytes
//! and rejects a caller-supplied non-zero value by name. `contract_digest`
//! is an integrity identity: a carried-over wrong one hides a corrupted
//! record and nothing downstream can tell.
//!
//! `logical_payload_bytes` is determined, never accepted, for a raw tensor
//! as well as a quantized one (Section 8.0.1): it is
//! `product(dims) * width` from the encoding's element size. A caller's
//! byte vector of a different length is rejected rather than stored.
//!
//! Every record the writer emits is decoded back out of the buffer it was
//! just written into. A record that the reader would reject — a bad rank, a
//! non-zero reserved byte, an undefined enum value — fails here, in the
//! producer, instead of in the consumer.
//!
//! # What the writer does not own
//!
//! Payload layout. A block payload is the GGML stream the producer's
//! quantizer emitted, stored as-is; no file in this module holds a bit
//! position, nibble index, or plane order.
//!
//! # Where each pass lives
//!
//! This file holds the [`TcfWriter`] struct, the record `add_*` and
//! [`TcfWriter::intern`] API, and the [`TcfWriter::finish`] entry points.
//! The tensor entry points are in `tensors`. Pass 1 and pass 3 are in
//! `layout`; pass 2 is in `payload`;
//! [`TcfWriter::finish_streaming`] is in [`crate::tcf::streaming`].

use std::collections::HashMap;
use std::io::{Seek, Write};

use crate::tcf::consts::{CONTRACT_RECORD_BYTES, MODULE_RECORD_BYTES, RELATION_RECORD_BYTES};
use crate::tcf::digest::{contract_digest, policy_digest, relation_digest};
use crate::tcf::error::TcfError;
use crate::tcf::record::field::StringRef;
use crate::tcf::record::{
    CalibrationRecord, ContractRecord, ModuleRecord, Record, RelationRecord, TensorRecord,
    WorkloadProfileRecord,
};

pub(crate) mod layout;
mod payload;
mod tensors;

pub use payload::Payload;

use layout::{Layout, finalize, header_count, seek_write};
use tensors::writer_owned;

/// A layout arithmetic step that would overflow `u64` or leave `usize`.
/// Section 4 requires the same rejection on the reading side.
pub(crate) const LAYOUT_BOUNDS: TcfError = TcfError::SectionBounds {
    section: "file layout",
};

/// The TCF file writer. Section 4.1.
///
/// A caller adds records and tensors in the order they should appear in the
/// file, then calls [`TcfWriter::finish`] once. See the module documentation
/// for the three-pass order and the division of ownership between caller and
/// writer.
pub struct TcfWriter {
    modules: Vec<ModuleRecord>,
    tensors: Vec<TensorRecord>,
    contracts: Vec<ContractRecord>,
    calibrations: Vec<CalibrationRecord>,
    relations: Vec<RelationRecord>,
    workloads: Vec<WorkloadProfileRecord>,
    /// One entry per tensor, in the same order as `tensors`. `None` is a
    /// tensor registered without a payload, which only
    /// [`TcfWriter::finish_streaming`] can complete.
    payloads: Vec<Option<Payload>>,
    /// The string table's bytes, in intern order. Section 6.
    strings: Vec<u8>,
    /// Name to its already-interned reference, so a repeated name costs no
    /// second copy. Section 6: names are provenance, so sharing one is free.
    interned: HashMap<String, StringRef>,
}

impl Default for TcfWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl TcfWriter {
    /// An empty writer. Section 4.1.
    #[must_use]
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            tensors: Vec::new(),
            contracts: Vec::new(),
            calibrations: Vec::new(),
            relations: Vec::new(),
            workloads: Vec::new(),
            payloads: Vec::new(),
            strings: Vec::new(),
            interned: HashMap::new(),
        }
    }

    /// Append a `ModuleRecord`, returning its index in the module array.
    /// Section 7.
    ///
    /// # Errors
    /// [`TcfError::SectionBounds`] if the array would exceed the `u32` count
    /// the header stores.
    pub fn add_module(&mut self, m: ModuleRecord) -> Result<u32, TcfError> {
        let index = header_count(self.modules.len(), "ModuleRecord[]")?;
        self.modules.push(m);
        Ok(index)
    }

    /// Append a `ContractRecord`, returning its index in the contract array.
    /// Section 9.
    ///
    /// # Errors
    /// [`TcfError::SectionBounds`] if the array would exceed the `u32` count
    /// the header stores.
    pub fn add_contract(&mut self, c: ContractRecord) -> Result<u32, TcfError> {
        let index = header_count(self.contracts.len(), "ContractRecord[]")?;
        self.contracts.push(c);
        Ok(index)
    }

    /// Append a `CalibrationRecord`, returning its index in the calibration
    /// array. Section 10.
    ///
    /// # Errors
    /// [`TcfError::SectionBounds`] if the array would exceed the `u32` count
    /// the header stores.
    pub fn add_calibration(&mut self, c: CalibrationRecord) -> Result<u32, TcfError> {
        let index = header_count(self.calibrations.len(), "CalibrationRecord[]")?;
        self.calibrations.push(c);
        Ok(index)
    }

    /// Append a `WorkloadProfileRecord`, returning its index in the workload
    /// array. Section 10.5. Its presence sets `required_features` bit 5
    /// (Section 5.2).
    ///
    /// # Errors
    /// [`TcfError::SectionBounds`] if the array would exceed the `u32` count
    /// the header stores.
    pub fn add_workload_profile(&mut self, w: WorkloadProfileRecord) -> Result<u32, TcfError> {
        let index = header_count(self.workloads.len(), "WorkloadProfileRecord[]")?;
        self.workloads.push(w);
        Ok(index)
    }

    /// Append a `RelationRecord`. Section 11. Its presence sets
    /// `required_features` bit 4 (Section 5.2).
    ///
    /// A relation has no index a later call needs, so this returns nothing.
    ///
    /// # Errors
    /// [`TcfError::SectionBounds`] if the array would exceed the `u32` count
    /// the header stores.
    pub fn add_relation(&mut self, r: RelationRecord) -> Result<(), TcfError> {
        header_count(self.relations.len(), "RelationRecord[]")?;
        self.relations.push(r);
        Ok(())
    }

    /// Intern a name into the string table, returning its [`StringRef`].
    /// Section 6.
    ///
    /// The same name interned twice yields the same reference and stores one
    /// copy. The empty name is `(0, 0)` and stores nothing, which is the
    /// value a record carrying no name holds.
    ///
    /// # Errors
    /// [`TcfError::SectionBounds`] if the table or the name exceeds the
    /// offsets the record fields can hold.
    pub fn intern(&mut self, name: &str) -> Result<StringRef, TcfError> {
        if name.is_empty() {
            return Ok(StringRef::new(0, 0));
        }
        if let Some(existing) = self.interned.get(name) {
            return Ok(*existing);
        }
        let off = u64::try_from(self.strings.len()).map_err(|_| LAYOUT_BOUNDS)?;
        let len = u32::try_from(name.len()).map_err(|_| LAYOUT_BOUNDS)?;
        self.strings.extend_from_slice(name.as_bytes());
        let reference = StringRef::new(off, len);
        self.interned.insert(name.to_owned(), reference);
        Ok(reference)
    }

    /// Emit the complete file as one buffer. Section 4.1,
    ///
    /// The whole file is held in memory, on top of the payloads the writer
    /// already holds. [`TcfWriter::finish_into`] drops the file copy by
    /// writing to a sink; [`TcfWriter::finish_streaming`] drops the payload
    /// set too, and is the one to use for model weights that do not fit.
    /// All three produce identical bytes.
    ///
    /// # Errors
    /// Every error the three passes raise: layout overflow
    /// ([`TcfError::SectionBounds`]), a record the reader would reject (its
    /// own decode error), a shape error from the
    /// proof vector, a payload whose length disagrees with Section 8.0.1
    /// reported as [`TcfError::InvalidQuantShape`], or
    /// [`TcfError::NonzeroReserved`] naming a caller-supplied
    /// `policy_digest`, `contract_digest`, or `relation_digest`. A tensor
    /// registered with [`TcfWriter::register_block_tensor`] or
    /// [`TcfWriter::register_raw_tensor`] and never finished with
    /// [`TcfWriter::finish_streaming`] is [`TcfError::PayloadMismatch`]. The
    /// in-memory sink this method uses raises no I/O error of its own.
    pub fn finish(self) -> Result<Vec<u8>, TcfError> {
        let mut cursor = std::io::Cursor::new(Vec::<u8>::new());
        self.finish_into(&mut cursor)?;
        Ok(cursor.into_inner())
    }

    /// Emit the complete file into `sink`. Section 4.1,
    ///
    /// The streaming form of [`TcfWriter::finish`], and the one a producer of
    /// multi-gigabyte weights uses: only the directory — the header, the six
    /// record arrays, the string table, and the proof section, that is
    /// `[0, data_off)` — is held in memory, and its size tracks record
    /// **count**, never payload bytes. Each tensor's payload goes straight to
    /// `sink` at the absolute `data_offset` pass 1 computed and is dropped
    /// before the next tensor is touched, so no buffer scales with the file.
    ///
    /// Every tensor's payload is still resident when this starts, because
    /// [`TcfWriter::add_block_tensor`] and [`TcfWriter::add_raw_tensor`]
    /// store it. [`TcfWriter::finish_streaming`] is the path that holds one
    /// payload at a time.
    ///
    /// The three produce byte-identical files.
    ///
    /// # Contract
    ///
    /// - **No flush.** This method never calls `flush`. A caller wrapping a
    ///   `BufWriter` owns flushing it, and owns the error that surfaces
    ///   there.
    /// - **Stream position is unspecified on return.** The last write is the
    ///   directory at offset `0`, not the end of the file. A caller that
    ///   needs a particular position seeks itself.
    /// - **Payload before directory.** Pass 2 writes at offsets at or after
    ///   `data_off` while nothing below it has been written yet, so on a real
    ///   file the first writes seek past the end and leave a hole. The single
    ///   final directory write fills it. That ordering is intended: the two
    ///   header digests are only known once every payload digest is.
    /// - **With no tensors** `file_len == data_off`, and that one directory
    ///   write is the whole file.
    ///
    /// # Errors
    /// Every error the three passes raise: layout overflow
    /// ([`TcfError::SectionBounds`]), a record the reader would reject (its
    /// own decode error), a shape error from the
    /// proof vector, a payload whose length disagrees with Section 8.0.1
    /// reported as [`TcfError::InvalidQuantShape`], or
    /// [`TcfError::NonzeroReserved`] naming a caller-supplied
    /// `policy_digest`, `contract_digest`, or `relation_digest`. A tensor
    /// registered with [`TcfWriter::register_block_tensor`] or
    /// [`TcfWriter::register_raw_tensor`] and never finished with
    /// [`TcfWriter::finish_streaming`] is [`TcfError::PayloadMismatch`]. A
    /// failed seek or write on `sink` is [`TcfError::Io`], carrying that
    /// error's `ErrorKind` and message.
    pub fn finish_into<W: Write + Seek>(mut self, sink: &mut W) -> Result<(), TcfError> {
        let (layout, mut directory) = self.prepare()?;
        self.emit_payloads(&mut directory, sink, &layout)?;
        self.write_directory(&mut directory, &layout, sink)
    }

    /// Pass 1 in full: the record digests, the layout, and the directory
    /// buffer with the header, the six record arrays, and the string table
    /// already in it. MIGRATION.md Section 4.5.1.
    ///
    /// The buffer is `[0, data_off)` only — a function of record count, not
    /// of payload size. Pass 3 hashes it, and it is written to the sink
    /// last. Shared by [`TcfWriter::finish_into`] and
    /// [`TcfWriter::finish_streaming`], which differ only in where pass 2
    /// gets each payload.
    pub(crate) fn prepare(&mut self) -> Result<(Layout, Vec<u8>), TcfError> {
        self.fill_record_digests()?;
        let layout = self.plan()?;
        let size = usize::try_from(layout.data_off).map_err(|_| LAYOUT_BOUNDS)?;
        let mut directory = vec![0u8; size];
        self.emit_directory(&mut directory, &layout)?;
        Ok((layout, directory))
    }

    /// Pass 3, then the single directory write at offset `0`. Section 5.3.
    ///
    /// Runs after every payload digest is in `directory`, which is why the
    /// directory is the last thing to reach the sink on both paths.
    pub(crate) fn write_directory<W: Write + Seek>(
        &self,
        directory: &mut [u8],
        layout: &Layout,
        sink: &mut W,
    ) -> Result<(), TcfError> {
        finalize(directory, layout)?;
        seek_write(sink, 0, directory)
    }

    /// Compute `policy_digest`, `contract_digest`, and `relation_digest`
    /// on every record that carries one. Section 7, Section 9, Section 11.
    ///
    /// Each is BLAKE3-128 over a range of the record's own encoded bytes,
    /// so each record is encoded into a scratch buffer with its digest
    /// field still zero, hashed, and the result written back. Section 9
    /// makes all three mandatory for a producer, and all three verifiable
    /// by a reader recomputing exactly this.
    ///
    /// Runs before `plan`, because the digests cover only fields the caller
    /// supplied — no digest here depends on a layout offset.
    fn fill_record_digests(&mut self) -> Result<(), TcfError> {
        let strings = &self.strings;
        for module in &mut self.modules {
            if module.policy_digest != [0u8; 16] {
                return Err(writer_owned("ModuleRecord.policy_digest"));
            }
            let mut image = [0u8; MODULE_RECORD_BYTES];
            module.encode(&mut image)?;
            let name = name_bytes(strings, module.name)?;
            module.policy_digest = *policy_digest(&image, name)?.as_bytes();
        }
        for contract in &mut self.contracts {
            if contract.contract_digest != [0u8; 16] {
                return Err(writer_owned("ContractRecord.contract_digest"));
            }
            let mut image = [0u8; CONTRACT_RECORD_BYTES];
            contract.encode(&mut image)?;
            contract.contract_digest = *contract_digest(&image)?.as_bytes();
        }
        for relation in &mut self.relations {
            if relation.relation_digest != [0u8; 16] {
                return Err(writer_owned("RelationRecord.relation_digest"));
            }
            let mut image = [0u8; RELATION_RECORD_BYTES];
            relation.encode(&mut image)?;
            relation.relation_digest = *relation_digest(&image)?.as_bytes();
        }
        Ok(())
    }
}

/// The string-table bytes a [`StringRef`] names, for the name
/// `policy_digest` concatenates. Section 6, Section 7.
///
/// The empty name is `(0, 0)` and yields no bytes, which is what a record
/// carrying no name contributes to its digest.
fn name_bytes(strings: &[u8], r: StringRef) -> Result<&[u8], TcfError> {
    let off = usize::try_from(r.off()).map_err(|_| LAYOUT_BOUNDS)?;
    let len = usize::try_from(r.len()).map_err(|_| LAYOUT_BOUNDS)?;
    let end = off.checked_add(len).ok_or(LAYOUT_BOUNDS)?;
    strings.get(off..end).ok_or(TcfError::SectionBounds {
        section: "String table",
    })
}

#[cfg(test)]
mod tests;
