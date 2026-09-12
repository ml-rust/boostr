//! Pass 2 of the writer: one tensor's payload at a time. `FORMAT.md`
//! Section 8.0.1, Section 14, Section 14.4, Section 15.
//!
//! Each payload is moved out of the writer, digested, written at the
//! `data_offset` pass 1 computed, zero-filled to `physical_span_bytes`, and
//! dropped before the next tensor is touched. That drop is what bounds
//! [`TcfWriter::finish_streaming`] to one payload in flight.
//!
//! This file owns no block layout: a block payload is the GGML stream the
//! producer's quantizer emitted, stored as-is.

use std::io::{Seek, Write};

use crate::tcf::consts::{PROOF_COUNT, SECTION_ALIGN};
use crate::tcf::digest::payload_digest;
use crate::tcf::encoding::Encoding;
use crate::tcf::error::TcfError;
use crate::tcf::proof::PROOF_BYTES;
use crate::tcf::record::TensorRecord;

use super::layout::{Layout, emit_record, put, seek_write};
use super::{LAYOUT_BOUNDS, TcfWriter};

/// One tensor's payload, dropped as soon as pass 2 has written it.
///
/// The variant MUST match the tensor's encoding: a block encoding takes
/// [`Payload::Block`], a raw one takes [`Payload::Raw`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Payload {
    /// A GGML block stream, stored as-is, with the proof values the
    /// producer's own block decoder read at the proof indices, as binary16
    /// bits. TCF has no decoder for GGML blocks, so the proof travels with
    /// the bytes; a reader checks it with its decoder.
    Block { bytes: Vec<u8>, proof: Vec<u16> },
    /// Verbatim bytes, stored as-is (Section 15.3: nothing to prove).
    Raw(Vec<u8>),
}

impl Payload {
    /// The variant's name, for an error that reports what a producer
    /// supplied.
    pub(crate) const fn kind(&self) -> &'static str {
        match self {
            Self::Block { .. } => "a block stream plus proof values for a block encoding",
            Self::Raw(_) => "verbatim bytes for a raw encoding",
        }
    }
}

impl TcfWriter {
    /// The tensor record at `index`, or [`TcfError::SectionBounds`].
    pub(crate) fn tensor_at(&self, index: usize) -> Result<&TensorRecord, TcfError> {
        self.tensors.get(index).ok_or(TcfError::SectionBounds {
            section: "TensorRecord[]",
        })
    }

    /// Take every stored payload, leaving each tensor's slot empty.
    /// [`TcfWriter::finish_streaming`] walks the result and asks its
    /// callback for each `None`.
    pub(crate) fn take_payloads(&mut self) -> Vec<Option<Payload>> {
        core::mem::take(&mut self.payloads)
    }

    /// Pass 2: one tensor at a time, in directory order.
    /// MIGRATION.md Section 4.5.1.
    ///
    /// Each payload is moved out of the writer, digested, packed, written at
    /// its precomputed `data_offset`, zero-filled to `physical_span_bytes`
    /// (Section 14.4), and dropped before the next tensor is touched.
    ///
    /// Payload bytes and their padding go to `sink`. The proof vectors and
    /// the tensor-record backpatch go to `buf`, the directory buffer, which
    /// pass 3 hashes and the caller writes out afterwards.
    ///
    /// Every payload MUST already be present. A tensor registered without
    /// one is [`TcfError::PayloadMismatch`]: only
    /// [`TcfWriter::finish_streaming`] can supply it, and writing a hole
    /// instead would produce a file whose digests do not cover its data.
    pub(super) fn emit_payloads<W: Write + Seek>(
        &mut self,
        buf: &mut [u8],
        sink: &mut W,
        layout: &Layout,
    ) -> Result<(), TcfError> {
        let payloads = core::mem::take(&mut self.payloads);
        for (index, payload) in payloads.into_iter().enumerate() {
            let Some(payload) = payload else {
                return Err(TcfError::PayloadMismatch {
                    tensor_index: u32::try_from(index).map_err(|_| LAYOUT_BOUNDS)?,
                    expected: "a payload".to_owned(),
                    supplied: "none: registered without one, so finish_streaming must write it"
                        .to_owned(),
                });
            };
            self.write_payload(index, payload, buf, sink, layout)?;
        }
        Ok(())
    }

    /// Pass 2 for one tensor: digest, pack, write at its precomputed
    /// `data_offset`, zero-fill to `physical_span_bytes`, backpatch the
    /// record. Section 14, Section 14.4, Section 15.2, Section 15.3.
    ///
    /// `payload` is consumed and dropped before this returns, which is what
    /// bounds [`TcfWriter::finish_streaming`] to one tensor in flight.
    pub(crate) fn write_payload<W: Write + Seek>(
        &mut self,
        index: usize,
        payload: Payload,
        buf: &mut [u8],
        sink: &mut W,
        layout: &Layout,
    ) -> Result<(), TcfError> {
        let record = self.tensors.get_mut(index).ok_or(TcfError::SectionBounds {
            section: "TensorRecord[]",
        })?;

        let bytes = match payload {
            Payload::Block { bytes, proof } => {
                if !matches!(record.encoding, Encoding::Block(_)) {
                    return Err(TcfError::UnsupportedEncoding {
                        raw: record.encoding.to_u16(),
                    });
                }
                // The block stream is GGML's logical form; nothing is packed
                // on this side of it, so the semantic digest is the payload
                // digest. A reader compares the two.
                record.semantic_digest = *payload_digest(&bytes).as_bytes();
                if proof.len() != PROOF_COUNT as usize {
                    return Err(TcfError::InvalidQuantShape {
                        tensor_id: record.tensor_id,
                    });
                }
                let mut proof_le = Vec::with_capacity(PROOF_BYTES);
                for value in proof {
                    proof_le.extend_from_slice(&value.to_le_bytes());
                }
                let at = layout
                    .proof_off
                    .checked_add(record.proof_rel_off)
                    .ok_or(LAYOUT_BOUNDS)?;
                put(buf, at, &proof_le)?;
                bytes
            }
            Payload::Raw(bytes) => bytes,
        };

        let written = u64::try_from(bytes.len()).map_err(|_| LAYOUT_BOUNDS)?;
        if written != record.logical_payload_bytes {
            return Err(TcfError::InvalidQuantShape {
                tensor_id: record.tensor_id,
            });
        }
        record.payload_digest = *payload_digest(&bytes).as_bytes();

        seek_write(sink, record.data_offset, &bytes)?;
        // Section 14.4: trailing alignment bytes MUST be zero, and are
        // counted in `physical_span_bytes`. When nonempty, they are
        // written explicitly: a sink is not assumed to read back as
        // zero where nothing was written. An already-aligned payload
        // has no padding, and the seek below is skipped for it.
        let pad_at = record
            .data_offset
            .checked_add(record.logical_payload_bytes)
            .ok_or(LAYOUT_BOUNDS)?;
        let pad_len = record
            .physical_span_bytes
            .checked_sub(record.logical_payload_bytes)
            .ok_or(LAYOUT_BOUNDS)?;
        let pad_len = usize::try_from(pad_len).map_err(|_| LAYOUT_BOUNDS)?;
        // `physical_span_bytes` is `logical_payload_bytes` rounded up to
        // `SECTION_ALIGN`, so the padding is shorter than one alignment
        // unit; the bound is still checked rather than assumed.
        if pad_len > 0 {
            let zeros = [0u8; SECTION_ALIGN as usize];
            let pad = zeros.get(..pad_len).ok_or(LAYOUT_BOUNDS)?;
            seek_write(sink, pad_at, pad)?;
        }

        // The tensor record was written in pass 1 with every field but
        // its two digests; rewrite it now that they are known. This is
        // the in-memory form of the seek-back MIGRATION.md Section 4.5.1
        // describes.
        let finished = *record;
        emit_record(buf, layout.tensor_off, index, &finished)
    }
}
