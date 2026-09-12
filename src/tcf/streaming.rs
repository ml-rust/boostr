//! [`TcfWriter::finish_streaming`]: the write path that holds one tensor's
//! payload at a time. FORMAT.md Section 4.1, Section 8.0.1,
//! Section 12.3, Section 15.2; MIGRATION.md Section 4.5.1.
//!
//! # Why a callback
//!
//! The directory is computable from headers alone: `physical_span_bytes` is
//! a function of shape and encoding, and `data_offset` is a running
//! 64-aligned sum (Section 4.1). Only `semantic_digest`, `payload_digest`,
//! and the 64 proof values need tensor data, and all three live in the
//! `TensorRecord`, which the writer emits last.
//!
//! So a caller registers every record first — with
//! [`TcfWriter::register_quantized_tensor`] or
//! [`TcfWriter::register_raw_tensor`] — and hands over a callback the writer
//! pulls from once per tensor in directory order. The producer decides when
//! to read and quantize each tensor; the writer digests it, packs it, writes
//! it at its precomputed offset, and drops it before asking for the next.
//! Peak memory is the directory plus one tensor, which is the invariant a
//! producer converting a multi-gigabyte model needs.
//!
//! # What this module does not do
//!
//! Layout arithmetic, packing, and digest order all stay in
//! [`crate::tcf::writer`]: this module supplies payloads to the same pass 2 that
//! [`TcfWriter::finish_into`] drives, so the two produce identical bytes by
//! construction rather than by agreement.

use std::io::{Seek, Write};

use crate::tcf::encoding::Encoding;
use crate::tcf::error::TcfError;
use crate::tcf::writer::{LAYOUT_BOUNDS, Payload, TcfWriter};

impl TcfWriter {
    /// Emit the complete file into `sink`, pulling each tensor's payload
    /// from `produce` as it is needed. Section 4.1,
    /// MIGRATION.md Section 4.5.1.
    ///
    /// The writer calls `produce` once for every tensor registered without
    /// a payload, in directory order, passing that tensor's index in the
    /// tensor array — the value
    /// [`TcfWriter::register_quantized_tensor`] returned. The payload is
    /// digested, packed, written at its precomputed `data_offset`, and
    /// dropped before the next call, so at most one payload is resident.
    ///
    /// A tensor added with [`TcfWriter::add_quantized_tensor`] or
    /// [`TcfWriter::add_raw_tensor`] already holds its payload, and
    /// `produce` is not called for it. Mixing the two is allowed and
    /// changes nothing about the bytes; it only changes how much is
    /// resident.
    ///
    /// The file this writes is byte-for-byte identical to what
    /// [`TcfWriter::finish`] and [`TcfWriter::finish_into`] write for the
    /// same tensors in the same order.
    ///
    /// # Contract
    ///
    /// The sink contract is [`TcfWriter::finish_into`]'s, unchanged: no
    /// flush, an unspecified stream position on return, and payloads
    /// written before the directory that fills the hole below `data_off`.
    ///
    /// `produce` MUST return a payload matching the registered record: the
    /// [`Payload::Block`] variant for a block encoding with exactly the
    /// byte count the shape implies (Section 12.3), or the
    /// [`Payload::Raw`] variant for a raw encoding with exactly
    /// `product(dims) * width` bytes (Section 8.0.1). A payload that
    /// disagrees is [`TcfError::PayloadMismatch`], naming the tensor index,
    /// what the record requires, and what arrived. Nothing is truncated and
    /// no partial file is presented as complete.
    ///
    /// # Errors
    /// Every error [`TcfWriter::finish_into`] raises, plus
    /// [`TcfError::PayloadMismatch`] for a payload that disagrees with its
    /// record. An error `produce` itself returns propagates unchanged, so a
    /// producer's own read or quantize failure reaches the caller as it
    /// stands.
    pub fn finish_streaming<W, F>(mut self, sink: &mut W, mut produce: F) -> Result<(), TcfError>
    where
        W: Write + Seek,
        F: FnMut(usize) -> Result<Payload, TcfError>,
    {
        let (layout, mut directory) = self.prepare()?;

        // Pass 2. `stored` holds one `Option` per tensor and no payload for
        // a registered one, so the loop's live payload is whatever the last
        // iteration produced — dropped by `write_payload` before the next
        // `produce` call.
        let stored = self.take_payloads();
        for (index, slot) in stored.into_iter().enumerate() {
            let payload = match slot {
                Some(payload) => payload,
                None => {
                    let produced = produce(index)?;
                    self.check_produced(index, &produced)?;
                    produced
                }
            };
            self.write_payload(index, payload, &mut directory, sink, &layout)?;
        }

        self.write_directory(&mut directory, &layout, sink)
    }

    /// Reject a produced payload that disagrees with the record registered
    /// at `index`. Section 8.0.1, Section 12.3, Section 15.2.
    ///
    /// Runs before pass 2 touches the payload, so a mismatch costs no
    /// partial write. The record's own fields are the expectation: `plan`
    /// has already set `logical_payload_bytes` from the shape and the
    /// encoding.
    fn check_produced(&self, index: usize, payload: &Payload) -> Result<(), TcfError> {
        let record = self.tensor_at(index)?;
        let tensor_index = u32::try_from(index).map_err(|_| LAYOUT_BOUNDS)?;
        let mismatch = |expected: String| TcfError::PayloadMismatch {
            tensor_index,
            expected,
            supplied: payload.kind().to_owned(),
        };

        match (record.encoding, payload) {
            (Encoding::Raw(_), Payload::Raw(bytes))
            | (Encoding::Block(_), Payload::Block { bytes, .. }) => {
                let expected = record.logical_payload_bytes;
                let supplied = u64::try_from(bytes.len()).map_err(|_| LAYOUT_BOUNDS)?;
                if supplied != expected {
                    return Err(TcfError::PayloadMismatch {
                        tensor_index,
                        expected: format!("{expected} payload bytes"),
                        supplied: format!("{supplied} payload bytes"),
                    });
                }
                Ok(())
            }
            (Encoding::Block(_), _) => Err(mismatch(
                "a block stream plus proof values for a block encoding".to_owned(),
            )),
            (Encoding::Raw(_), _) => Err(mismatch("verbatim bytes for a raw encoding".to_owned())),
        }
    }
}
