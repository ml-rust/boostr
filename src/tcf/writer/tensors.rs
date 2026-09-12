//! The tensor entry points of [`TcfWriter`]: block and raw, each with a
//! payload now (`add_*`) or streamed later (`register_*`). Section 8,
//! Section 12, Section 15.3.

use crate::tcf::binary16::f32_to_bits;
use crate::tcf::consts::PROOF_COUNT;
use crate::tcf::encoding::Encoding;
use crate::tcf::enums::ProofFormat;
use crate::tcf::error::TcfError;
use crate::tcf::record::TensorRecord;

use super::layout::header_count;
use super::payload::Payload;
use super::{LAYOUT_BOUNDS, TcfWriter};

/// A writer-owned field a caller set to a conflicting non-zero value.
///
/// Section 17 defines no separate code for it, and the meaning matches:
/// these bytes belong to the writer, so a caller MUST leave them zero, and a
/// non-zero one is rejected naming the exact field.
pub(super) const fn writer_owned(field: &'static str) -> TcfError {
    TcfError::NonzeroReserved { field }
}

impl TcfWriter {
    /// Declare a block-encoded tensor: `bytes` is the GGML block stream for
    /// the tensor, stored verbatim, and `proof` the dequantized value at each
    /// proof index (Section 15.3), read by the producer's own block decoder.
    /// Returns the tensor's index in the tensor array.
    ///
    /// TCF holds no decoder for GGML blocks, so the proof vector is the
    /// producer's statement of what the bytes mean; a reader with a decoder
    /// checks it, and one without has the payload digest alone.
    ///
    /// # Errors
    /// - [`TcfError::NonzeroReserved`] naming the field, if a writer-owned
    ///   field is non-zero.
    /// - [`TcfError::UnsupportedEncoding`] if `record.encoding` is not a
    ///   block encoding.
    /// - [`TcfError::InvalidQuantShape`] if `rank < 2`, the row width is not
    ///   a whole number of blocks, `bytes.len()` disagrees with the shape,
    ///   or `proof.len()` is not [`PROOF_COUNT`](crate::tcf::consts::PROOF_COUNT).
    /// - [`TcfError::SectionBounds`] if the array would exceed the `u32`
    ///   count the header stores.
    pub fn add_block_tensor(
        &mut self,
        record: TensorRecord,
        bytes: Vec<u8>,
        proof: &[f32],
    ) -> Result<u32, TcfError> {
        let expected = check_block(&record)?;
        if u64::try_from(bytes.len()).map_err(|_| LAYOUT_BOUNDS)? != expected
            || proof.len() != PROOF_COUNT as usize
        {
            return Err(TcfError::InvalidQuantShape {
                tensor_id: record.tensor_id,
            });
        }
        let proof = proof.iter().map(|&v| f32_to_bits(v)).collect();
        self.push_tensor(record, Some(Payload::Block { bytes, proof }))
    }

    /// Declare a block-encoded tensor's record with no payload yet. Returns
    /// the tensor's index, which [`TcfWriter::finish_streaming`] passes to
    /// its callback; the callback answers with [`Payload::Block`].
    ///
    /// # Errors
    /// As [`TcfWriter::add_block_tensor`], less the length checks pass 2
    /// makes against the produced payload.
    pub fn register_block_tensor(&mut self, record: TensorRecord) -> Result<u32, TcfError> {
        check_block(&record)?;
        self.push_tensor(record, None)
    }

    /// Declare a raw tensor: `bytes` are stored verbatim. Section 8,
    /// Section 15.3. Returns the tensor's index in the tensor array.
    ///
    /// A raw tensor has no semantic digest and no proof vector — the bytes
    /// are the values, and `payload_digest` already covers them
    /// (Section 15.3).
    ///
    /// `bytes.len()` MUST be `product(dims) * width` for the encoding's
    /// element width (Section 8.0.1). The writer computes that length
    /// rather than accepting one, and [`TcfWriter::finish`] rejects a
    /// vector that disagrees with [`TcfError::InvalidQuantShape`]: nothing
    /// else in the file constrains a raw tensor's length, so a truncated or
    /// padded payload would otherwise round-trip unnoticed.
    ///
    /// # Errors
    /// - [`TcfError::NonzeroReserved`] naming the field, if a writer-owned
    ///   field is non-zero.
    /// - [`TcfError::UnsupportedEncoding`] if `record.encoding` is a block
    ///   encoding; use [`TcfWriter::add_block_tensor`].
    /// - [`TcfError::SectionBounds`] if the array would exceed the `u32`
    ///   count the header stores.
    pub fn add_raw_tensor(
        &mut self,
        record: TensorRecord,
        bytes: Vec<u8>,
    ) -> Result<u32, TcfError> {
        check_raw(&record)?;
        self.push_tensor(record, Some(Payload::Raw(bytes)))
    }

    /// Declare a raw tensor's record with no payload yet. Section 8,
    /// Section 15.3. Returns the tensor's index in the tensor array, which
    /// is the index [`TcfWriter::finish_streaming`] passes to its callback.
    ///
    /// This is [`TcfWriter::add_raw_tensor`] without the bytes. See
    /// [`TcfWriter::register_block_tensor`] for why the record alone is
    /// enough to place the tensor, and for the requirement that a writer
    /// holding one be finished with [`TcfWriter::finish_streaming`].
    ///
    /// # Errors
    /// - [`TcfError::NonzeroReserved`] naming the field, if a writer-owned
    ///   field is non-zero.
    /// - [`TcfError::UnsupportedEncoding`] if `record.encoding` is a block
    ///   encoding; use [`TcfWriter::register_block_tensor`].
    /// - [`TcfError::SectionBounds`] if the array would exceed the `u32`
    ///   count the header stores.
    pub fn register_raw_tensor(&mut self, record: TensorRecord) -> Result<u32, TcfError> {
        check_raw(&record)?;
        self.push_tensor(record, None)
    }

    /// Append a tensor and its payload slot, keeping the two arrays in step.
    /// `None` is a tensor registered without a payload.
    fn push_tensor(
        &mut self,
        record: TensorRecord,
        payload: Option<Payload>,
    ) -> Result<u32, TcfError> {
        let index = header_count(self.tensors.len(), "TensorRecord[]")?;
        self.tensors.push(record);
        self.payloads.push(payload);
        Ok(index)
    }
}

/// Validate a block-encoded tensor's record and return its payload length.
///
/// Shared by [`TcfWriter::add_block_tensor`] and
/// [`TcfWriter::register_block_tensor`].
fn check_block(record: &TensorRecord) -> Result<u64, TcfError> {
    check_writer_owned(record)?;
    let Encoding::Block(block) = record.encoding else {
        return Err(TcfError::UnsupportedEncoding {
            raw: record.encoding.to_u16(),
        });
    };
    block.payload_bytes(record.shape(), record.rank, record.tensor_id)
}

/// Validate a raw tensor's record. Section 8, Section 15.3.
///
/// Shared by [`TcfWriter::add_raw_tensor`] and
/// [`TcfWriter::register_raw_tensor`]. The payload length is not checked
/// here: Section 8.0.1 determines it from the shape, and `plan` computes it.
fn check_raw(record: &TensorRecord) -> Result<(), TcfError> {
    check_writer_owned(record)?;
    if record.encoding.is_quantized() {
        return Err(TcfError::UnsupportedEncoding {
            raw: record.encoding.to_u16(),
        });
    }
    Ok(())
}

/// Reject a writer-owned field a caller filled in. See the module
/// documentation for why this is an error and not an overwrite.
fn check_writer_owned(t: &TensorRecord) -> Result<(), TcfError> {
    let u64_fields = [
        (t.data_offset, "TensorRecord.data_offset"),
        (
            t.logical_payload_bytes,
            "TensorRecord.logical_payload_bytes",
        ),
        (t.physical_span_bytes, "TensorRecord.physical_span_bytes"),
        (t.resident_bytes, "TensorRecord.resident_bytes"),
        (t.transfer_bytes, "TensorRecord.transfer_bytes"),
        (t.proof_rel_off, "TensorRecord.proof_rel_off"),
    ];
    for (value, field) in u64_fields {
        if value != 0 {
            return Err(writer_owned(field));
        }
    }
    if t.proof_count != 0 {
        return Err(writer_owned("TensorRecord.proof_count"));
    }
    if t.proof_format != ProofFormat::None {
        return Err(writer_owned("TensorRecord.proof_format"));
    }
    if t.semantic_digest != [0u8; 16] {
        return Err(writer_owned("TensorRecord.semantic_digest"));
    }
    if t.payload_digest != [0u8; 16] {
        return Err(writer_owned("TensorRecord.payload_digest"));
    }
    Ok(())
}
