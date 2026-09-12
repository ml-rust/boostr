//! [`TcfFile::verify_tensor`]: the Section 15 payload checks, the only
//! reads of a payload page besides [`TcfFile::payload`].

use crate::tcf::digest::{Digest128, payload_digest};
use crate::tcf::encoding::Encoding;
use crate::tcf::error::TcfError;
use crate::tcf::proof::{BlockDecoder, PROOF_BYTES, block_proof_values};
use crate::tcf::record::TensorRecord;
use crate::tcf::record::field::RecordField;

use super::sections::bounds;
use super::{TcfFile, rel_data_offset};

impl<'a> TcfFile<'a> {
    /// Verify one tensor, in the Section 15 order: `payload_digest` over the
    /// stored bytes, then the recomputed logical stream against
    /// `semantic_digest`, then the 64 proof values.
    ///
    /// A tensor in a raw encoding has no semantic digest and no proof
    /// vector — the bytes are the values — so only `payload_digest` is
    /// checked (Section 15.3).
    ///
    /// A tensor in a block encoding is a GGML block stream TCF does not
    /// decode: its semantic digest is checked to equal its payload digest,
    /// and its proof vector is checked only by
    /// [`TcfFile::verify_tensor_with`], which takes the caller's decoder.
    /// This method skips that step for a block tensor.
    ///
    /// The three mechanisms are ordered deliberately: the cheapest check
    /// that can fail is run first, and the proof vector is a screen, never
    /// the authority (Section 15.3.1).
    ///
    /// # Errors
    /// - [`TcfError::PayloadDigestMismatch`]: the stored bytes changed.
    /// - [`TcfError::NonzeroReserved`]: a trailing alignment padding byte in
    ///   `[data_offset + logical_payload_bytes, data_offset +
    ///   physical_span_bytes)` is non-zero (Section 14.4, Section 15.2).
    /// - [`TcfError::SemanticDigestMismatch`]: a block tensor's semantic
    ///   digest is not its payload digest.
    /// - [`TcfError::SectionBounds`]: the payload or the proof vector runs
    ///   outside its section.
    pub fn verify_tensor(&self, t: &TensorRecord) -> Result<(), TcfError> {
        self.verify_tensor_with(t, None)
    }

    /// [`TcfFile::verify_tensor`] with a block decoder, so a block-encoded
    /// tensor's proof vector is checked too: each stored value must equal
    /// the decoder's value at that proof index, as binary16 bits. Raw
    /// tensors are verified exactly as without one.
    ///
    /// # Errors
    /// As [`TcfFile::verify_tensor`], plus [`TcfError::ProofMismatch`] for a
    /// block tensor whose bytes the decoder reads differently from the
    /// producer, and the decoder's own errors.
    pub fn verify_tensor_with(
        &self,
        t: &TensorRecord,
        decoder: Option<&dyn BlockDecoder>,
    ) -> Result<(), TcfError> {
        let payload = self.payload(t)?;
        let stored_payload_digest = payload_digest(payload);
        if stored_payload_digest != Digest128::from_bytes(t.payload_digest) {
            return Err(TcfError::PayloadDigestMismatch {
                tensor_id: t.tensor_id,
            });
        }

        // Section 14.4, Section 15.2: trailing alignment padding cannot be
        // checked at `open` without touching a payload page, so it is
        // checked here, for every encoding — raw included, since alignment
        // padding applies to any payload.
        if self.padding(t)?.iter().any(|&b| b != 0) {
            return Err(TcfError::NonzeroReserved {
                field: "TensorRecord.padding",
            });
        }

        let Encoding::Block(block) = t.encoding else {
            // A raw tensor: the bytes are the values, and the payload digest
            // has covered them.
            return Ok(());
        };
        // The block stream is the logical form, so the two digests coincide
        // by construction; a producer that wrote anything else in
        // `semantic_digest` is caught here.
        if stored_payload_digest != Digest128::from_bytes(t.semantic_digest) {
            return Err(TcfError::SemanticDigestMismatch {
                tensor_id: t.tensor_id,
            });
        }
        let Some(decoder) = decoder else {
            return Ok(());
        };
        let expected = block_proof_values(decoder, block, payload, t.shape(), t.rank, t.tensor_id)?;
        self.check_proof(t, &expected)
    }

    /// Compare `expected` proof values against the stored proof vector.
    fn check_proof(&self, t: &TensorRecord, expected: &[u16]) -> Result<(), TcfError> {
        let stored = self.proof_vector(t)?;
        for (index, value) in expected.iter().enumerate() {
            let at = index
                .checked_mul(2)
                .ok_or(TcfError::TileArithmeticOverflow)?;
            let raw = <u16 as RecordField>::read(stored, at, "proof section")?;
            if raw != *value {
                return Err(TcfError::ProofMismatch {
                    tensor_id: t.tensor_id,
                    proof_index: u32::try_from(index).unwrap_or(u32::MAX),
                });
            }
        }
        Ok(())
    }

    /// This tensor's trailing alignment padding: bytes `[data_offset +
    /// logical_payload_bytes, data_offset + physical_span_bytes)`. Section
    /// 14.4. Reads a payload page, so only [`TcfFile::verify_tensor`] calls
    /// this — never `open` (Section 16).
    fn padding(&self, t: &TensorRecord) -> Result<&'a [u8], TcfError> {
        let rel = rel_data_offset(t, &self.header)?;
        let start = rel
            .checked_add(t.logical_payload_bytes)
            .ok_or(bounds("tensor payload"))?;
        let end = rel
            .checked_add(t.physical_span_bytes)
            .ok_or(bounds("tensor payload"))?;
        let start = usize::try_from(start).map_err(|_| bounds("tensor payload"))?;
        let end = usize::try_from(end).map_err(|_| bounds("tensor payload"))?;
        self.data.get(start..end).ok_or(bounds("tensor payload"))
    }

    /// This tensor's stored proof bytes, inside the proof section.
    /// Section 15.3. The proof section is directory, not payload, so this
    /// reads no page Section 16 protects.
    fn proof_vector(&self, t: &TensorRecord) -> Result<&'a [u8], TcfError> {
        let start = self
            .header
            .proof_off
            .checked_add(t.proof_rel_off)
            .ok_or(bounds("proof section"))?;
        let start = usize::try_from(start).map_err(|_| bounds("proof section"))?;
        let end = start
            .checked_add(PROOF_BYTES)
            .ok_or(bounds("proof section"))?;
        self.directory
            .get(start..end)
            .ok_or(bounds("proof section"))
    }
}
