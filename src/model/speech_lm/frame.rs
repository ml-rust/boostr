//! Per-frame conversion between codec codes and flat vocabulary ids.
//!
//! Split out of [`vocab`](super::vocab) because it is the only part that reasons
//! about a frame's INTERNAL structure — position-to-codebook order — rather than
//! about the id layout itself.

use crate::error::{Error, Result};

use super::vocab::SpeechVocab;

impl SpeechVocab {
    /// One frame's codes to ids, in frame position order.
    ///
    /// `codes.len()` must equal
    /// [`CodecVocab::codes_per_frame`](super::codec::CodecVocab::codes_per_frame);
    /// position `i` belongs to the codebook named by
    /// [`CodecVocab::frame_codebooks`](super::codec::CodecVocab::frame_codebooks).
    pub fn encode_frame(&self, codes: &[usize]) -> Result<Vec<u32>> {
        let layout = self.codec().frame_codebooks();
        if codes.len() != layout.len() {
            return Err(Error::InvalidArgument {
                arg: "codes",
                reason: format!(
                    "expected {} codes per frame, got {}",
                    layout.len(),
                    codes.len()
                ),
            });
        }
        let mut ids = Vec::with_capacity(codes.len());
        for (codebook, code) in layout.iter().zip(codes.iter()) {
            ids.push(self.audio_token(*codebook, *code)?);
        }
        Ok(ids)
    }

    /// One frame's ids back to codes, validating codebook index against position.
    ///
    /// The positional check is the point of this method. A model that emits a
    /// codebook-2 token where codebook 0 belongs produces audio that decodes to
    /// noise with no other symptom, so the mismatch is raised as an error here
    /// rather than passed on to the codec.
    pub fn decode_frame(&self, ids: &[u32]) -> Result<Vec<usize>> {
        let layout = self.codec().frame_codebooks();
        if ids.len() != layout.len() {
            return Err(Error::InvalidArgument {
                arg: "ids",
                reason: format!("expected {} ids per frame, got {}", layout.len(), ids.len()),
            });
        }
        let mut codes = Vec::with_capacity(ids.len());
        for (pos, (expected, id)) in layout.iter().zip(ids.iter()).enumerate() {
            let (codebook, code) =
                self.decode_audio_token(*id)
                    .ok_or_else(|| Error::InvalidArgument {
                        arg: "ids",
                        reason: format!("id {id} at frame position {pos} is not an audio token"),
                    })?;
            if codebook != *expected {
                return Err(Error::InvalidArgument {
                    arg: "ids",
                    reason: format!(
                        "id {id} at frame position {pos} belongs to codebook {codebook}, \
                         expected codebook {expected}"
                    ),
                });
            }
            codes.push(code);
        }
        Ok(codes)
    }
}

#[cfg(test)]
#[path = "frame_tests.rs"]
mod tests;
