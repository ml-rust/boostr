//! Token space of a neural audio codec, independent of any LM vocabulary.
//!
//! Kept separate from [`crate::model::speech_lm::vocab`] because a codec's shape
//! is a property of the audio model, not of the text LM it is grafted onto: the
//! same codec description drives layouts over different base vocabularies.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Description of a neural audio codec's token space.
///
/// A codec is fully described, for layout purposes, by three things:
/// - how many codebooks it has,
/// - how many entries each codebook holds,
/// - how many codes each codebook contributes to one frame.
///
/// The third field is what makes interleaved codecs work. NeuCodec emits one code
/// per frame from its one codebook. SNAC emits a hierarchy where deeper codebooks
/// fire more often per frame, so a frame is a flat run of codes whose codebook
/// index varies by position. `codes_per_codebook` records that shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodecVocab {
    num_codebooks: usize,
    codebook_size: usize,
    codes_per_codebook: Vec<usize>,
}

impl CodecVocab {
    /// Codec where every codebook contributes exactly one code per frame.
    ///
    /// Covers NeuCodec (`new(1, 65_536)`) and flat residual codecs (`new(3, 4096)`).
    pub fn new(num_codebooks: usize, codebook_size: usize) -> Result<Self> {
        Self::with_frame_layout(codebook_size, vec![1; num_codebooks])
    }

    /// Codec with a per-codebook code count per frame, for interleaved layouts.
    ///
    /// `codes_per_codebook[c]` is how many codes codebook `c` emits in one frame.
    /// A SNAC-style hierarchy is `with_frame_layout(4096, vec![1, 2, 4])`.
    pub fn with_frame_layout(codebook_size: usize, codes_per_codebook: Vec<usize>) -> Result<Self> {
        let num_codebooks = codes_per_codebook.len();
        if num_codebooks == 0 {
            return Err(Error::InvalidArgument {
                arg: "num_codebooks",
                reason: "codec must have at least one codebook".to_string(),
            });
        }
        if codebook_size == 0 {
            return Err(Error::InvalidArgument {
                arg: "codebook_size",
                reason: "codebook must have at least one entry".to_string(),
            });
        }
        for (c, n) in codes_per_codebook.iter().enumerate() {
            if *n == 0 {
                return Err(Error::InvalidArgument {
                    arg: "codes_per_codebook",
                    reason: format!("codebook {c} contributes 0 codes per frame"),
                });
            }
        }
        // Ids are u32 on the wire (token ids into an embedding table), so the audio
        // span alone must fit even before text and control tokens are added.
        let span = num_codebooks
            .checked_mul(codebook_size)
            .filter(|s| *s <= u32::MAX as usize)
            .ok_or_else(|| Error::InvalidArgument {
                arg: "codebook_size",
                reason: format!(
                    "{num_codebooks} codebooks x {codebook_size} entries overflows the u32 id space"
                ),
            })?;
        debug_assert!(span > 0);
        Ok(Self {
            num_codebooks,
            codebook_size,
            codes_per_codebook,
        })
    }

    /// Number of codebooks.
    pub fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }

    /// Entries in each codebook.
    pub fn codebook_size(&self) -> usize {
        self.codebook_size
    }

    /// Total audio ids this codec occupies: `num_codebooks * codebook_size`.
    pub fn total_audio_tokens(&self) -> usize {
        self.num_codebooks * self.codebook_size
    }

    /// Codes in one frame, summed over codebooks.
    pub fn codes_per_frame(&self) -> usize {
        self.codes_per_codebook.iter().sum()
    }

    /// How many codes codebook `c` contributes per frame, or `None` if out of range.
    pub fn codes_of_codebook(&self, codebook: usize) -> Option<usize> {
        self.codes_per_codebook.get(codebook).copied()
    }

    /// Codebook index expected at each position within a frame.
    ///
    /// Length is [`codes_per_frame`](Self::codes_per_frame). This is the ground
    /// truth [`SpeechVocab::decode_frame`](super::vocab::SpeechVocab::decode_frame) validates a model's emissions against.
    pub fn frame_codebooks(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.codes_per_frame());
        for (c, n) in self.codes_per_codebook.iter().enumerate() {
            out.extend(std::iter::repeat_n(c, *n));
        }
        out
    }
}
