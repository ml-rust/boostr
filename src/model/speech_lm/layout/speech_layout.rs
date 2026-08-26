//! [`SpeechLayout`]: which token layout a record is packed into, named
//! explicitly.
//!
//! There is more than one correct answer, and the wrong one is silent. Training
//! on boostr's own layout points every id at an embedding row that means
//! something else on a base model that already has a trained layout of its own,
//! and nothing about the resulting checkpoint reports the mismatch — it just
//! learns worse.
//!
//! So the layout is a VALUE the caller states, never a default:
//!
//! - [`SpeechLayout::Native`] — boostr's own layout over a
//!   [`SpeechVocab`]. The right thing when training a speech LM from scratch on
//!   boostr, where nothing constrains the id space and the reserved control
//!   region buys future headroom.
//! - [`SpeechLayout::ExpressiveTts`] — the layout already trained into
//!   `Scicom-intl/Multilingual-Expressive-TTS-1.7B`. The right thing when
//!   fine-tuning THAT base, and the only thing that base understands.
//!
//! The two produce different sequences from the same record, on purpose. Neither
//! is a fallback for the other.

use crate::error::{Error, Result};
use crate::model::speech_lm::pack::{SpeechRecord, pack_record, pack_records};
use crate::model::speech_lm::vocab::SpeechVocab;

use super::expressive_tts::ExpressiveTtsLayout;

/// The token layout a speech record is flattened into.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeechLayout {
    /// boostr's own layout: text, then a reserved control region, then audio.
    ///
    /// See [`crate::model::speech_lm::pack`] for the sequence it emits.
    Native(SpeechVocab),
    /// The `Multilingual-Expressive-TTS-1.7B` layout.
    ///
    /// See [`super::expressive_tts`] for the sequence it emits and for what it
    /// cannot express.
    ExpressiveTts(ExpressiveTtsLayout),
}

impl SpeechLayout {
    /// The `Multilingual-Expressive-TTS-1.7B` layout, with the ids read from
    /// that checkpoint's `added_tokens.json`.
    pub const fn expressive_tts() -> Self {
        Self::ExpressiveTts(ExpressiveTtsLayout::new())
    }

    /// Short name, for error messages and corpus sidecars.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Native(_) => "boostr-native",
            Self::ExpressiveTts(_) => "expressive-tts-1.7b",
        }
    }

    /// First audio id.
    pub fn audio_base(&self) -> usize {
        match self {
            Self::Native(vocab) => vocab.audio_base(),
            Self::ExpressiveTts(layout) => layout.audio_base() as usize,
        }
    }

    /// Codes in one codebook.
    pub fn codebook_size(&self) -> usize {
        match self {
            Self::Native(vocab) => vocab.codec().codebook_size(),
            Self::ExpressiveTts(layout) => layout.codebook_size(),
        }
    }

    /// Total ids, i.e. the embedding and output-projection row count a
    /// checkpoint must have to read this layout.
    pub fn total_size(&self) -> usize {
        match self {
            Self::Native(vocab) => vocab.total_size(),
            Self::ExpressiveTts(layout) => layout.vocab_size(),
        }
    }

    /// True if a checkpoint with `rows` embedding rows matches this layout.
    /// True if `id` is an audio token under this layout.
    ///
    /// Mirrors [`SpeechVocab::is_audio`] and answers it for BOTH layouts, which
    /// is what lets a trainer build an audio-only loss mask for a corpus packed
    /// under either. Without it, masking could only be expressed for
    /// [`Self::Native`], and a corpus packed as `expressive_tts` had to train
    /// with loss on its text tokens too — where the text is the conditioning,
    /// not the target.
    pub fn is_audio(&self, id: u32) -> bool {
        let id = id as usize;
        id >= self.audio_base() && id < self.total_size()
    }

    pub fn matches_embedding_rows(&self, rows: usize) -> bool {
        rows == self.total_size()
    }

    /// The [`SpeechVocab`] behind [`SpeechLayout::Native`], or `None`.
    ///
    /// `None` is not an oversight. `ExpressiveTts` puts `<|description|>` AFTER
    /// its audio run, and `SpeechVocab` reserves control ids BEFORE audio by
    /// construction, so no `SpeechVocab` describes that base.
    pub fn vocab(&self) -> Option<&SpeechVocab> {
        match self {
            Self::Native(vocab) => Some(vocab),
            Self::ExpressiveTts(_) => None,
        }
    }

    /// Flatten one record into this layout's sequence.
    pub fn pack_record(&self, record: &SpeechRecord<'_>) -> Result<Vec<u32>> {
        match self {
            Self::Native(vocab) => pack_record(vocab, record),
            Self::ExpressiveTts(layout) => layout.pack_record(record),
        }
    }

    /// Flatten records back to back, with no separator between them.
    ///
    /// Under either layout the boundary is already unambiguous: a record ends
    /// with its terminator (`<|/speech|>` or `<|im_end|>`) and the next opens
    /// with its own opening token.
    pub fn pack_records(&self, records: &[SpeechRecord<'_>]) -> Result<Vec<u32>> {
        match self {
            Self::Native(vocab) => pack_records(vocab, records),
            Self::ExpressiveTts(layout) => {
                let mut out = Vec::new();
                for (i, record) in records.iter().enumerate() {
                    let packed =
                        layout
                            .pack_record(record)
                            .map_err(|e| Error::InvalidArgument {
                                arg: "records",
                                reason: format!("record {i}: {e}"),
                            })?;
                    out.extend_from_slice(&packed);
                }
                Ok(out)
            }
        }
    }

    /// [`pack_records`](Self::pack_records), then pad the tail up to a multiple
    /// of `pad_to_multiple` with this layout's pad id.
    ///
    /// Native pads with `<|speech_pad|>`; ExpressiveTts pads with the
    /// checkpoint's own `<|endoftext|>` (151_643), which is its `pad_token_id`.
    pub fn pack_records_padded(
        &self,
        records: &[SpeechRecord<'_>],
        pad_to_multiple: usize,
    ) -> Result<Vec<u32>> {
        if pad_to_multiple == 0 {
            return Err(Error::InvalidArgument {
                arg: "pad_to_multiple",
                reason: "window length must be at least 1".to_string(),
            });
        }
        match self {
            Self::Native(vocab) => {
                crate::model::speech_lm::pack::pack_records_padded(vocab, records, pad_to_multiple)
            }
            Self::ExpressiveTts(layout) => {
                let mut out = self.pack_records(records)?;
                let remainder = out.len() % pad_to_multiple;
                if remainder != 0 {
                    out.resize(out.len() + (pad_to_multiple - remainder), layout.pad_id());
                }
                Ok(out)
            }
        }
    }
}

#[cfg(test)]
mod tests;
