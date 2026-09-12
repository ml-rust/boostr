//! The record types: a borrowing [`SpeechRecord`] going in, an
//! [`OwnedSpeechRecord`] coming back out.

/// One utterance: who says it, optionally how, what the text is, and the codec
/// frames that realise it.
///
/// All text fields hold ids ALREADY produced by the base tokenizer. This type
/// does no tokenization; it only checks that what it is handed is text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechRecord<'a> {
    /// Already-tokenised speaker NAME, as plain text ids. `None` omits the
    /// speaker section and both its delimiters.
    pub speaker: Option<&'a [u32]>,
    /// Already-tokenised free-form style description. `None` omits the style
    /// section and both its delimiters.
    pub style: Option<&'a [u32]>,
    /// Already-tokenised text to be spoken.
    pub text: &'a [u32],
    /// Per-frame codec codes; one inner `Vec` per frame, each of length
    /// [`CodecVocab::codes_per_frame`](super::codec::CodecVocab::codes_per_frame).
    pub frames: &'a [Vec<usize>],
}

/// A record recovered from a packed stream, owning its ids.
///
/// The borrowing [`SpeechRecord`] cannot be returned from unpacking: the text
/// sections are copied out of the stream and the frames are decoded into fresh
/// `Vec`s. Use [`OwnedSpeechRecord::as_record`] to feed one straight back into
/// [`pack_record`](super::pack_record), which is how a packed corpus is checked
/// rather than trusted.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OwnedSpeechRecord {
    /// Speaker name ids, or `None` if the packed stream had no speaker section.
    pub speaker: Option<Vec<u32>>,
    /// Style description ids, or `None` if the packed stream had no style section.
    pub style: Option<Vec<u32>>,
    /// Text ids.
    pub text: Vec<u32>,
    /// Per-frame codec codes, in frame order.
    pub frames: Vec<Vec<usize>>,
}

impl OwnedSpeechRecord {
    /// Borrow this record as a [`SpeechRecord`], for re-packing.
    pub fn as_record(&self) -> SpeechRecord<'_> {
        SpeechRecord {
            speaker: self.speaker.as_deref(),
            style: self.style.as_deref(),
            text: &self.text,
            frames: &self.frames,
        }
    }
}
