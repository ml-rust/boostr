//! Flat token layout shared by text and neural-audio-codec tokens.
//!
//! A text-to-speech LM is a causal decoder over ONE vocabulary: it reads text ids
//! and emits audio-codec ids. That only works if both live in a single flat id
//! space with a layout every downstream stage agrees on — embedding resize, loss
//! masking, sampling constraints, and audio decode all index into it.
//!
//! The layout is EXPLICIT and SERIALIZABLE. Control ids are stored as absolute
//! numbers, never derived from a position in a list, and the control region has a
//! size the caller reserves up front. Both properties exist so a layout can be
//! written next to a checkpoint and checked back against it: a vocabulary whose
//! layout is not recorded cannot be reloaded correctly.
//!
//! Nothing here is codec-specific. Every size is derived from [`CodecVocab`], so
//! a single-codebook codec (NeuCodec: 1 codebook x 65_536 entries, 50 frames/sec)
//! and a residual/interleaved codec (SNAC: 3-4 codebooks x 4096 entries, several
//! codes per frame) are both expressible without touching this file.
//!
//! Control tokens themselves, and the evidence for the set that exists, live in
//! [`super::special`].
//!
//! # Extension path: how each future addition lands
//!
//! The region order below is chosen so growth does not move existing ids. State
//! plainly what each kind of extension costs.
//!
//! - **A new control token.** Takes the next FREE reserved slot inside the control
//!   region. `audio_base` is unchanged, every audio id keeps its embedding row,
//!   and checkpoints trained before the addition stay valid. This is free.
//! - **A richer codec** (more codebooks, or a bigger codebook). NOT free. It is a
//!   new [`SpeechVocab`] over a different [`CodecVocab`]: every id at or after
//!   `audio_base` changes meaning, so it is a new layout and a NEW CHECKPOINT.
//!   There is no in-place upgrade — do not pretend otherwise.
//! - **A second modality** (say a video codec block). Appends AFTER the audio
//!   block. Text, control and audio ids are all untouched, so an audio-only
//!   checkpoint's rows stay valid and the new rows are appended to the embedding
//!   matrix.
//!
//! That last case is why control ids are reserved BEFORE audio rather than
//! appended after it. Appending a MODALITY after audio is fine — it is one more
//! contiguous block. Appending CONTROL tokens after audio, which Scicom's
//! checkpoint does, permanently fragments the layout: control ids end up split
//! across two disjoint ranges, and every mask, resize and range check has to carry
//! both forever.
//!
//! Pure logic: no tensors, no `Runtime`, no device code. Testable without weights.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

use super::codec::CodecVocab;
use super::special::{ALL_SPECIAL_TOKENS, DEFAULT_CONTROL_REGION, SpecialToken};

/// The flat id layout holding text, control, and audio tokens.
///
/// **The order of the three regions is load-bearing.** Downstream code depends on
/// it: embedding resize appends rows for control+audio to the pretrained matrix,
/// loss masking selects `>= audio_base` to score audio-only, and sampling
/// constrains logits to a contiguous audio slice. Reordering these regions
/// invalidates every checkpoint trained under the old layout.
///
/// ```text
/// [0, text_vocab)                                        text tokens
/// [text_vocab, text_vocab + control_region_size)         control tokens
/// [audio_base, audio_base + num_codebooks*codebook_size) audio tokens
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechVocab {
    text_vocab_size: usize,
    /// Ids RESERVED for control tokens, chosen by the caller — NOT the count of
    /// tokens currently defined.
    ///
    /// This is what stops a future control token from invalidating a trained
    /// checkpoint. `audio_base` is derived from this reservation, so reserving
    /// more slots than are defined lets a fourteenth (or fiftieth) control token
    /// take an unused reserved id WITHOUT moving a single audio id. Deriving the
    /// boundary from the number of defined specials instead would shift the whole
    /// audio region every time a control token is added.
    control_region_size: usize,
    /// Absolute id of each defined control token.
    ///
    /// Explicit and persisted: an id is never recomputed from a position, so the
    /// map read back from a layout file is the same map the checkpoint trained on.
    specials: BTreeMap<SpecialToken, u32>,
    codec: CodecVocab,
}

impl SpeechVocab {
    /// Build the layout from explicit control ids.
    ///
    /// This is the form to persist and reload. Every id is stated, so the layout
    /// is self-describing and checkable against a checkpoint.
    pub fn new(
        text_vocab_size: usize,
        control_region_size: usize,
        specials: BTreeMap<SpecialToken, u32>,
        codec: CodecVocab,
    ) -> Result<Self> {
        if text_vocab_size == 0 {
            return Err(Error::InvalidArgument {
                arg: "text_vocab_size",
                reason: "text vocabulary must be non-empty".to_string(),
            });
        }
        if control_region_size < specials.len() {
            return Err(Error::InvalidArgument {
                arg: "control_region_size",
                reason: format!(
                    "reserved region of {control_region_size} ids cannot hold {} control tokens",
                    specials.len()
                ),
            });
        }
        let audio_base = text_vocab_size
            .checked_add(control_region_size)
            .ok_or_else(|| Error::ModelError {
                reason: format!(
                    "{text_vocab_size} text + {control_region_size} control ids \
                     overflows the id space"
                ),
            })?;
        for (tok, id) in specials.iter() {
            let id = *id as usize;
            if id < text_vocab_size || id >= audio_base {
                return Err(Error::InvalidArgument {
                    arg: "specials",
                    reason: format!(
                        "control token {tok:?} has id {id} outside the reserved control region \
                         [{text_vocab_size}, {audio_base})"
                    ),
                });
            }
        }
        // Two tokens on one id would share an embedding row and split its gradient.
        for (tok, id) in specials.iter() {
            if specials.iter().any(|(o, oid)| o != tok && oid == id) {
                return Err(Error::InvalidArgument {
                    arg: "specials",
                    reason: format!("control token {tok:?} shares id {id} with another token"),
                });
            }
        }
        let total = audio_base
            .checked_add(codec.total_audio_tokens())
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or_else(|| Error::ModelError {
                reason: format!(
                    "vocabulary of {text_vocab_size} text + {control_region_size} control + {} \
                     audio tokens overflows the u32 id space",
                    codec.total_audio_tokens()
                ),
            })?;
        debug_assert!(total > 0);
        Ok(Self {
            text_vocab_size,
            control_region_size,
            specials,
            codec,
        })
    }

    /// Build the layout by assigning control ids sequentially from `text_vocab_size`.
    ///
    /// Ergonomic path for a fresh layout. The assignment happens ONCE, here; the
    /// resulting ids are STORED and become the source of truth. Reordering the
    /// slice afterwards cannot change a persisted layout's ids.
    pub fn with_sequential_specials(
        text_vocab_size: usize,
        control_region_size: usize,
        specials: &[SpecialToken],
        codec: CodecVocab,
    ) -> Result<Self> {
        let mut map = BTreeMap::new();
        for (i, tok) in specials.iter().enumerate() {
            let id = text_vocab_size
                .checked_add(i)
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| Error::ModelError {
                    reason: format!("control id for {tok:?} exceeds the u32 id space"),
                })?;
            if map.insert(*tok, id).is_some() {
                return Err(Error::InvalidArgument {
                    arg: "specials",
                    reason: format!("duplicate special token {tok:?} at index {i}"),
                });
            }
        }
        Self::new(text_vocab_size, control_region_size, map, codec)
    }

    /// The default layout: all 13 control tokens in canonical order, in a region
    /// of [`DEFAULT_CONTROL_REGION`] ids.
    ///
    /// Use this to build a fresh model. The remaining 627 reserved ids are the
    /// headroom that keeps a future control token from moving `audio_base`; see
    /// [`DEFAULT_CONTROL_REGION`] for why the region is that size.
    pub fn with_default_specials(text_vocab_size: usize, codec: CodecVocab) -> Result<Self> {
        Self::with_sequential_specials(
            text_vocab_size,
            DEFAULT_CONTROL_REGION,
            &ALL_SPECIAL_TOKENS,
            codec,
        )
    }

    /// The codec this layout was built for.
    pub fn codec(&self) -> &CodecVocab {
        &self.codec
    }

    /// Size of the text region. Text ids are `[0, text_vocab_size)`.
    ///
    /// Text ids keep their ORIGINAL values from the base tokenizer. The pretrained
    /// embedding matrix is indexed by id, so shifting text ids would point every
    /// token at another token's learned row and destroy the pretrained model.
    /// New regions are therefore only ever appended after it.
    pub fn text_vocab_size(&self) -> usize {
        self.text_vocab_size
    }

    /// Ids reserved for control tokens, defined or not. See the field docs.
    pub fn control_region_size(&self) -> usize {
        self.control_region_size
    }

    /// Number of control tokens actually defined, at most
    /// [`control_region_size`](Self::control_region_size).
    pub fn num_specials(&self) -> usize {
        self.specials.len()
    }

    /// First audio id. Audio occupies `[audio_base, total_size)`.
    ///
    /// Derived from the RESERVED region, not from the defined tokens, so adding a
    /// control token leaves every audio id where it is.
    pub fn audio_base(&self) -> usize {
        self.text_vocab_size + self.control_region_size
    }

    /// Total ids, i.e. the embedding/output-projection row count.
    pub fn total_size(&self) -> usize {
        self.audio_base() + self.codec.total_audio_tokens()
    }

    /// True if a checkpoint with `rows` embedding rows matches this layout.
    ///
    /// A loader MUST check this before using a layout. A mismatch means the
    /// checkpoint and the layout disagree about the id space, so every id past the
    /// first divergence points at the wrong embedding row; loading must abort
    /// rather than proceed with silently wrong tokens.
    pub fn matches_embedding_rows(&self, rows: usize) -> bool {
        rows == self.total_size()
    }

    /// Id of a control token, or `None` if it is not defined in this layout.
    pub fn special_id(&self, tok: SpecialToken) -> Option<u32> {
        self.specials.get(&tok).copied()
    }

    /// True if `id` sits in the reserved control region but no token claims it.
    ///
    /// These rows exist in the embedding matrix and are trained on nothing.
    pub fn is_reserved_unused(&self, id: u32) -> bool {
        self.is_special(id) && !self.specials.values().any(|claimed| *claimed == id)
    }

    /// Ids that MUST be suppressed in the logits before sampling.
    ///
    /// This is a correctness requirement, not hygiene. Under
    /// `tie_word_embeddings: true` the embedding matrix IS the output projection,
    /// so every reserved-but-undefined row is also an output logit — and those
    /// rows are never a training target, so nothing ever pushes their logit down.
    /// Untrained rows of a tied matrix are exactly the "glitch token" class
    /// described in *Fishing for Magikarp* (Land & Bartolo, EMNLP 2024): they are
    /// occasionally sampled with high probability, and the id they emit decodes to
    /// nothing at all. Suppressing them is the only reliable fix at inference time.
    ///
    /// Returned ids, ascending:
    /// - every id in the control region with no defined [`SpecialToken`], and
    /// - [`SpecialToken::SpeechPad`], which is padding for packed batches and is
    ///   defined but must never be generated.
    ///
    /// Text ids and audio ids are NOT here: both are trained targets, and
    /// restricting sampling to one of those regions is the caller's decision.
    pub fn sampling_forbidden_ids(&self) -> Vec<u32> {
        let mut out = Vec::new();
        for id in self.text_vocab_size..self.audio_base() {
            let Ok(id) = u32::try_from(id) else {
                break;
            };
            if self.is_reserved_unused(id) {
                out.push(id);
            }
        }
        if let Some(pad) = self.special_id(SpecialToken::SpeechPad) {
            match out.binary_search(&pad) {
                Ok(_) => {}
                Err(at) => out.insert(at, pad),
            }
        }
        out
    }

    /// Flat id for `code` in `codebook`, codebook-major.
    ///
    /// `audio_base + codebook * codebook_size + code`. Codebook-major keeps each
    /// codebook's ids contiguous, so a residual codec can mask logits to the one
    /// codebook legal at the current frame position with a single range.
    pub fn audio_token(&self, codebook: usize, code: usize) -> Result<u32> {
        if codebook >= self.codec.num_codebooks() {
            return Err(Error::InvalidArgument {
                arg: "codebook",
                reason: format!(
                    "codebook {codebook} out of range for {} codebooks",
                    self.codec.num_codebooks()
                ),
            });
        }
        if code >= self.codec.codebook_size() {
            return Err(Error::InvalidArgument {
                arg: "code",
                reason: format!(
                    "code {code} out of range for codebook size {}",
                    self.codec.codebook_size()
                ),
            });
        }
        let id = self.audio_base() + codebook * self.codec.codebook_size() + code;
        u32::try_from(id).map_err(|_| Error::ModelError {
            reason: format!("audio id {id} exceeds the u32 id space"),
        })
    }

    /// Inverse of [`audio_token`](Self::audio_token); `None` for non-audio ids.
    pub fn decode_audio_token(&self, id: u32) -> Option<(usize, usize)> {
        let id = id as usize;
        let offset = id.checked_sub(self.audio_base())?;
        if offset >= self.codec.total_audio_tokens() {
            return None;
        }
        Some((
            offset / self.codec.codebook_size(),
            offset % self.codec.codebook_size(),
        ))
    }

    /// True if `id` is a text token.
    pub fn is_text(&self, id: u32) -> bool {
        (id as usize) < self.text_vocab_size
    }

    /// True if `id` falls in the reserved control region.
    ///
    /// The whole reserved region answers true, including ids no token claims yet.
    /// Those rows exist in the embedding matrix and are neither text nor audio.
    pub fn is_special(&self, id: u32) -> bool {
        let id = id as usize;
        id >= self.text_vocab_size && id < self.audio_base()
    }

    /// True if `id` is an audio token.
    pub fn is_audio(&self, id: u32) -> bool {
        let id = id as usize;
        id >= self.audio_base() && id < self.total_size()
    }
}

#[cfg(test)]
#[path = "vocab_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "vocab_audio_tests.rs"]
mod audio_tests;
