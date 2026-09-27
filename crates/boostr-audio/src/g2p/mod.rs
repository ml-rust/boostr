//! Grapheme-to-phoneme (G2P) conversion.
//!
//! [`misaki`] ports the misaki English lexicon G2P behind the `misaki`
//! feature. [`spacy_tokenizer`] ports the spaCy `en_core_web_sm` tokenizer
//! behind the `spacy-tokenizer` feature. [`spacy_tagger`] runs its
//! `tok2vec` + `tagger` behind the `spacy-tagger` feature. `misaki` implies
//! both. [`neural`] is [`misaki::OovFallback`] backed by g2p_lm, a
//! char-level model, behind the `misaki` feature (weights load from disk)
//! or `g2p-neural` (weights embedded).

#[cfg(feature = "misaki")]
pub mod misaki;
#[cfg(feature = "misaki")]
pub mod neural;
#[cfg(feature = "spacy-tokenizer")]
pub(crate) mod pystr;
#[cfg(feature = "spacy-tagger")]
pub mod spacy_tagger;
#[cfg(feature = "spacy-tokenizer")]
pub mod spacy_tokenizer;
