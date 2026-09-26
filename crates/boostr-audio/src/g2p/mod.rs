//! Grapheme-to-phoneme (G2P) conversion.
//!
//! [`espeak`] wraps espeak-ng behind the `g2p` feature. [`misaki`] ports the
//! misaki English lexicon G2P behind the `misaki` feature. [`spacy_tokenizer`]
//! ports the spaCy `en_core_web_sm` tokenizer behind the `spacy-tokenizer`
//! feature, which `misaki` implies.

pub mod espeak;
#[cfg(feature = "misaki")]
pub mod misaki;
#[cfg(feature = "spacy-tokenizer")]
pub(crate) mod pystr;
#[cfg(feature = "spacy-tokenizer")]
pub mod spacy_tokenizer;

pub use espeak::{G2pError, Lang, Phonemizer, tokenize_ipa};
