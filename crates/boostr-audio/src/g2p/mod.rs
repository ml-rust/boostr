//! Grapheme-to-phoneme (G2P) conversion.
//!
//! [`espeak`] wraps espeak-ng behind the `g2p` feature. [`misaki`] ports the
//! misaki English lexicon G2P behind the `misaki` feature.

pub mod espeak;
#[cfg(feature = "misaki")]
pub mod misaki;

pub use espeak::{G2pError, Lang, Phonemizer, tokenize_ipa};
