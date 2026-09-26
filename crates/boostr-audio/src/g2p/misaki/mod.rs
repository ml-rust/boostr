//! Port of misaki 0.9.4's English G2P (`misaki.en`, Apache-2.0,
//! <https://github.com/hexgrad/misaki>), the G2P Kokoro-82M was trained on.
//!
//! Input is tagged tokens, as misaki builds them from spaCy; the tagger and
//! the out-of-vocabulary fallback are separate components. Output matches
//! `misaki.en.G2P(trf=False, british=..., fallback=None, unk=...)` for the
//! same tokens. The US and GB gold and silver lexicons are embedded
//! zlib-compressed; see `assets/misaki/build_lexicon.py`.

mod chars;
mod data;
pub mod g2p;
mod lexicon;
mod num2words;
mod number;
mod resolve;
mod retokenize;
mod special;
mod stem;
mod stress;
mod subtokenize;
pub mod token;

pub use g2p::{DEFAULT_UNK, G2pOutput, KOKORO_UNK, MisakiG2p, OovFallback};
pub use token::MToken;
