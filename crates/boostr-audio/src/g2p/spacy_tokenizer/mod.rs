//! Port of the spaCy 3.8 tokenizer (`spacy/tokenizer.pyx`, MIT,
//! <https://github.com/explosion/spaCy>) with the settings of the
//! `en_core_web_sm` 3.8.0 pipeline, the tokenizer misaki reads its tokens
//! and whitespace from.
//!
//! The patterns, special cases and norm tables are embedded; see
//! `assets/spacy_en/build_tokenizer.py`.

pub(crate) mod data;
pub mod hash;
pub(crate) mod lexeme;
pub(crate) mod patterns;
pub(crate) mod specials;
pub mod token;
pub mod tokenizer;

pub use hash::hash_string;
pub use token::SpacyToken;
pub use tokenizer::SpacyTokenizer;
