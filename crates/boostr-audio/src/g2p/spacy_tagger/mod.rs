//! The `tok2vec` + `tagger` pipeline of spaCy's `en_core_web_sm` 3.8.0:
//! Penn Treebank tags for [`SpacyToken`](super::spacy_tokenizer::SpacyToken)s.
//!
//! [`attrs`] turns token strings into spaCy's `StringStore` IDs; the network
//! (`boostr::model::text::spacy_tagger`) hashes those IDs into its embedding
//! tables and runs on numr's CPU backend. The weights are embedded; see
//! `assets/spacy_en/convert_tagger.py`.

pub mod attrs;
pub(crate) mod data;
pub mod tagger;

pub use attrs::{Attr, StringIds};
pub use tagger::EnglishTagger;
