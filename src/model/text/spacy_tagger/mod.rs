//! spaCy's CNN tagger (`spacy.Tok2Vec.v2` with `MultiHashEmbed.v2` and
//! `MaxoutWindowEncoder.v2`, then `spacy.Tagger.v2`), as `en_core_web_sm`
//! ships it.
//!
//! The input is one row of `u64` attribute IDs per token, as spaCy's
//! `FeatureExtractor` builds them. Turning token text into those IDs is the
//! text pipeline's job (`boostr_audio::g2p::spacy_tagger`).

pub mod config;
mod embed;
mod encoder;
pub mod hash;
mod maxout;
pub mod model;
pub mod weights;

pub use config::SpacyTaggerConfig;
pub use hash::{bucket_keys, murmurhash3_x86_128_u64};
pub use model::SpacyTagger;
pub use weights::{MaxoutWeights, SpacyTaggerWeights};
