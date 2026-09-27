//! Kokoro's text front end: misaki English G2P (US for `a*` voices, GB for
//! `b*`), reference Kokoro's `KPipeline` chunking so long text splits at the
//! same boundaries, and the phoneme-character → vocab-id mapping
//! `KModel.forward` applies.

mod chunk;
mod dialect;
mod ids;
mod text_front;

pub use dialect::Dialect;
pub use text_front::{KokoroG2p, TextChunk};

pub(crate) use ids::phonemes_to_input_ids;
