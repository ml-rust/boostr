//! Tier-3 Kokoro loader — reads a full `hexgrad/Kokoro-82M` checkpoint into a
//! [`KokoroModelV2`].
//!
//! Depends on the tier-1 helpers in [`super::loader`] (`load_plain_conv1d`,
//! `load_weight_normed_conv1d`, `load_bilstm`, `load_linear_tensors`,
//! `load_weight_norm_pair`). Builds up tier-2 helpers for each Kokoro
//! primitive (`AdaIn1d`, `AdaLayerNorm`, `AdainResBlk1d`, `AdaINResBlock1`,
//! `AlbertLayer`, ALBERT embeddings, …) in [`primitives`], then tier-3
//! [`full::load_kokoro_v2`] stitches everything under the confirmed checkpoint
//! prefixes.
//!
//! Checkpoint prefix map:
//!
//! ```text
//! bert.*                 bert_encoder.*
//! text_encoder.*
//! predictor.{text_encoder,lstm,duration_proj,shared,F0,N,F0_proj,N_proj}
//! decoder.{asr_res,F0_conv,N_conv,encode,decode.{0..3},generator}
//! ```

mod full;
mod primitives;

pub use full::{load_kokoro_full, load_kokoro_v2};
pub use primitives::{
    AdainResBlk1dLoadOpts, AdainResBlock1LoadOpts, load_ada_layer_norm, load_adain_resblk1d,
    load_adain_resblock1, load_albert_embeddings, load_albert_layer, load_kokoro_adain,
};
