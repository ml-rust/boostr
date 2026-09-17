//! Overlap-and-trim windowed decode for [`super::loader::VoxCpm2Model::decode_patches`]
//! and its streaming counterpart `decode_patches_from`.
//!
//! [`context`] derives the left context a window needs to reproduce the
//! whole-utterance decode; [`windowed`] runs the windows on a grid anchored
//! at absolute latent frame 0, so a whole decode and a suffix decode issue
//! the same decoder calls for every complete window.

mod context;
mod windowed;

#[cfg(test)]
mod test_decoder;

pub(crate) use context::{CONTEXT_FRAMES, WINDOW_FRAMES};
pub(crate) use windowed::{decode_latent_windowed, decode_latent_windowed_from};
