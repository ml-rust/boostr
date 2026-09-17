//! Unit A of the VoxCPM2 end-to-end orchestrator: reference-audio encoding
//! and the deterministic two-LM prefill, for one row or a left-padded batch.
//!
//! ```text
//! ref_wav_16k -> pad to patch_size*640 -> AudioVAE -> fold -> [T_ref, 4, 64]
//!             -> reference prefix + prompt (SequenceLayout), per row
//!             -> left-pad every row to S_max (PaddedBatch)
//!             -> base_lm.prefill(kv_start) -> fsq blend -> lm_hidden
//!             -> residual_lm.prefill(kv_start)          -> residual_hidden
//! ```
//!
//! The per-patch sampling loop, the stop logic and the VAE decode path live
//! in `generate` and `decode`.
//!
//! # Left padding
//!
//! Rows of different length share one `[B, S_max, ...]` tensor set. Each
//! row's pad sits in FRONT, so row `S_max - 1` is the last real position of
//! every row and one scalar `position` counter drives every row's cache.
//! `base_lm`'s RoPE is shift-invariant (attention reads only position
//! differences) and `residual_lm` is NoPE, so a row's values do not depend
//! on how far right it was shifted. Pad positions carry zero embeddings,
//! zero masks and the filler token id, and `kv_start[b]` keeps every real
//! query from attending them.
//!
//! # Traps this module exists to get right
//!
//! - `lm_hidden` is the LAST row of the post-blend `enc_outputs`. In BOTH
//!   modes that row is a TEXT position, so the blend left it UN-fsq'd. Do
//!   NOT apply `fsq` to it again.
//! - `feat_encoder` runs over ALL `S` rows, text rows included, whose patches
//!   are zeros. Skipping them is not an optimization; it changes the result.
//! - `fusion_concat_proj`'s argument is `cat(enc_outputs, audio_mask *
//!   feat_embed)` in THAT order. Swapping the halves of a 4096-wide concat is
//!   shape-valid and silently computes a different model.
//! - `enc_outputs` is NOT masked again inside the fusion. Only `feat_embed`
//!   is.
//! - Both KV caches come back primed to `current_length == S_max`. The later
//!   sampling loop advances ONE shared position counter, starting at
//!   [`PrefillState::position`] (`== S_max`), across both caches.
//! - A batch of one row, or a batch whose rows all have the same length,
//!   carries `kv_start == None` and runs the exact unpadded kernels.
//!
//! # Layout
//!
//! - `state`: [`PrefillState`], [`PrefillIntermediates`], [`PrefillRow`]
//! - `rows`: [`PaddedBatch`], the host-side padded ids and masks
//! - `encode`: [`VoxCpm2Model::encode_reference`]
//! - `run`: the prefill entry points and the shared body
//! - `tensors`: the body's tensor helpers
//!
//! [`VoxCpm2Model::encode_reference`]: super::VoxCpm2Model::encode_reference

mod encode;
mod rows;
mod run;
mod state;
mod tensors;

pub use rows::PaddedBatch;
pub use state::{PrefillIntermediates, PrefillRow, PrefillState};
