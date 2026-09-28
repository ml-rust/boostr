//! Unit A of the VoxCPM2 end-to-end orchestrator: reference and prompt
//! audio encoding and the deterministic two-LM prefill, for one row or a
//! left-padded batch.
//!
//! ```text
//! ref_wav_16k    -> right-pad to patch_size*640 -> AudioVAE -> fold -> [T_ref, 4, 64]
//! prompt_wav_16k -> left-pad to patch_size*640  -> AudioVAE -> fold -> [T_prompt, 4, 64]
//!             -> reference prefix + text + continuation tail (SequenceLayout), per row
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
//! - `lm_hidden` is the LAST row of the post-blend `enc_outputs`. Without a
//!   continuation tail that row is a TEXT position, so the blend left it
//!   UN-fsq'd. With one it is an AUDIO position and the blend fsq'd it. Do
//!   NOT apply `fsq` to it again in either case.
//! - `prefix_feat_cond` is each row's last raw `audio_feat` patch: zeros
//!   without a tail, the prompt's last patch with one. It is sliced on the
//!   device and handed to the generation loop in [`PrefillState`].
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
//! - `encode`: [`VoxCpm2Model::encode_reference`], [`VoxCpm2Model::encode_prompt`]
//! - `entry`: the prefill entry points
//! - `run`: the shared body
//! - `tensors`: the body's tensor helpers
//!
//! [`VoxCpm2Model::encode_reference`]: super::VoxCpm2Model::encode_reference
//! [`VoxCpm2Model::encode_prompt`]: super::VoxCpm2Model::encode_prompt

mod encode;
mod entry;
mod rows;
mod run;
mod state;
mod tensors;

pub use rows::PaddedBatch;
pub use state::{PrefillIntermediates, PrefillRow, PrefillState};
