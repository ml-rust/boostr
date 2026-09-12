//! [`NeuCodecEncoder`] — the full "16 kHz waveform in, FSQ code indices out"
//! half of NeuCodec, wiring the already-ported pieces into the reference
//! implementation's `NeuCodec.encode_code`.
//!
//! ```text
//! samples [T] @ 16 kHz
//!   -> right zero-pad to a multiple of 320          [1, 1, Tp]
//!   -> semantic: fbank -> SemanticEncoder -> ᵀ -> SemanticAdapter   [1, 1024, Ts]
//!   -> acoustic: AcousticEncoder                                    [1, 1024, Ta]
//!   -> truncate BOTH to min(Ts, Ta) = T
//!   -> cat([semantic, acoustic], dim = 1)           [1, 2048, T]
//!   -> fc_prior (applied on [1, T, 2048])           [1, 2048, T]
//!   -> ResidualFsq::encode -> indices [1, T, 1] -> permute  [1, 1, T]
//! ```
//!
//! Two steps below look like bugs and are not; both are documented at their
//! call sites and verified against the reference implementation: the padding ALWAYS fires (even
//! when the length is already a multiple of 320), and the two branches produce
//! different frame counts that are TRUNCATED, never interpolated or aligned.
//!
//! - `limits`: the input-length guard and the always-fires padding arithmetic
//! - `model`: [`NeuCodecEncoder`], its weights bundle, construction and loading
//! - `encode`: the encode pipeline and its per-branch intermediates
//! - `axes`: rank-3 layout helpers (axis swap, time-axis truncation)
//! - `frames`: [`NeuCodecEncoder::encode_frames`]

mod axes;
mod encode;
mod frames;
mod limits;
mod model;

pub use encode::EncodeStages;
pub use limits::{MAX_ENCODE_SAMPLES, check_encode_len, encode_alignment, encode_padding};
pub use model::{NeuCodecEncoder, NeuCodecEncoderWeights, PRIOR_DIM};
