//! Input-length guard and the always-fires right zero-padding arithmetic.

use crate::error::{Error, Result};
use crate::model::audio::neucodec::acoustic_encoder::encoder_hop_length;
use crate::model::audio::neucodec::fbank::SAMPLE_RATE;

/// Longest input [`NeuCodecEncoder::encode`](super::NeuCodecEncoder::encode)
/// accepts, in samples: 60 s at 16 kHz.
///
/// The semantic branch's self-attention (16 heads, 16 layers) builds a dense
/// `[1, heads, T, T]` score tensor per layer, so cost grows with the SQUARE
/// of the input length. At this limit `T` is ~3,000 frames (50 Hz), and one
/// layer's score tensor alone is `16 * 3,000 * 3,000 * 4` bytes ~= 576 MB. A
/// 26-minute clip (`T` ~ 80,000) would need on the order of a terabyte per
/// layer and cannot succeed — `encode` refuses it up front instead of dying
/// deep inside a matmul with no indication that input length was the cause.
pub const MAX_ENCODE_SAMPLES: usize = 60 * SAMPLE_RATE;

/// Check a waveform length against the encode limit before anything is
/// allocated or uploaded.
///
/// Split out as a free function so the refusal can be tested without a
/// checkpoint: it is the whole point of [`MAX_ENCODE_SAMPLES`], and a guard
/// whose only test skips when no model is present is not a tested guard.
pub fn check_encode_len(len: usize, max_samples: usize) -> Result<()> {
    if len == 0 {
        return Err(Error::InvalidArgument {
            arg: "samples",
            reason: "expected a non-empty 16 kHz mono waveform".to_string(),
        });
    }
    if len > max_samples {
        let seconds = len as f64 / SAMPLE_RATE as f64;
        let limit_seconds = max_samples as f64 / SAMPLE_RATE as f64;
        return Err(Error::InvalidArgument {
            arg: "samples",
            reason: format!(
                "{len} samples ({seconds:.1} s) exceeds the {max_samples}-sample \
                 ({limit_seconds:.1} s) encode limit: the semantic branch's attention \
                 cost is quadratic in input length; split the audio into shorter \
                 utterance clips and encode each separately"
            ),
        });
    }
    Ok(())
}

/// Number of samples the waveform must be a multiple of before encoding: the
/// acoustic encoder's total stride (product of `ENCODER_STRIDES`, 320), i.e.
/// the 16 kHz -> 50 Hz ratio.
pub fn encode_alignment() -> usize {
    encoder_hop_length()
}

/// Samples of right zero-padding the reference NeuCodec implementation
/// appends before encoding.
///
/// **This always returns a non-zero count.** The reference implementation computes
/// `pad = 320 - (T % 320)` unconditionally, so a length that is already a
/// multiple of 320 gets a FULL extra 320 samples appended (8000 -> 8320,
/// 8320 -> 8640). Do not "optimize" the exact-multiple case away — it changes
/// the frame count and therefore every emitted code index.
pub fn encode_padding(len: usize) -> usize {
    let stride = encode_alignment();
    stride - (len % stride)
}

#[cfg(test)]
mod tests {
    //! Tests that need no checkpoint: the padding arithmetic (the regression
    //! guard for the always-fires rule) and the length guard.

    use super::*;

    #[test]
    fn alignment_is_the_acoustic_stride() {
        assert_eq!(encode_alignment(), 320);
    }

    /// THE regression guard: an exact multiple of 320 still gets a FULL 320
    /// samples of padding. `pad = 320 - (T % 320)` is unconditional in the
    /// reference NeuCodec implementation.
    #[test]
    fn exact_multiple_still_pads_a_full_stride() {
        assert_eq!(encode_padding(8000), 320);
        assert_eq!(8000 + encode_padding(8000), 8320);
        assert_eq!(encode_padding(8320), 320);
        assert_eq!(encode_padding(320), 320);
    }

    #[test]
    fn partial_frame_pads_up_to_the_next_multiple() {
        assert_eq!(encode_padding(1), 319);
        assert_eq!(encode_padding(100), 220);
        assert_eq!(encode_padding(8321), 319);
        for len in [1usize, 100, 321, 8000, 8321, 16_000] {
            assert_eq!((len + encode_padding(len)) % encode_alignment(), 0);
        }
    }

    /// The refusal is the entire reason [`MAX_ENCODE_SAMPLES`] exists, so it is
    /// tested directly rather than through a model that needs a checkpoint.
    /// Without the guard, an over-long clip dies as an allocation failure inside a
    /// matmul, naming nothing the caller can act on.
    #[test]
    fn encode_len_guard_refuses_an_over_long_clip() {
        let over = MAX_ENCODE_SAMPLES + 1;
        let Err(err) = check_encode_len(over, MAX_ENCODE_SAMPLES) else {
            panic!("a clip one sample over the limit must be refused");
        };
        let msg = err.to_string();
        assert!(msg.contains(&over.to_string()), "{msg}");
        assert!(msg.contains("utterance"), "{msg}");
        // The corpus case: 26.8 minutes is ~80k frames of quadratic attention.
        assert!(check_encode_len(26 * 60 * SAMPLE_RATE, MAX_ENCODE_SAMPLES).is_err());
    }

    /// Exactly at the limit must pass — an off-by-one here would silently reject
    /// the longest legitimate utterance.
    #[test]
    fn encode_len_guard_accepts_exactly_the_limit() {
        assert!(check_encode_len(MAX_ENCODE_SAMPLES, MAX_ENCODE_SAMPLES).is_ok());
        assert!(check_encode_len(1, 1).is_ok());
        assert!(check_encode_len(2, 1).is_err());
    }

    /// Empty input is refused separately from the length limit, so a caller that
    /// hands over a zero-length decode result gets a distinct message.
    #[test]
    fn encode_len_guard_refuses_empty_input() {
        let Err(err) = check_encode_len(0, MAX_ENCODE_SAMPLES) else {
            panic!("an empty waveform must be refused");
        };
        assert!(err.to_string().contains("non-empty"), "{err}");
    }
}
