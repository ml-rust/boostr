//! [`NeuCodecEncoder::encode_frames`] — bridges the `[1, 1, T]` I32 FSQ index
//! tensor [`NeuCodecEncoder::encode`] returns into the `Vec<Vec<usize>>`
//! shape `speech_lm::pack::SpeechRecord::frames` expects: one `Vec` per
//! frame, holding that frame's per-codebook indices (length 1 for NeuCodec,
//! which has a single FSQ codebook).

use super::NeuCodecEncoder;
use crate::error::{Error, Result};
use crate::model::audio::neucodec::client::NeuCodecClient;
use numr::dtype::DType;
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> NeuCodecEncoder<R> {
    /// Encode `samples` and return per-frame codebook indices ready for
    /// [`crate::model::speech_lm::pack::SpeechRecord::frames`].
    ///
    /// Brings the index tensor to host ONCE. A negative index is a codec bug,
    /// not a valid code, so it is reported as [`Error::InvalidArgument`]
    /// rather than cast into a huge `usize`.
    pub fn encode_frames<C>(
        &self,
        client: &C,
        samples: &[f32],
        device: &R::Device,
    ) -> Result<Vec<Vec<usize>>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let indices = self.encode(client, samples, device)?;
        let flat: Vec<i32> = indices.contiguous().map_err(Error::Numr)?.to_vec();
        flat.into_iter()
            .enumerate()
            .map(|(frame, code)| {
                usize::try_from(code)
                    .map(|code| vec![code])
                    .map_err(|_| Error::InvalidArgument {
                        arg: "code",
                        reason: format!(
                            "NeuCodec emitted negative FSQ index {code} at frame {frame}"
                        ),
                    })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::model::audio::neucodec::encoder::{MAX_ENCODE_SAMPLES, NeuCodecEncoder};
    use crate::test_utils::{cpu_setup, neucodec_checkpoint};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    /// A real full pipeline (16-layer semantic conformer + BigCodec acoustic
    /// stack) is needed to exercise `encode`/`encode_frames` end to end —
    /// nothing in this crate synthesizes one, so these tests load the real
    /// checkpoint and skip (rather than fail) when it is not configured, the
    /// same convention `tests/neucodec_encoder_parity.rs` uses.
    fn require_encoder(device: &CpuDevice) -> Option<NeuCodecEncoder<CpuRuntime>> {
        let ckpt = neucodec_checkpoint()?;
        Some(
            NeuCodecEncoder::<CpuRuntime>::from_safetensors(&ckpt, device)
                .expect("load neucodec checkpoint"),
        )
    }

    /// The guard must not fire yet at the exact limit, and must fire one
    /// sample over it — checked via `encode_with_limit` against a small
    /// limit so the test stays fast (no 60 s encode).
    #[test]
    fn limit_guard_is_off_by_one_correct() {
        let (client, device) = cpu_setup();
        let Some(encoder) = require_encoder(&device) else {
            eprintln!("skipping: set NEUCODEC_CHECKPOINT or BOOSTR_MODELS_DIR");
            return;
        };

        let n = 640usize; // two 320-sample frames worth, well-formed input
        let samples = vec![0.0f32; n];

        // Passes at limit == n.
        let ok = encoder.encode_with_limit(&client, &samples, &device, n);
        assert!(
            ok.is_ok(),
            "limit == input length must not be refused: {ok:?}"
        );

        // Fails at limit == n - 1.
        let err = encoder
            .encode_with_limit(&client, &samples, &device, n - 1)
            .expect_err("limit < input length must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains(&n.to_string()),
            "message must name the actual sample count: {msg}"
        );
        assert!(
            msg.contains("utterance"),
            "message must say to split into utterances: {msg}"
        );
    }

    /// The default guard fires one sample over `MAX_ENCODE_SAMPLES`, and does
    /// so before any allocation — this must stay fast: it never reaches the
    /// model, so it does not need a checkpoint to run.
    #[test]
    fn default_limit_rejects_one_sample_over() {
        let (client, device) = cpu_setup();
        let Some(encoder) = require_encoder(&device) else {
            eprintln!("skipping: set NEUCODEC_CHECKPOINT or BOOSTR_MODELS_DIR");
            return;
        };

        let samples = vec![0.0f32; MAX_ENCODE_SAMPLES + 1];
        let err = encoder
            .encode(&client, &samples, &device)
            .expect_err("input over MAX_ENCODE_SAMPLES must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains(&(MAX_ENCODE_SAMPLES + 1).to_string()),
            "message must name the actual sample count: {msg}"
        );
        assert!(
            msg.contains("utterance"),
            "message must say to split into utterances: {msg}"
        );
    }

    /// `encode_frames` returns one `Vec` per frame, each holding NeuCodec's
    /// single codebook index, matching a direct read of `encode`'s tensor.
    #[test]
    fn encode_frames_matches_encode_tensor() {
        let (client, device) = cpu_setup();
        let Some(encoder) = require_encoder(&device) else {
            eprintln!("skipping: set NEUCODEC_CHECKPOINT or BOOSTR_MODELS_DIR");
            return;
        };

        // A short, well-formed input: a couple of alignment strides.
        let samples = vec![0.01f32; 640];

        let indices = encoder.encode(&client, &samples, &device).expect("encode");
        let want: Vec<i32> = indices.contiguous().expect("contiguous").to_vec();

        let frames = encoder
            .encode_frames(&client, &samples, &device)
            .expect("encode_frames");

        assert_eq!(frames.len(), want.len(), "one Vec per frame");
        for (frame, (got, &want)) in frames.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                got.len(),
                1,
                "NeuCodec has a single FSQ codebook (frame {frame})"
            );
            assert_eq!(got[0], want as usize, "frame {frame} value mismatch");
        }
    }
}
