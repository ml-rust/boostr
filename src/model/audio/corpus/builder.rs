//! [`SpeechCorpusBuilder`]: VAD, Whisper, NeuCodec and a text tokenizer held
//! together, driven one recording at a time.

use splintr::{AnyTokenizer, Tokenize};

use crate::error::{Error, Result};
use crate::model::audio::corpus::options::{
    CorpusOptions, TextTokenizer, check_max_speech_duration,
};
use crate::model::audio::corpus::utterance::{Utterance, pack_utterances};
use crate::model::audio::neucodec::NeuCodecEncoder;
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::model::audio::vad::SileroVad;
use crate::model::audio::whisper_loader::WhisperBundle;
use crate::model::audio::whisper_transcribe::TranscribeOptions;
use crate::model::speech_lm::codec::CodecVocab;
use crate::model::speech_lm::vocab::SpeechVocab;
use numr::dtype::DType;
use numr::ops::MatmulOps;
use numr::runtime::Runtime;

/// The three models and the tokenizer a corpus run needs, plus the
/// [`SpeechVocab`] derived from them.
///
/// Built once and reused across every recording: each of the three checkpoints
/// costs a full safetensors load, and the vocabulary they imply is fixed the
/// moment they are chosen.
pub struct SpeechCorpusBuilder<R: Runtime> {
    vad: SileroVad<R>,
    whisper: WhisperBundle<R>,
    codec: NeuCodecEncoder<R>,
    tokenizer: AnyTokenizer,
    vocab: SpeechVocab,
}

impl<R: Runtime<DType = DType>> SpeechCorpusBuilder<R> {
    /// Take ownership of the loaded models and resolve the text tokenizer.
    ///
    /// The [`SpeechVocab`] is derived here, never passed in: its text region is
    /// the resolved tokenizer's vocabulary size, and its audio region comes
    /// from the encoder's own quantizer configuration — `num_quantizers`
    /// codebooks of `levels.product()` codes each (`1 x 65_536` for the
    /// published NeuCodec checkpoint). Reading it off the encoder rather than
    /// hardcoding it means a checkpoint with a different FSQ grid produces a
    /// matching vocabulary instead of a silently wrong one.
    pub fn new(
        vad: SileroVad<R>,
        whisper: WhisperBundle<R>,
        codec: NeuCodecEncoder<R>,
        tokenizer: TextTokenizer<'_>,
    ) -> Result<Self> {
        let tokenizer = tokenizer.resolve()?;
        let text_vocab_size = Tokenize::vocab_size(&tokenizer);

        let quantizer = codec.quantizer().config();
        let codebook_size =
            usize::try_from(quantizer.layer_config()?.codebook_size()).map_err(|_| {
                Error::ModelError {
                    reason: "NeuCodec codebook size does not fit in usize".to_string(),
                }
            })?;
        let codec_vocab = CodecVocab::new(quantizer.num_quantizers, codebook_size)?;
        let vocab = SpeechVocab::with_default_specials(text_vocab_size, codec_vocab)?;

        Ok(Self {
            vad,
            whisper,
            codec,
            tokenizer,
            vocab,
        })
    }

    /// The vocabulary the packed stream is written against.
    ///
    /// Callers need it to write the corpus layout sidecar: a stream is only
    /// readable by a model whose embedding rows match
    /// [`SpeechVocab::total_size`].
    pub fn vocab(&self) -> &SpeechVocab {
        &self.vocab
    }

    /// The resolved base text tokenizer, for callers that must tokenize
    /// alongside this pipeline with the identical vocabulary.
    pub fn tokenizer(&self) -> &AnyTokenizer {
        &self.tokenizer
    }

    /// Segment, transcribe and codec-encode one decoded recording.
    ///
    /// `samples` must be mono at `opts.sample_rate`. Two classes of segment
    /// never reach the returned vector:
    ///
    /// - shorter than `opts.min_utterance_secs`, dropped BEFORE transcription,
    /// - transcribing to empty or whitespace-only text, dropped after it.
    ///
    /// The second count is reported through `tracing` at `debug` level, since a
    /// recording that is mostly music or noise drops nearly everything and the
    /// caller otherwise sees only a short list with no reason.
    pub fn utterances<C>(
        &self,
        client: &C,
        samples: &[f32],
        opts: &CorpusOptions<'_>,
    ) -> Result<Vec<Utterance>>
    where
        C: NeuCodecClient<R> + MatmulOps<R>,
        R::Client: NeuCodecClient<R>,
    {
        if opts.sample_rate == 0 {
            return Err(Error::InvalidArgument {
                arg: "opts.sample_rate",
                reason: "sample rate must be non-zero".to_string(),
            });
        }
        check_max_speech_duration(opts.vad.max_speech_duration_s)?;

        let segments = self.vad.speech_timestamps(client, samples, &opts.vad)?;

        let min_samples = if opts.min_utterance_secs > 0.0 {
            (f64::from(opts.min_utterance_secs) * opts.sample_rate as f64).ceil() as usize
        } else {
            0
        };

        let transcribe_opts = TranscribeOptions {
            language: opts.language,
            translate: false,
            max_new_tokens: opts.max_new_tokens,
        };

        let mut out = Vec::with_capacity(segments.len());
        let mut empty_transcripts = 0usize;
        for (i, segment) in segments.iter().enumerate() {
            if segment.start > segment.end || segment.end > samples.len() {
                return Err(Error::InvalidArgument {
                    arg: "samples",
                    reason: format!(
                        "segment {i} covers samples {}..{}, outside the {} samples given",
                        segment.start,
                        segment.end,
                        samples.len()
                    ),
                });
            }
            if segment.len() < min_samples {
                continue;
            }
            let slice = &samples[segment.start..segment.end];

            let transcription =
                self.whisper
                    .transcribe(client, slice, opts.sample_rate, &transcribe_opts)?;
            let text = transcription.text.trim();
            if text.is_empty() {
                empty_transcripts += 1;
                continue;
            }
            let text = text.to_string();
            let text_tokens = self.tokenizer.encode(&text);

            let frames = self.codec.encode_frames(client, slice, client.device())?;

            out.push(Utterance {
                segment: *segment,
                text,
                text_tokens,
                frames,
            });
        }

        if empty_transcripts > 0 {
            tracing::debug!(
                empty_transcripts,
                segments = segments.len(),
                kept = out.len(),
                "dropped segments that transcribed to no text"
            );
        }

        Ok(out)
    }

    /// Pack prepared utterances into the token stream, padding when
    /// `opts.pad_to_multiple` says to.
    pub fn pack(&self, utterances: &[Utterance], opts: &CorpusOptions<'_>) -> Result<Vec<u32>> {
        pack_utterances(&self.vocab, utterances, opts)
    }
}
