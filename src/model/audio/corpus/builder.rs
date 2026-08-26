//! [`SpeechCorpusBuilder`]: VAD, Whisper, NeuCodec and a text tokenizer held
//! together, driven one recording at a time.

use splintr::{AnyTokenizer, Tokenize};

use crate::error::{Error, Result};
use crate::model::audio::corpus::options::{
    CorpusOptions, TextTokenizer, check_max_speech_duration,
};
use crate::model::audio::corpus::utterance::{Utterance, pack_utterances_with_layout};
use crate::model::audio::neucodec::NeuCodecEncoder;
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::model::audio::vad::SileroVad;
use crate::model::audio::whisper_loader::WhisperBundle;
use crate::model::audio::whisper_transcribe::TranscribeOptions;
use crate::model::speech_lm::codec::CodecVocab;
use crate::model::speech_lm::layout::SpeechLayout;
use crate::model::speech_lm::layout::expressive_tts::{CODEBOOK_SIZE, ExpressiveTtsLayout};
use crate::model::speech_lm::vocab::SpeechVocab;
use numr::dtype::DType;
use numr::ops::MatmulOps;
use numr::runtime::Runtime;

/// The three models and the tokenizer a corpus run needs, plus the
/// [`SpeechLayout`] its packed stream is written in.
///
/// Built once and reused across every recording: each of the three checkpoints
/// costs a full safetensors load, and the layout is fixed the moment the
/// constructor is chosen — [`new`](Self::new) for boostr's native layout,
/// [`new_expressive_tts`](Self::new_expressive_tts) for the
/// `Multilingual-Expressive-TTS-1.7B` base.
pub struct SpeechCorpusBuilder<R: Runtime> {
    vad: SileroVad<R>,
    whisper: WhisperBundle<R>,
    codec: NeuCodecEncoder<R>,
    tokenizer: AnyTokenizer,
    layout: SpeechLayout,
}

/// The codec's token space, read off the encoder's own quantizer config.
///
/// `num_quantizers` codebooks of `levels.product()` codes each — `1 x 65_536`
/// for the published NeuCodec checkpoint. Reading it off the encoder rather
/// than hardcoding it means a checkpoint with a different FSQ grid produces a
/// matching vocabulary instead of a silently wrong one.
fn codec_vocab<R: Runtime<DType = DType>>(codec: &NeuCodecEncoder<R>) -> Result<CodecVocab> {
    let quantizer = codec.quantizer().config();
    let codebook_size =
        usize::try_from(quantizer.layer_config()?.codebook_size()).map_err(|_| {
            Error::ModelError {
                reason: "NeuCodec codebook size does not fit in usize".to_string(),
            }
        })?;
    CodecVocab::new(quantizer.num_quantizers, codebook_size)
}

impl<R: Runtime<DType = DType>> SpeechCorpusBuilder<R> {
    /// Take ownership of the loaded models and resolve the text tokenizer,
    /// packing into boostr's NATIVE layout.
    ///
    /// The [`SpeechVocab`] is derived here, never passed in: its text region is
    /// the resolved tokenizer's vocabulary size and its audio region comes from
    /// the encoder's quantizer configuration.
    ///
    /// Use [`new_expressive_tts`](Self::new_expressive_tts) instead when
    /// fine-tuning `Multilingual-Expressive-TTS-1.7B`: that base has its own
    /// trained layout, and this one would point every id at a different
    /// embedding row.
    pub fn new(
        vad: SileroVad<R>,
        whisper: WhisperBundle<R>,
        codec: NeuCodecEncoder<R>,
        tokenizer: TextTokenizer<'_>,
    ) -> Result<Self> {
        let tokenizer = tokenizer.resolve()?;
        let text_vocab_size = Tokenize::vocab_size(&tokenizer);
        let codec_vocab = codec_vocab(&codec)?;
        let vocab = SpeechVocab::with_default_specials(text_vocab_size, codec_vocab)?;

        Ok(Self {
            vad,
            whisper,
            codec,
            tokenizer,
            layout: SpeechLayout::Native(vocab),
        })
    }

    /// Same, but packing into the `Multilingual-Expressive-TTS-1.7B` layout.
    ///
    /// No [`SpeechVocab`] is derived: under this layout the vocabulary is the
    /// BASE's 217_208 ids and the audio ids are fixed at `151_670 + code`, so
    /// deriving one from the tokenizer would be a vocabulary nothing reads.
    ///
    /// `tokenizer` MUST be that checkpoint's own `tokenizer.json` — pass it as
    /// [`TextTokenizer::JsonFile`]. Nothing here can check that, because a
    /// tokenizer knows its size but not which checkpoint trained it; a
    /// different one silently retokenizes the corpus against the wrong
    /// embedding rows.
    ///
    /// The encoder is checked: this layout has ONE codebook of
    /// [`CODEBOOK_SIZE`] codes, so a NeuCodec checkpoint with any other
    /// quantizer shape is rejected here rather than at the first frame.
    pub fn new_expressive_tts(
        vad: SileroVad<R>,
        whisper: WhisperBundle<R>,
        codec: NeuCodecEncoder<R>,
        tokenizer: TextTokenizer<'_>,
    ) -> Result<Self> {
        let codec_vocab = codec_vocab(&codec)?;
        if codec_vocab.num_codebooks() != 1 || codec_vocab.codebook_size() != CODEBOOK_SIZE {
            return Err(Error::InvalidArgument {
                arg: "codec",
                reason: format!(
                    "this encoder has {} codebooks of {} codes, but the ExpressiveTTS layout \
                     holds exactly 1 codebook of {CODEBOOK_SIZE} codes (<|s_0|> .. <|s_65535|>); \
                     a differently-shaped codec cannot be expressed in it",
                    codec_vocab.num_codebooks(),
                    codec_vocab.codebook_size()
                ),
            });
        }
        let tokenizer = tokenizer.resolve()?;

        Ok(Self {
            vad,
            whisper,
            codec,
            tokenizer,
            layout: SpeechLayout::ExpressiveTts(ExpressiveTtsLayout::new()),
        })
    }

    /// The layout the packed stream is written against.
    ///
    /// Callers need it to write the corpus layout sidecar: a stream is only
    /// readable by a model whose embedding rows match
    /// [`SpeechLayout::total_size`].
    pub fn layout(&self) -> &SpeechLayout {
        &self.layout
    }

    /// The [`SpeechVocab`] behind a native layout, or `None` under
    /// [`SpeechLayout::ExpressiveTts`], which no `SpeechVocab` describes.
    pub fn vocab(&self) -> Option<&SpeechVocab> {
        self.layout.vocab()
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
    /// When `opts.speaker` is set, each kept transcript is prefixed with
    /// `"{speaker}: "` BEFORE tokenizing, so the prefix and the words share one
    /// tokenization — which is what the ExpressiveTTS base is trained on.
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

        // Enhancement runs over the WHOLE recording, before segmentation: the
        // gate needs the pauses between phrases to read a floor from, and one
        // gain per recording keeps the relative loudness between utterances
        // that per-clip normalization would erase. See `CorpusOptions::enhance`.
        let enhanced;
        let samples = match opts.enhance {
            Some(enhance_opts) => {
                let rate = u32::try_from(opts.sample_rate).map_err(|_| Error::InvalidArgument {
                    arg: "opts.sample_rate",
                    reason: format!("{} does not fit a u32 sample rate", opts.sample_rate),
                })?;
                let (out, report) =
                    crate::model::audio::enhance::enhance(samples, rate, enhance_opts)?;
                tracing::info!(
                    input_lufs = report.input_lufs,
                    output_lufs = report.output_lufs,
                    input_floor_dbfs = report.input_noise_floor_dbfs,
                    output_floor_dbfs = report.output_noise_floor_dbfs,
                    limiting_db = report.limiter_reduction_db,
                    reached_target = report.reached_target,
                    "enhanced recording before segmentation"
                );
                enhanced = out;
                enhanced.as_slice()
            }
            None => samples,
        };

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
            let text = match opts.speaker {
                Some(speaker) => format!("{speaker}: {text}"),
                None => text.to_string(),
            };
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
        pack_utterances_with_layout(&self.layout, utterances, opts)
    }
}
