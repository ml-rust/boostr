pub mod corpus;
pub mod decode;
pub mod enhance;
pub mod eval;
pub mod g2p;
pub mod kokoro;
pub mod mel;
pub mod neucodec;
pub mod pitch;
pub mod quality;
pub mod reflection_pad;
pub mod resample;
pub mod stft;
pub mod tts_bundle;
pub mod vad;
pub mod wav_decode;
pub mod wav_encode;
pub(crate) mod wav_format;
pub mod whisper;
pub mod whisper_decoder;
pub mod whisper_loader;
pub mod whisper_model;
pub mod whisper_transcribe;

pub use corpus::{
    CorpusOptions, MAX_UTTERANCE_SECS, PRETRAINED_TOKENIZER_NAMES, SpeechCorpusBuilder,
    TextTokenizer, Utterance, check_max_speech_duration, pack_utterances,
    pack_utterances_with_layout,
};
pub use decode::{decode_audio, decode_audio_file_mono_at, decode_audio_mono_at, extension_hint};
pub use eval::{ErrorRate, align, character_error_rate, normalize, total, word_error_rate};
pub use g2p::{G2pError, Lang, Phonemizer};
pub use mel::{
    LogSpec, MelNorm, MelOptions, MelScale, compute_mel_spectrogram, compute_mel_spectrogram_with,
};
pub use pitch::{PitchOptions, PitchTrack, estimate_pitch};
pub use quality::{TakeQuality, measure_quality};
pub use reflection_pad::reflection_pad_1d;
pub use resample::{
    DEFAULT_TAPS_PER_PHASE, MAX_FILTER_TAPS, resample, resample_with_taps, to_mono_at_rate,
};
pub use stft::{StftClient, StftOptions, stft};
pub use tts_bundle::{SynthesizeOptions, TtsBundle, TtsError, Voice, default_kokoro_voices};
pub use vad::{
    SileroVad, SileroVadWeights, SpeechSegment, VadConfig, VadSegmentOptions, VadState,
    segments_from_probabilities,
};
pub use wav_decode::{WavData, decode_wav, to_mono};
pub use wav_encode::{encode_pcm16_raw, encode_wav_f32, encode_wav_pcm16};
pub use whisper::WhisperEncoder;
pub use whisper_decoder::{DecoderCache, DecoderLayerCache, WhisperDecoder, WhisperDecoderLayer};
pub use whisper_loader::{WhisperBundle, WhisperGenerationConfig};
pub use whisper_model::{GenerateOptions, WhisperModel};
pub use whisper_transcribe::{TranscribeOptions, Transcription};
