//! Kokoro TTS model components (StyleTTS2-lite derivative).
//!
//! Inference-only. Training path is not planned — Kokoro weights ship
//! pre-trained from upstream.

pub mod ada_layer_norm;
pub mod adain_resblk1d;
pub mod adain_resblock;
pub mod bert;
pub mod decoder;
pub mod decoder_block;
pub mod duration_predictor;
pub mod engine;
pub mod frame_predictor;
pub mod generator;
pub mod istft;
pub mod kokoro_v2;
pub mod loader;
pub mod loader_v2;
pub mod mag_phase_head;
pub mod model;
pub mod phoneme_vocab;
pub mod prosody_predictor;
pub mod snake;
pub mod source_filter;
pub mod style_projector;
pub mod text_encoder;
pub mod voice;
pub mod weight_source;

// `KokoroConfig` lives with the other model configs under
// `boostr::model::config::tts`; we re-export it here for convenience so the
// kokoro module is a one-stop import for callers.
pub use crate::model::config::KokoroConfig;
// Kokoro's AdaIN1d (holds fc + norm weights, uses (1+γ)·norm+β formula).
// Shadows `boostr::nn::AdaIn1d`, which is the generic utility variant with
// external gamma/beta.
pub use ada_layer_norm::AdaLayerNorm;
pub use adain_resblk1d::{AdainResBlk1d, PoolParams};
pub use adain_resblock::{AdaINResBlock1, AdaIn1d as KokoroAdaIn1d};
pub use bert::{AlbertConfig, AlbertEmbeddings, AlbertLayer, AlbertModel, BertEncoder};
pub use decoder::Decoder;
pub use decoder_block::{DecoderBlock, UpsampleBlock};
pub use duration_predictor::{DurationPredictor, decode_durations, length_regulator};
pub use engine::KokoroEngine;
pub use frame_predictor::{EnergyPredictor, FramePredictor, PitchPredictor};
pub use generator::{GeneratorStftParams, IStftNetGenerator, IStftNetGeneratorOpts};
pub use istft::{IStftOptions, istft};
pub use kokoro_v2::{KokoroModelV2, alignment_matrix_from_durations};
#[allow(deprecated)]
pub use loader::load_voice_style;
pub use loader::{
    load_bilstm, load_kokoro, load_linear_tensors, load_lstm_direction, load_plain_conv1d,
    load_voice_pack, load_weight_norm_pair, load_weight_normed_conv1d,
};
pub use loader_v2::{
    AdainResBlk1dLoadOpts, AdainResBlock1LoadOpts, load_ada_layer_norm, load_adain_resblk1d,
    load_adain_resblock1, load_albert_embeddings, load_albert_layer, load_kokoro_adain,
    load_kokoro_full, load_kokoro_v2,
};
pub use mag_phase_head::MagPhaseHead;
pub use phoneme_vocab::KokoroPhonemeVocab;
// `KokoroModel`, `DecoderBlockStyle`, `DecoderStage` live in `model.rs` but
// are intentionally NOT re-exported — that file is speculative scaffolding
// (see its module docstring) and the real assembly will land alongside the
// ALBERT backbone port. Import directly from
// `boostr::model::audio::kokoro::model` if you need to read the old code.
pub use model::{SynthesisControls, hann_window};
pub use prosody_predictor::{
    DurationEncoder, ProsodyBranch, ProsodyPredictor, decode_prosody_durations,
};
pub use snake::snake;
pub use source_filter::{SineGen, SourceModuleHnNSF};
pub use style_projector::StyleProjector;
pub use text_encoder::{ConvBlock, TextEncoder, TextEncoderConfig};
pub use voice::{VoiceResolver, resolve_and_load, select_voice_style, split_voice_style};
pub use weight_source::KokoroWeightSource;
