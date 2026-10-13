//! The flat `kokoro::*` names, re-exported from each component module.

// Kokoro's AdaIN1d (holds fc + norm weights, uses (1+γ)·norm+β formula).
// Shadows `boostr::nn::AdaIn1d`, which is the generic utility variant with
// external gamma/beta.
pub use super::ada_layer_norm::AdaLayerNorm;
pub use super::adain_resblk1d::{AdainResBlk1d, PoolParams};
pub use super::adain_resblock::{AdaINResBlock1, AdaIn1d as KokoroAdaIn1d};
pub use super::bert::{AlbertConfig, AlbertEmbeddings, AlbertLayer, AlbertModel, BertEncoder};
pub use super::client::KokoroClient;
pub use super::decoder::Decoder;
pub use super::decoder_block::{DecoderBlock, UpsampleBlock};
pub use super::duration_predictor::{DurationPredictor, decode_durations, length_regulator};
pub use super::frame_predictor::{EnergyPredictor, FramePredictor, PitchPredictor};
pub use super::generator::{GeneratorStftParams, IStftNetGenerator, IStftNetGeneratorOpts};
pub use super::kokoro_v2::{KokoroModelV2, alignment_matrix_from_durations};
#[allow(deprecated)]
pub use super::loader::load_voice_style;
pub use super::loader::{
    load_bilstm, load_linear_tensors, load_lstm_direction, load_plain_conv1d, load_voice_pack,
    load_weight_norm_pair, load_weight_normed_conv1d,
};
pub use super::loader_v2::{
    AdainResBlk1dLoadOpts, AdainResBlock1LoadOpts, load_ada_layer_norm, load_adain_resblk1d,
    load_adain_resblock1, load_albert_embeddings, load_albert_layer, load_kokoro_adain,
    load_kokoro_full, load_kokoro_v2,
};
pub use super::mag_phase_head::MagPhaseHead;
pub use super::phoneme_vocab::KokoroPhonemeVocab;
pub use super::prosody_predictor::{
    DurationEncoder, ProsodyBranch, ProsodyPredictor, decode_prosody_durations,
};
pub use super::snake::snake;
pub use super::source_filter::{SineGen, SourceModuleHnNSF};
pub use super::style_projector::StyleProjector;
pub use super::synth_options::KokoroSynthOptions;
pub use super::text_encoder::{ConvBlock, TextEncoder, TextEncoderConfig};
pub use super::voice_style::{select_voice_style, split_voice_style};
pub use super::weight_source::KokoroWeightSource;
// `KokoroConfig` lives with the other model configs under
// `boostr::model::config::tts`; we re-export it here for convenience so the
// kokoro module is a one-stop import for callers.
pub use crate::model::config::KokoroConfig;
