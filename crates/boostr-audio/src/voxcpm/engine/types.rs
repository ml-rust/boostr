//! The engine struct, its per-voice state and the constants both the loader
//! and the request path use.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use splintr::AnyTokenizer;

use boostr::model::audio::voxcpm::model::{LoraAdapterReport, VoxCpm2Model};

use crate::voxcpm::options::VoxCpm2SynthOptions;

/// Sample rate the reference encoder expects.
pub(super) const REF_RATE: u32 = 16_000;

/// Hard cap on generated patches for one request, whatever the text length.
pub(super) const MAX_LEN_CAP: usize = 4096;

/// Containers `decode_audio` probes; a file with any other extension in the
/// voices directory is not a voice.
pub(super) const VOICE_EXTENSIONS: [&str; 4] = ["wav", "flac", "mp3", "ogg"];

/// Reserved voice id selecting zero-shot rendering: no reference recording,
/// no reference encode, `ref_feat = None` straight through to `prefill`.
///
/// A fixed literal, not an `Option<String>` layered on top of the render
/// path: an omitted voice must be an explicit, listable choice — present in
/// [`VoxCpm2Engine::voices`](crate::tts::TtsEngine::voices) like any other
/// id — never a silent fallback a caller has to infer. A literal also can't
/// collide with a voice file by accident: [`VoxCpm2Engine::load`] refuses
/// any file stemmed `zero-shot`.
pub const ZERO_SHOT_VOICE_ID: &str = "zero-shot";

/// A reference voice, encoded once.
pub(super) struct EncodedVoice<R: Runtime> {
    /// `[T_ref, feat_dim]` reference patches from `encode_reference`.
    pub(super) ref_feat: Tensor<R>,
}

/// VoxCPM2 voice-cloning engine on runtime `R`.
pub struct VoxCpm2Engine<R: Runtime<DType = DType>> {
    pub(super) model: VoxCpm2Model<R>,
    pub(super) client: Arc<R::Client>,
    pub(super) tokenizer: AnyTokenizer,
    pub(super) voices: BTreeMap<String, EncodedVoice<R>>,
    pub(super) options: VoxCpm2SynthOptions,
    /// The adapter this engine was loaded with, if
    /// [`VoxCpm2LoadOptions::adapter`](crate::voxcpm::VoxCpm2LoadOptions::adapter)
    /// was `Some`. Kept so a caller (e.g. blazr) can log
    /// rank/alpha/targets/counts via [`Self::adapter`] without reaching into
    /// the model.
    pub(super) adapter: Option<LoraAdapterReport>,
    /// Serialises renders. Holds no data; see the module docs.
    pub(super) render: Mutex<()>,
}

impl<R: Runtime<DType = DType>> VoxCpm2Engine<R> {
    /// The adapter this engine was loaded with, if
    /// [`VoxCpm2LoadOptions::adapter`](crate::voxcpm::VoxCpm2LoadOptions::adapter)
    /// was `Some` — rank, alpha, targets and tensor counts, for a caller to
    /// log without reaching into the model. `None` when the engine is
    /// serving the base (unadapted) model.
    pub fn adapter(&self) -> Option<&LoraAdapterReport> {
        self.adapter.as_ref()
    }
}
