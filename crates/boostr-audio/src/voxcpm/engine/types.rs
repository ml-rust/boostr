//! The engine struct, the state its owner thread shares with every caller,
//! the client-capability bundle the request path needs, and the constants
//! both the loader and the request path use.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::mpsc::Sender;
use std::thread::JoinHandle;

use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, HostCopyOps, IndexingOps, RandomOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use splintr::AnyTokenizer;

use boostr::model::audio::voxcpm::VoxCpmClient;
use boostr::model::audio::voxcpm::model::{LoraAdapterReport, VoxCpm2Model};
use boostr::quant::traits::DequantOps;

use crate::voxcpm::engine::batch::Job;
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

/// Every client capability the VoxCPM2 request path needs, as one bound.
///
/// A supertrait bundle with a blanket impl, so a client implementing the
/// parts implements this automatically and no caller names it.
pub trait VoxCpm2RenderClient<R: Runtime<DType = DType>>:
    VoxCpmClient<R>
    + TensorOps<R>
    + ScalarOps<R>
    + ReduceOps<R>
    + IndexingOps<R>
    + ShapeOps<R>
    + ActivationOps<R>
    + BinaryOps<R>
    + UnaryOps<R>
    + CompareOps<R>
    + ConditionalOps<R>
    + TypeConversionOps<R>
    + RandomOps<R>
    + HostCopyOps<R>
    + DequantOps<R>
    + Send
    + Sync
    + 'static
{
}

impl<R, C> VoxCpm2RenderClient<R> for C
where
    R: Runtime<DType = DType>,
    C: VoxCpmClient<R>
        + TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + ActivationOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + TypeConversionOps<R>
        + RandomOps<R>
        + HostCopyOps<R>
        + DequantOps<R>
        + Send
        + Sync
        + 'static,
{
}

/// A reference voice, encoded once.
pub(super) struct EncodedVoice<R: Runtime> {
    /// `[T_ref, patch_size, feat_dim]` reference patches from
    /// `encode_reference`.
    pub(super) ref_feat: Tensor<R>,
}

/// Everything a render reads, shared by every caller thread and the owner
/// thread behind an [`Arc`]. Immutable after [`VoxCpm2Engine::load`]: a
/// caller reads the voice table and the tokenizer to plan its request, the
/// owner thread reads the model and the client to run it.
pub(super) struct EngineInner<R: Runtime<DType = DType>> {
    pub(super) model: VoxCpm2Model<R>,
    pub(super) client: Arc<R::Client>,
    pub(super) tokenizer: AnyTokenizer,
    pub(super) voices: BTreeMap<String, EncodedVoice<R>>,
    pub(super) options: VoxCpm2SynthOptions,
    pub(super) adapter: Option<LoraAdapterReport>,
}

/// VoxCPM2 voice-cloning engine on runtime `R`.
pub struct VoxCpm2Engine<R: Runtime<DType = DType>> {
    pub(super) inner: Arc<EngineInner<R>>,
    /// Queue into the owner thread. `None` only while [`Drop`] closes it.
    pub(super) jobs: Option<Sender<Job>>,
    /// The owner thread. `None` only while [`Drop`] joins it.
    pub(super) worker: Option<JoinHandle<()>>,
}

impl<R: Runtime<DType = DType>> VoxCpm2Engine<R> {
    /// The adapter this engine was loaded with, if
    /// [`VoxCpm2LoadOptions::adapter`](crate::voxcpm::VoxCpm2LoadOptions::adapter)
    /// was `Some` — rank, alpha, targets and tensor counts, for a caller to
    /// log without reaching into the model. `None` when the engine is
    /// serving the base (unadapted) model.
    pub fn adapter(&self) -> Option<&LoraAdapterReport> {
        self.inner.adapter.as_ref()
    }
}

/// Close the queue, then join the owner thread: no request outlives the
/// engine and no thread is left detached. A thread that already died of a
/// panic joins with an error, which is ignored here — unwinding out of
/// `drop` would abort the process.
impl<R: Runtime<DType = DType>> Drop for VoxCpm2Engine<R> {
    fn drop(&mut self) {
        self.jobs.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
