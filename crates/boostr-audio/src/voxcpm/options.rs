//! Load-time and per-request settings for [`VoxCpm2Engine`](super::engine::VoxCpm2Engine).

use numr::dtype::DType;
use std::path::PathBuf;
use std::time::Duration;

/// Generation settings applied to every request. Defaults are the clone
/// pipeline's verified values.
#[derive(Debug, Clone)]
pub struct VoxCpm2SynthOptions {
    /// Flow-matching solver steps per patch.
    pub n_timesteps: usize,
    /// Classifier-free guidance scale.
    pub cfg_value: f32,
    /// Patches during which the stop token is ignored.
    pub min_len: usize,
    /// Base seed; every request draws from it, so equal requests render
    /// equal audio on one backend.
    pub seed: u64,
    /// Patches generated between two chunks handed to a streaming sink.
    /// Each chunk costs one suffix decode, which re-decodes the containing
    /// window's left context, so a smaller value lowers latency to the first
    /// sample and raises decoder work per utterance. Applies to
    /// `synthesize_stream` only; `0` is rejected at render time.
    pub stream_chunk_patches: usize,
    /// How long the engine's owner thread keeps taking newly queued requests
    /// into the batch it is about to start, measured from the first request
    /// of that batch.
    ///
    /// The thread blocks on an empty queue, so an idle engine pays this only
    /// once a request has arrived, and the drain ends early when the batch
    /// reaches [`max_batch`](Self::max_batch) or the queue closes. A request
    /// that arrives alone and stays alone therefore starts at most this late
    /// and renders as a batch of one.
    pub batch_window: Duration,
    /// Most requests one batch generates together.
    ///
    /// Rounded DOWN to a width the generation loop runs at (1, 2, 4 or 8) and
    /// never below 1. Requests past the cap stay queued and open the next
    /// batch. Every row of a batch steps on every iteration, a finished row
    /// included, so a wider batch trades per-request latency for throughput.
    pub max_batch: usize,
    /// Whether the owner thread keeps ONE batch open and admits queued
    /// requests into its free slots between steps.
    ///
    /// Off (the default), a batch is built, run to completion and torn
    /// down, so a request arriving while one runs waits for that batch to
    /// drain — the whole of a long utterance, in the worst case. That wait
    /// is what this option removes, and it is the only thing it removes.
    ///
    /// On, the batch is persistent: a request arriving mid-generation
    /// prefills on the owner thread and joins on the NEXT step. The batch
    /// steps only the rows actually seated, rounded up to a width the Euler
    /// graph cache holds, so an idle slot costs no compute and a lone
    /// request is stepped as a batch of one, exactly as the off route steps
    /// it.
    ///
    /// What an idle slot does cost is MEMORY, and that is why this is off
    /// by default. Both caches are sized once at
    /// [`max_batch`](Self::max_batch) slots of
    /// [`continuous_capacity`](Self::continuous_capacity) positions and held
    /// for the engine's whole life, so `max_batch` is a standing
    /// reservation rather than a peak one — and it is per ENGINE, so a host
    /// serving several bundles pays it once per bundle. Turn this on where
    /// requests genuinely overlap and the reservation fits, and set
    /// `max_batch` to the concurrency actually served.
    ///
    /// A row renders the same samples either way: its noise draw follows its
    /// own emitted-patch count and its own seed, neither of which depends on
    /// when it joined or on what else shares the batch.
    pub continuous_batching: bool,
    /// Cache positions one slot of a persistent batch holds, when
    /// [`continuous_batching`](Self::continuous_batching) is on. Ignored
    /// otherwise.
    ///
    /// A batch that is torn down after every generation sizes its caches to
    /// the rows it actually holds. A persistent one is sized ONCE, before it
    /// knows what it will be asked for, so the budget is fixed here.
    ///
    /// One request spends `t_ref + 2 + text_len` positions on its prefill
    /// (zero-shot spends `text_len`, with no reference prefix) plus
    /// `text_len * 6 + 10` on its patches, and the lane table advances one
    /// position past the last patch. The default covers a reference
    /// recording of a few hundred patches together with a sentence of
    /// roughly a hundred tokens — an ordinary clone request, not the
    /// 4096-patch ceiling a single request is allowed to reach.
    ///
    /// Raising this is not free. Two caches — the base stack's and the
    /// residual stack's — each hold a K and a V tensor per layer, every one
    /// of them `max_batch` rows by this many positions, resident for the
    /// batch's whole life. Doubling either this value or
    /// [`max_batch`](Self::max_batch) doubles that.
    ///
    /// A request that needs more than a slot holds is REFUSED, with an error
    /// naming the positions it needs and the positions a slot has. It is
    /// never queued, never truncated, and never allowed to overrun the
    /// cache. The model's rope table caps this in turn: a position the table
    /// does not hold could be filled and never rotated, so a larger value is
    /// clamped to what the table holds.
    pub continuous_capacity: usize,
}

impl Default for VoxCpm2SynthOptions {
    fn default() -> Self {
        Self {
            n_timesteps: 10,
            cfg_value: 2.0,
            min_len: 2,
            seed: 0,
            stream_chunk_patches: 4,
            batch_window: Duration::from_millis(20),
            max_batch: 8,
            continuous_batching: false,
            continuous_capacity: 1024,
        }
    }
}

/// Everything [`VoxCpm2Engine::load`] needs beyond the checkpoint location:
/// the dtype to cast to, per-request generation defaults, and an optional
/// LoRA adapter to fold into the weights. Bundled into one struct because
/// `load` already takes five positional arguments (weights, audiovae,
/// voices_dir, device, client) — three more flat parameters would trip
/// clippy's too-many-arguments limit.
#[derive(Debug, Clone, Default)]
pub struct VoxCpm2LoadOptions {
    /// Casts every transformer-stack tensor; `None` keeps the checkpoint's
    /// own dtype (BF16).
    pub dtype: Option<DType>,
    /// Casts every AudioVAE DECODER tensor (conv weights/biases, `Snake`
    /// alphas, sr-cond embeds), independently of `dtype`. `None` keeps the
    /// checkpoint's own F32, verified against PyTorch fixtures at that
    /// dtype.
    ///
    /// The encoder is NOT affected: it always loads and runs at F32. It
    /// runs once per render, on a short reference clip, so its cost is
    /// negligible — but casting it changes the latent handed to the
    /// transformer stack, shifting the stack's own conditioning and
    /// therefore the whole generation, including where it stops.
    pub vae_decoder_dtype: Option<DType>,
    /// Per-request generation settings — see [`VoxCpm2SynthOptions`].
    pub synth: VoxCpm2SynthOptions,
    /// A LoRA adapter safetensors file, folded into the model's weights
    /// once, at load time. `None` loads the base model, unchanged.
    ///
    /// The adapter is folded in ONCE: an engine serves ONE adapted model for
    /// its lifetime. Per-request adapter switching is not offered — serving
    /// several adapters means loading several engines (a `--tts-model
    /// NAME=DIR` bundle per adapter, in blazr's terms), one adapter each.
    pub adapter: Option<PathBuf>,
}
