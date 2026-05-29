//! CUDA graph capture cache for encoder forward passes.
//!
//! Keyed by `(batch_size, seq_len)`. On the first call for a given shape the
//! encoder forward is captured into a `CapturedGraph`. Subsequent calls with
//! the same shape replay the graph — collapsing ~192 kernel launches (24-layer
//! reranker) into a single `cuGraphLaunch` per forward.
//!
//! Only compiled when the `cuda` feature is active. Non-CUDA runtimes bypass
//! this module entirely; `Encoder` carries the cache only under `cfg(feature = "cuda")`.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use numr::runtime::CapturedGraph;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

/// Maximum number of distinct `(batch_size, seq_len)` shapes cached per encoder.
///
/// Workloads typically hit two shapes (reranker B=4 S≈500, embedder B=64 S≈128).
pub const GRAPH_CACHE_CAP: usize = 16;

/// A captured encoder forward for one `(batch_size, seq_len)` shape.
///
/// Wraps a [`CapturedGraph`] whose `inputs` slice is `[input_ids_buf, pos_ids_buf,
/// mask_buf]` and whose `outputs` slice is `[output_buf]`. The `CapturedGraph`
/// holds `Arc` clones of all four tensors, ensuring their device addresses remain
/// valid for the lifetime of the compiled graph.
///
/// On every replay:
/// 1. Copy fresh `input_ids` + `pos_ids` + `mask` host data into the input
///    buffers (H2D) — buffers are accessible via the typed accessor methods.
/// 2. `cuGraphLaunch` via [`CapturedForward::launch`] — replays all kernels in
///    one driver call.
/// 3. The output tensor (accessible via [`CapturedForward::output_buf`]) already
///    contains the result.
///
/// # Drop ordering
///
/// Drop ordering is **delegated to `CapturedGraph`**, which declares its fields
/// in the order `graph`, `inputs`, `outputs`. Rust drops struct fields in
/// **declaration order**, so the compiled graph handle is always destroyed before
/// the I/O tensors. This is **load-bearing**: on some NVIDIA driver versions
/// `cuGraphExecDestroy` dereferences the device pointers that were encoded into
/// the graph at capture time. The tensors must therefore still be alive (their
/// device memory still allocated) when the graph handle is destroyed.
///
/// The single `captured` field here means `CapturedForward` drops in one step,
/// preserving the invariant entirely inside `CapturedGraph`.
///
/// **DO NOT add fields that hold raw device pointers outside `captured`.**
pub struct CapturedForward {
    // Drop ordering is handled by CapturedGraph (graph drops before inputs/outputs).
    // See CapturedGraph doc for the NVIDIA driver rationale. DO NOT add raw
    // device-pointer fields outside this wrapper.
    captured: CapturedGraph<CudaRuntime>,
}

impl CapturedForward {
    /// Construct a `CapturedForward` from a `CapturedGraph`.
    ///
    /// The `CapturedGraph` must have been built with the I/O slice order:
    /// `inputs = [input_ids_buf, pos_ids_buf, mask_buf]`, `outputs = [output_buf]`.
    pub fn new(captured: CapturedGraph<CudaRuntime>) -> Self {
        Self { captured }
    }

    /// Replay the captured encoder forward (single `cuGraphLaunch`).
    pub fn launch(&self) -> numr::error::Result<()> {
        self.captured.launch()
    }

    /// Fixed-address `[B, S]` i64 input tensor for token ids (inputs\[0\]).
    pub fn input_ids_buf(&self) -> &Tensor<CudaRuntime> {
        &self.captured.inputs()[0]
    }

    /// Fixed-address `[B, S]` i64 input tensor for position ids (inputs\[1\]).
    pub fn pos_ids_buf(&self) -> &Tensor<CudaRuntime> {
        &self.captured.inputs()[1]
    }

    /// Fixed-address `[B, S]` f32 input tensor for attention mask (inputs\[2\]).
    pub fn mask_buf(&self) -> &Tensor<CudaRuntime> {
        &self.captured.inputs()[2]
    }

    /// Fixed-address `[B, hidden]` f32 output tensor (outputs\[0\]).
    pub fn output_buf(&self) -> &Tensor<CudaRuntime> {
        &self.captured.outputs()[0]
    }
}

struct CacheEntry {
    key: (usize, usize),
    captured: CapturedForward,
    last_used: usize,
}

/// Thread-safe LRU cache of captured encoder graphs, bounded to `GRAPH_CACHE_CAP`.
pub struct EncoderForwardCache {
    entries: Mutex<Vec<CacheEntry>>,
    clock: AtomicUsize,
    capture_count: AtomicUsize,
}

impl EncoderForwardCache {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::with_capacity(GRAPH_CACHE_CAP)),
            clock: AtomicUsize::new(0),
            capture_count: AtomicUsize::new(0),
        }
    }

    /// Total number of shapes captured (monotonically increasing; never decrements).
    pub fn capture_count(&self) -> usize {
        self.capture_count.load(Ordering::Relaxed)
    }

    /// Returns `true` if a captured graph exists for `(batch, seq)`.
    pub fn contains(&self, batch: usize, seq: usize) -> bool {
        let entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        entries.iter().any(|e| e.key == (batch, seq))
    }

    /// Call `f` with the `CapturedForward` for `(batch, seq)`.
    ///
    /// Returns `None` if not cached. Updates the LRU clock on hit.
    pub fn with_entry<F, T>(&self, batch: usize, seq: usize, f: F) -> Option<T>
    where
        F: FnOnce(&CapturedForward) -> T,
    {
        let tick = self.clock.fetch_add(1, Ordering::Relaxed);
        let mut entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        let idx = entries.iter().position(|e| e.key == (batch, seq))?;
        entries[idx].last_used = tick;
        Some(f(&entries[idx].captured))
    }

    /// Insert a new captured forward. Evicts the LRU entry when at capacity.
    pub fn insert(&self, batch: usize, seq: usize, captured: CapturedForward) {
        let tick = self.clock.fetch_add(1, Ordering::Relaxed);
        self.capture_count.fetch_add(1, Ordering::Relaxed);
        let mut entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        if entries.len() >= GRAPH_CACHE_CAP {
            let lru_idx = entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i)
                .unwrap_or(0);
            entries.swap_remove(lru_idx);
        }
        entries.push(CacheEntry {
            key: (batch, seq),
            captured,
            last_used: tick,
        });
    }
}

impl Default for EncoderForwardCache {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: inner Mutex serializes all access; AtomicUsize is inherently thread-safe.
unsafe impl Send for EncoderForwardCache {}
unsafe impl Sync for EncoderForwardCache {}
