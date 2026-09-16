//! LRU cache of captured Euler-loop graphs, one entry per [`EulerGraphKey`].
//!
//! Mirrors `crate::model::encoder::model::graph_cache::EncoderForwardCache`.
//! The one deliberate difference: [`EulerGraphCache::with_entry_or_capture`]
//! holds the entry lock across lookup, capture, AND replay. Stream capture is
//! not re-entrant on one stream, and every replay writes the entry's shared
//! input buffers, so two callers on one `LocalDit` must never overlap.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use numr::dtype::DType;
use numr::runtime::CapturedGraph;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

use crate::error::Result;

/// Distinct Euler configurations kept per `LocalDit`. Generation uses one
/// shape per run, so this bound is a leak guard, not a working-set size.
pub const EULER_GRAPH_CACHE_CAP: usize = 16;

/// Everything the captured graph bakes in. Two calls that agree on this key
/// differ only in the CONTENTS of `z`, `mu` and `cond`, which replay copies
/// into the entry's stable buffers.
///
/// `schedule` is the full `t_span` as bit patterns, not its length: the
/// per-step `t` scalars and the host-side `dt` recurrence are baked into
/// the graph, and the sway coefficient changes them at a fixed step count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EulerGraphKey {
    pub batch: usize,
    pub patch_size: usize,
    pub feat_dim: usize,
    pub mu_tokens: usize,
    pub schedule: Vec<u32>,
    pub cfg_bits: u32,
    pub use_cfg_zero_star: bool,
    pub dtype: DType,
}

/// One captured post-warmup Euler loop.
///
/// `inputs` order is fixed by the capture site: `z_buf`, `mu_buf`,
/// `cond_buf`, then every constant the graph reads (the zero `mu` half, the
/// zero `dt`, one `t` scalar per captured step). Constants are kept only so
/// `CapturedGraph` keeps their allocations alive. `outputs` is `[x_out_buf]`.
///
/// Drop order is delegated to `CapturedGraph` (graph before tensors); see
/// `EncoderForwardCache`'s doc for why that ordering matters to the driver.
/// Do not add raw device-pointer fields outside `captured`.
pub struct CapturedEuler {
    captured: CapturedGraph<CudaRuntime>,
}

impl CapturedEuler {
    pub fn new(captured: CapturedGraph<CudaRuntime>) -> Self {
        Self { captured }
    }

    pub fn launch(&self) -> numr::error::Result<()> {
        self.captured.launch()
    }

    /// Stable `[batch, patch_size, feat_dim]` buffer the loop starts from.
    pub fn z_buf(&self) -> &Tensor<CudaRuntime> {
        &self.captured.inputs()[0]
    }

    /// Stable `[batch, mu_tokens * hidden_dim]` conditional-`mu` buffer.
    pub fn mu_buf(&self) -> &Tensor<CudaRuntime> {
        &self.captured.inputs()[1]
    }

    /// Stable `[batch, patch_size, feat_dim]` prefix-condition buffer.
    pub fn cond_buf(&self) -> &Tensor<CudaRuntime> {
        &self.captured.inputs()[2]
    }

    /// Stable `[batch, patch_size, feat_dim]` buffer the graph's final D2D
    /// copy writes. Overwritten by every launch: callers copy out of it.
    pub fn x_out_buf(&self) -> &Tensor<CudaRuntime> {
        &self.captured.outputs()[0]
    }
}

struct CacheEntry {
    key: EulerGraphKey,
    captured: CapturedEuler,
    last_used: usize,
}

/// Thread-safe LRU cache of captured Euler loops, bounded to
/// [`EULER_GRAPH_CACHE_CAP`].
pub struct EulerGraphCache {
    entries: Mutex<Vec<CacheEntry>>,
    clock: AtomicUsize,
    capture_count: AtomicUsize,
}

impl EulerGraphCache {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::with_capacity(EULER_GRAPH_CACHE_CAP)),
            clock: AtomicUsize::new(0),
            capture_count: AtomicUsize::new(0),
        }
    }

    /// Captures performed since construction. Never decrements on eviction.
    pub fn capture_count(&self) -> usize {
        self.capture_count.load(Ordering::Relaxed)
    }

    /// Drop every captured graph. Required whenever the model's weights are
    /// swapped (a LoRA adapter attached or reloaded): the graphs bake the OLD
    /// weight addresses in, and would keep reading them after the swap.
    pub fn clear(&self) {
        let mut entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());
        entries.clear();
    }

    /// Run `use_entry` on the entry for `key`, capturing it first via
    /// `capture` when absent.
    ///
    /// The entry lock is held for the whole call, on purpose:
    ///
    /// - `capture` puts the client's stream into capture mode. A second
    ///   capture, or any other work on that stream, from another thread
    ///   would corrupt or abort it.
    /// - `use_entry` copies fresh inputs into the entry's shared buffers and
    ///   launches. A concurrent caller on the same entry would overwrite
    ///   those buffers before the launch reads them.
    ///
    /// A failed `capture` inserts nothing, so the next call retries it.
    pub fn with_entry_or_capture<T>(
        &self,
        key: &EulerGraphKey,
        capture: impl FnOnce() -> Result<CapturedEuler>,
        use_entry: impl FnOnce(&CapturedEuler) -> Result<T>,
    ) -> Result<T> {
        let tick = self.clock.fetch_add(1, Ordering::Relaxed);
        let mut entries = self.entries.lock().unwrap_or_else(|p| p.into_inner());

        let idx = match entries.iter().position(|e| &e.key == key) {
            Some(idx) => idx,
            None => {
                let captured = capture()?;
                self.capture_count.fetch_add(1, Ordering::Relaxed);
                if entries.len() >= EULER_GRAPH_CACHE_CAP {
                    let lru = entries
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, e)| e.last_used)
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    entries.swap_remove(lru);
                }
                entries.push(CacheEntry {
                    key: key.clone(),
                    captured,
                    last_used: tick,
                });
                entries.len() - 1
            }
        };

        entries[idx].last_used = tick;
        use_entry(&entries[idx].captured)
    }
}

impl Default for EulerGraphCache {
    fn default() -> Self {
        Self::new()
    }
}
