//! Per-layer attention parameters.
//!
//! Gemma3-derived encoders alternate between two attention types on a fixed
//! block period, and the two types differ in *two* independent ways: the RoPE
//! base they rotate at, and the span of keys they may attend to.
//!
//! Both are derived here, together, from one predicate. That is deliberate:
//! the defect this type exists to prevent was a build path that honoured one
//! consequence of the local/global flag while silently dropping the other.
//! Anything that needs to know how a layer attends asks for a [`LayerAttention`]
//! and gets both answers at once.

/// How one transformer block attends.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerAttention {
    /// RoPE frequency base for this block.
    pub rope_freq_base: f32,
    /// Symmetric attention window, in positions. A query at position `p` may
    /// attend to keys in `[p - w/2, p + w/2]`. `None` means full attention over
    /// the sequence.
    ///
    /// This mirrors llama.cpp's `LLAMA_SWA_TYPE_SYMMETRIC`, which masks when
    /// `|p1 - p0| > n_swa / 2` — note the half-width: a `sliding_window` of 512
    /// permits a distance of 256, not 512.
    pub window: Option<usize>,
    /// Whether this block attends causally (a query may not see later keys).
    pub causal: bool,
}

impl LayerAttention {
    /// Largest position distance this block may attend across, if bounded.
    ///
    /// `None` means unbounded. The half-width is the value that actually
    /// governs masking; callers should use this rather than re-deriving it.
    pub fn max_distance(&self) -> Option<usize> {
        self.window.map(|w| w / 2)
    }

    /// Whether a query at `q_pos` may attend to a key at `k_pos`.
    pub fn attends(&self, q_pos: usize, k_pos: usize) -> bool {
        if self.causal && k_pos > q_pos {
            return false;
        }
        match self.max_distance() {
            Some(half) => q_pos.abs_diff(k_pos) <= half,
            None => true,
        }
    }
}
