//! Sink for the cross-attention probabilities of selected decoder layers.

use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Per-layer sink for captured cross-attention probability tensors.
///
/// One slot per decoder layer. A requested layer's slot accumulates one
/// tensor per decoder forward call: the prefill call contributes one tensor
/// covering every prefix row, and each later incremental step contributes
/// one more row. A layer nobody asked for keeps `None` and costs nothing —
/// no clone, no tensor stored.
pub struct CrossAttentionCapture<R: Runtime> {
    pub(crate) slots: Vec<Option<Vec<Tensor<R>>>>,
}

impl<R: Runtime> CrossAttentionCapture<R> {
    /// Build a sink for a decoder with `num_layers` layers, recording only
    /// the layers named in `layers` (0-indexed). Callers validate that every
    /// index in `layers` is `< num_layers` before building this; an
    /// out-of-range index here is silently ignored.
    pub fn new(num_layers: usize, layers: &[usize]) -> Self {
        let mut slots: Vec<Option<Vec<Tensor<R>>>> = (0..num_layers).map(|_| None).collect();
        for &l in layers {
            if let Some(slot) = slots.get_mut(l) {
                *slot = Some(Vec::new());
            }
        }
        Self { slots }
    }
}
