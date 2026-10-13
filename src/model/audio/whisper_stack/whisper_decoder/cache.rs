//! KV cache types for the Whisper decoder.

use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// KV cache for the full decoder, one entry per layer.
pub struct DecoderCache<R: Runtime> {
    pub layers: Vec<DecoderLayerCache<R>>,
}

/// Per-layer KV cache for the decoder.
///
/// - `self_k` / `self_v`: grow by one time-step each decoder step (`[B, H, T, D]`).
/// - `cross_k` / `cross_v`: computed once from the encoder output at decode
///   start and reused on every step (`[B, H, S, D]`).
pub struct DecoderLayerCache<R: Runtime> {
    pub self_k: Option<Tensor<R>>,
    pub self_v: Option<Tensor<R>>,
    pub cross_k: Option<Tensor<R>>,
    pub cross_v: Option<Tensor<R>>,
}

impl<R: Runtime> Default for DecoderLayerCache<R> {
    fn default() -> Self {
        Self {
            self_k: None,
            self_v: None,
            cross_k: None,
            cross_v: None,
        }
    }
}
