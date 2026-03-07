//! Input types for multimodal model forward passes.

use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Input for multimodal model forward pass.
///
/// Distinguishes between text-only and multimodal inputs so the model
/// can skip encoder processing when no images/audio are present.
pub enum ModelInput<R: Runtime> {
    /// Text-only input (standard token IDs).
    TextOnly(Tensor<R>),
    /// Multimodal input with optional image/audio embeddings.
    Multimodal {
        /// Token IDs `[batch, seq_len]`
        input_ids: Tensor<R>,
        /// Image embeddings `[batch, num_image_tokens, hidden]` and insertion positions
        /// in the token sequence where image tokens should replace placeholder tokens.
        image_embeds: Option<(Tensor<R>, Vec<usize>)>,
        /// Audio embeddings `[batch, num_audio_tokens, hidden]` and insertion positions
        /// in the token sequence where audio tokens should replace placeholder tokens.
        audio_embeds: Box<Option<(Tensor<R>, Vec<usize>)>>,
    },
}
