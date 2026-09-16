//! The `Llama` model struct definition.

use crate::model::config::ModelConfig;
use crate::model::llama::model::blocks::LlamaBlock;
use crate::nn::{Embedding, MaybeQuantLinear, RmsNorm, RoPE};
use numr::runtime::Runtime;

/// Full LLaMA model
pub struct Llama<R: Runtime> {
    pub(in crate::model::llama::model) config: ModelConfig,
    pub(in crate::model::llama::model) embed_tokens: Embedding<R>,
    pub(in crate::model::llama::model) layers: Vec<LlamaBlock<R>>,
    pub(in crate::model::llama::model) norm: RmsNorm<R>,
    pub(in crate::model::llama::model) lm_head: MaybeQuantLinear<R>,
    pub(in crate::model::llama::model) rope: RoPE<R>,
}
