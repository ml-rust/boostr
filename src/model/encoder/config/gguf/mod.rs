//! GGUF metadata → [`EncoderConfig`](super::EncoderConfig) parsing, one module
//! per architecture namespace.

mod bert;
mod dispatch;
mod gemma;
mod jina_v2;
mod jina_v3;
mod nomic;
mod qwen3;
