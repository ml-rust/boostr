//! GGUF metadata → [`EncoderConfig`](super::EncoderConfig) parsing, one module
//! per architecture namespace.

mod bert;
mod dispatch;
mod gemma;
mod nomic;
mod qwen3;
