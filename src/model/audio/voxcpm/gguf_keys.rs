//! GGUF metadata keys a VoxCPM2 single-file model carries beside its
//! tensors. compressr writes them; the GGUF loader and the weights
//! selector read them.

/// GGUF metadata string key holding the verbatim contents of the
/// checkpoint's `config.json`.
///
/// compressr writes it for a VoxCPM2 conversion whose input directory holds
/// a `config.json`. A GGUF written without it still loads through the
/// `config_json` path argument.
///
/// cstr's ggml-conventional file embeds no `config.json` either, so it too
/// needs the path argument. Its `voxcpm2.*` metadata keys do carry every
/// config value, but reading config out of GGUF metadata is its own unit.
pub const GGUF_CONFIG_JSON_KEY: &str = "voxcpm2.config_json";

/// GGUF metadata string key holding the verbatim contents of the
/// checkpoint's `tokenizer.json`: gguf-py's `Keys.Tokenizer.HF_JSON`.
///
/// compressr writes it for a VoxCPM2 conversion whose input directory holds
/// a `tokenizer.json`. [`VoxCpm2Weights::tokenizer_source`] reads it first
/// and falls back to a `tokenizer.json` beside the file when it is absent.
///
/// [`VoxCpm2Weights::tokenizer_source`]: super::VoxCpm2Weights::tokenizer_source
pub const GGUF_TOKENIZER_JSON_KEY: &str = "tokenizer.huggingface.json";
