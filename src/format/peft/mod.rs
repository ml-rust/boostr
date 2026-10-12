//! PEFT LoRA adapters: `adapter_config.json` parsing, tensor-key
//! normalization, and loading `adapter_model.safetensors` into per-module
//! A/B pairs.

pub mod config;
pub mod load;
pub mod names;
pub mod paths;
pub mod patterns;

pub use config::{PeftConfig, TargetModules};
pub use load::{LoraPair, PeftAdapter};
pub use names::{AdapterKey, LoraPart, parse_adapter_key};
pub use paths::{ADAPTER_CONFIG_FILE, ADAPTER_WEIGHTS_FILE};
pub use patterns::PatternOverride;
