//! Gated DeltaNet block for `qwen35`: [`GdnBlock`] (`layer`), stable-id
//! construction (`with_ids`), `VarBuilder` init and load (`init`), the
//! inference forward (`forward`, `prefill`, `gate_proj`, `graph_mode`), the
//! tracked training forward (`train`, `gate_proj_var`, `checkpointed`), head
//! order (`heads`), training switch (`trainable`), parameters (`params`) and
//! LoRA adapters (`adapter`, `gate_adapter`).

mod adapter;
mod checkpointed;
mod forward;
mod gate_adapter;
mod gate_proj;
mod gate_proj_var;
mod graph_mode;
mod heads;
mod init;
mod layer;
mod params;
mod prefill;
mod train;
mod trainable;
mod with_ids;

pub use adapter::GdnProjection;
pub use layer::{GdnBlock, GdnWeights};
pub use with_ids::GdnWeightsWithIds;
