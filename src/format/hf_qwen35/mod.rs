//! Hugging Face Qwen3.5 SafeTensors checkpoints in the `qwen35` layout.
//!
//! - `rename`: Hugging Face tensor names to `qwen35` loader names
//! - `reorder`: [`reorder_v_heads`], grouped to tiled value-head order, and
//!   [`reorder_qkv_v_rows`], the same on the V rows of `in_proj_qkv`
//! - `edits`: the per-tensor edits of one layer
//! - `restructure`: [`restructure`], every name and layout edit on a `VarMap`

mod edits;
pub mod rename;
pub mod reorder;
pub mod restructure;

pub use rename::{is_dropped, qwen35_name};
pub use reorder::{reorder_qkv_v_rows, reorder_v_heads};
pub use restructure::restructure;
