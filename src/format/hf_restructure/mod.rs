//! Put a Hugging Face SafeTensors checkpoint into the Llama tensor layout.
//! [`restructure_hf_checkpoint`] is the entry point.

mod block_edits;
mod experts;
mod fused;
mod restructure;
#[cfg(test)]
mod test_support;

pub(crate) use block_edits::{query_scale_fold, rotary_row_order};
pub use restructure::restructure_hf_checkpoint;
