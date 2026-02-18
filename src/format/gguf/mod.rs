pub mod metadata;
pub mod reader;
pub mod types;
pub mod value;

pub use metadata::GgufMetadata;
pub use reader::Gguf;
pub use types::{GgmlType, GgufValueType};
pub use value::GgufValue;

/// Tensor info entry from a GGUF file
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub shape: Vec<usize>,
    pub ggml_type: GgmlType,
    pub offset: u64,
}

impl GgufTensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_bytes(&self) -> usize {
        let numel = self.numel();
        let bs = self.ggml_type.block_size();
        let bb = self.ggml_type.block_bytes();
        numel.div_ceil(bs) * bb
    }
}
