pub mod autograd_attention;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod impl_generic;
pub mod traits;
#[cfg(feature = "wgpu")]
pub mod wgpu;

pub use autograd_attention::var_flash_attention;
pub use traits::AlibiOps;
pub use traits::AttentionOps;
pub use traits::CalibrationOps;
pub use traits::FlashAttentionOps;
pub use traits::FusedFp8TrainingOps;
pub use traits::FusedOptimizerOps;
pub use traits::FusedQkvOps;
pub use traits::KvCacheOps;
pub use traits::MlaOps;
pub use traits::MoEOps;
pub use traits::PagedAttentionOps;
pub use traits::RoPEOps;
pub use traits::SamplingOps;
pub use traits::SpeculativeOps;
pub use traits::VarLenAttentionOps;
pub use traits::architecture::moe::MoEActivation;
pub use traits::{Int4GroupSize, KvCacheQuantOps, KvQuantMode};

// Re-export numr's TensorOps which bundles all operation traits
pub use numr::ops::traits::TensorOps;
