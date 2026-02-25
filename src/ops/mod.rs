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
pub use traits::FlashAttentionOps;
pub use traits::FusedOptimizerOps;
pub use traits::KvCacheOps;
pub use traits::MlaOps;
pub use traits::PagedAttentionOps;
pub use traits::RoPEOps;
pub use traits::VarLenAttentionOps;
pub use traits::{Int4GroupSize, KvCacheQuantOps, KvQuantMode};
