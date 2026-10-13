pub mod autograd_attention;
pub mod autograd_biased_attention;
pub mod autograd_rope;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod impl_generic;
pub mod traits;
#[cfg(feature = "wgpu")]
pub mod wgpu;

pub use autograd_attention::var_flash_attention;
pub use autograd_biased_attention::{AttentionCausality, var_attention_with_bias};
pub use autograd_rope::{RopeVariant, attach_rope_backward};
pub use impl_generic::position::mrope_stream_selector;
pub use traits::architecture::moe::MoEActivation;
pub use traits::{
    AlibiOps, AttentionOps, AttnOutLayout, CalibrationOps, DeviceGrammarDfa, FlashAttentionOps,
    FusedFp8TrainingOps, FusedOptimizerOps, FusedQkvOps, GatedDeltaNetOps, GrammarDfaOps,
    Int4GroupSize, KvCacheOps, KvCacheQuantOps, KvQuantMode, KvSeqLen, MRopeOps, MlaOps, MoEOps,
    PagedAttentionOps, RoPEOps, RoPEPackedOps, SamplingOps, SamplingScratch, SpeculativeOps,
    VarLenAttentionOps,
};

// Re-export numr's TensorOps which bundles all operation traits
pub use numr::ops::traits::TensorOps;
