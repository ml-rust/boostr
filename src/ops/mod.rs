pub mod autograd_attention;
pub mod autograd_biased_attention;
pub mod autograd_gated_delta_net;
pub mod autograd_rope;
pub mod autograd_varlen_attention;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod impl_generic;
pub mod traits;
#[cfg(feature = "wgpu")]
pub mod wgpu;

pub use autograd_attention::var_flash_attention;
pub use autograd_biased_attention::{AttentionCausality, var_attention_with_bias};
pub use autograd_gated_delta_net::var_gdn_chunk;
pub use autograd_rope::{RopeVariant, attach_rope_backward};
pub use autograd_varlen_attention::var_varlen_attention;
pub use impl_generic::position::mrope_stream_selector;
pub use traits::architecture::moe::MoEActivation;
pub use traits::{
    AlibiOps, AttentionOps, AttnOutLayout, CalibrationOps, DeviceGrammarDfa, FilterRule,
    FlashAttentionOps, FusedFp8TrainingOps, FusedOptimizerOps, FusedQkvOps, GatedDeltaNetOps,
    GdnGrads, GrammarDfaOps, Int4GroupSize, KvCacheOps, KvCacheQuantOps, KvQuantMode, KvSeqLen,
    MRopeOps, MlaOps, MoEOps, PackedSegments, PagedAttentionOps, RoPEOps, RoPEPackedOps,
    SampleSeed, SamplingOps, SamplingProbsOps, SamplingScratch, SpeculativeOps, VarLenAttentionOps,
};

// Re-export numr's TensorOps which bundles all operation traits
pub use numr::ops::traits::TensorOps;
