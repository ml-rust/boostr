//! NeuCodec's semantic branch: a 16-layer Wav2Vec2-BERT conformer encoder.

pub mod attention;
pub mod config;
pub mod conv_module;
pub mod encoder;
pub mod feature_projection;
pub mod layer;

pub use attention::{
    SemanticSelfAttention, SemanticSelfAttentionWeights, relative_distance_index_tensor,
    relative_distance_indices,
};
pub use config::SemanticEncoderConfig;
pub use conv_module::{ConvolutionModule, ConvolutionModuleWeights, causal_padding};
pub use encoder::{SemanticEncoder, SemanticEncoderWeights};
pub use feature_projection::{FeatureProjection, FeatureProjectionWeights};
pub use layer::{
    FFN_RESIDUAL_SCALE, SemanticEncoderLayer, SemanticEncoderLayerWeights, SemanticFeedForward,
};
