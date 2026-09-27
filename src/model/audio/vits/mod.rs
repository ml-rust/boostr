//! VITS text-to-speech model components.

pub mod attention;
pub mod coupling;
pub mod dds_conv;
pub mod duration_predictor;
pub mod encoder;
pub mod encoder_layer;
pub mod feed_forward;
pub mod flows;
pub mod hifigan;
mod hifigan_blocks;
mod layer_norm_channels;
pub mod length_regulator;
pub mod model;
pub mod spline;
pub mod text_encoder;
pub mod wavenet;
mod weight_norm_conv;

pub use attention::VitsAttention;
pub use coupling::{
    VitsResidualCouplingBlock, VitsResidualCouplingLayer, sample_prior_latents,
    sample_prior_latents_with_noise,
};
pub use dds_conv::VitsDilatedDepthSeparableConv;
pub use duration_predictor::{VitsDurationPredictor, VitsStochasticDurationPredictor};
pub use encoder::VitsEncoder;
pub use encoder_layer::VitsEncoderLayer;
pub use feed_forward::VitsFeedForward;
pub use flows::{VitsConvFlow, VitsElementwiseAffine};
pub use hifigan::VitsHifiGan;
pub use length_regulator::{LengthRegulated, LengthRegulatorClient, regulate_length};
pub use model::{VitsModel, VitsModelClient, VitsSynthOptions, VitsSynthesis};
pub use spline::{
    MIN_BIN_HEIGHT, MIN_BIN_WIDTH, MIN_DERIVATIVE, RqSplineClient,
    unconstrained_rational_quadratic_spline,
};
pub use text_encoder::{VitsTextEncoder, VitsTextEncoderOutput};
pub use wavenet::VitsWaveNet;
