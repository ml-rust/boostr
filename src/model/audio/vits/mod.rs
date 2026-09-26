//! VITS text-to-speech model components.

pub mod attention;
pub mod dds_conv;
pub mod encoder;
pub mod encoder_layer;
pub mod feed_forward;
pub mod spline;
pub mod text_encoder;
pub mod wavenet;

pub use attention::VitsAttention;
pub use dds_conv::VitsDilatedDepthSeparableConv;
pub use encoder::VitsEncoder;
pub use encoder_layer::VitsEncoderLayer;
pub use feed_forward::VitsFeedForward;
pub use spline::{
    MIN_BIN_HEIGHT, MIN_BIN_WIDTH, MIN_DERIVATIVE, RqSplineClient,
    unconstrained_rational_quadratic_spline,
};
pub use text_encoder::{VitsTextEncoder, VitsTextEncoderOutput};
pub use wavenet::VitsWaveNet;
