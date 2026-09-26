//! VITS text-to-speech model components.

pub mod attention;
pub mod dds_conv;
pub mod spline;
pub mod wavenet;

pub use attention::VitsAttention;
pub use dds_conv::VitsDilatedDepthSeparableConv;
pub use spline::{
    MIN_BIN_HEIGHT, MIN_BIN_WIDTH, MIN_DERIVATIVE, RqSplineClient,
    unconstrained_rational_quadratic_spline,
};
pub use wavenet::VitsWaveNet;
