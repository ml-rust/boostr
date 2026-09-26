//! VITS text-to-speech model components.

pub mod spline;

pub use spline::{
    MIN_BIN_HEIGHT, MIN_BIN_WIDTH, MIN_DERIVATIVE, RqSplineClient,
    unconstrained_rational_quadratic_spline,
};
