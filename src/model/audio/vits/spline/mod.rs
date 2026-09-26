//! Rational-quadratic spline used by the VITS stochastic duration predictor.

mod eval;
mod knots;
mod rq;

pub use rq::{
    MIN_BIN_HEIGHT, MIN_BIN_WIDTH, MIN_DERIVATIVE, RqSplineClient,
    unconstrained_rational_quadratic_spline,
};
