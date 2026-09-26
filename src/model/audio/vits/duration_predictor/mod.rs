//! VITS duration predictors: exact ports of HF
//! `transformers.models.vits.modeling_vits.VitsStochasticDurationPredictor`
//! and `VitsDurationPredictor`.

mod deterministic;
mod stochastic;

pub use deterministic::VitsDurationPredictor;
pub use stochastic::VitsStochasticDurationPredictor;
