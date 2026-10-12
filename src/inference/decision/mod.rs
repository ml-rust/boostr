//! Host-side math of decision models over read-back option logits.
//!
//! - `types`: question metadata, answer shapes, the per-request result
//! - `temperature`: temperature policies and the tempered softmax
//! - `confidence`: Choice and Score confidence rules
//! - `noul`: the Noul yes-probability map and the abstention band
//! - `answer`: one question's distribution to its answer
//! - `objective`: calibration samples and the temperature-scaling objective
//! - `calibrate`: fitting one temperature to labelled logits

pub mod answer;
pub mod calibrate;
pub mod confidence;
pub mod noul;
pub mod objective;
pub mod temperature;
pub mod types;

pub use answer::{AnswerOptions, SUM_TOLERANCE, answer_from_probs, combine_isolated};
pub use calibrate::{FitOutcome, TemperatureBounds, TemperatureFit, fit_temperature};
pub use confidence::{ScoreConfidence, choice_confidence, score_confidence};
pub use noul::{NoulBand, NoulMap};
pub use objective::{CalibrationSample, Target};
pub use temperature::{TemperaturePolicy, tempered_softmax};
pub use types::{Answer, DecisionResult, QuestionKind, QuestionMeta};
