//! Decision models: host math over read-back option logits, and the run
//! that drives a backbone over one shared prefix and many rows.
//!
//! - `types`, `temperature`, `confidence`, `noul`, `answer`: logits to answers
//! - `objective`, `calibrate`: fitting one temperature to labelled logits
//! - `run`: the plan, backbone and head traits, packing, and the runner

pub mod answer;
pub mod calibrate;
pub mod confidence;
pub mod noul;
pub mod objective;
pub mod run;
pub mod temperature;
pub mod types;

pub use answer::{AnswerOptions, SUM_TOLERANCE, answer_from_probs, combine_isolated};
pub use calibrate::{FitOutcome, TemperatureBounds, TemperatureFit, fit_temperature};
pub use confidence::{ScoreConfidence, choice_confidence, score_confidence};
pub use noul::{NoulBand, NoulMap};
pub use objective::{CalibrationSample, Target};
pub use run::{
    DecisionBackbone, DecisionHead, DecisionPlan, Gather, OptionLogits, PrefixInput, PrefixState,
    RunnerConfig, run_decision,
};
pub use temperature::{TemperaturePolicy, tempered_softmax};
pub use types::{Answer, DecisionResult, QuestionKind, QuestionMeta};
