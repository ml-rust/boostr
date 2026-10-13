//! Decision models: answers from option logits, temperature fits, requests
//! built on the host, and the run over one shared prefix and many rows.

pub mod answer;
pub mod calibrate;
pub mod confidence;
pub mod isolated;
pub mod noul;
pub mod objective;
pub mod request;
pub mod run;
pub mod temperature;
pub mod types;

pub use answer::{AnswerOptions, SUM_TOLERANCE, answer_from_probs, combine_isolated};
pub use calibrate::{FitOutcome, TemperatureBounds, TemperatureFit, fit_temperature};
pub use confidence::{ScoreConfidence, certainty, choice_confidence, score_confidence};
pub use isolated::{IsolatedSample, fit_temperature_isolated, fit_temperature_mixed};
pub use noul::{NoulBand, NoulMap};
pub use objective::{CalibrationSample, Target};
pub use request::{
    DecisionRequest, GatherIndexed, RowSlot, merge_requests, plan_from_rows, split_result,
};
pub use run::{
    CancelToken, DecisionBackbone, DecisionHead, DecisionPlan, Gather, OptionLogits, PrefixInput,
    PrefixState, RunnerConfig, run_decision,
};
pub use temperature::{TemperaturePolicy, tempered_softmax};
pub use types::{Answer, DecisionResult, QuestionKind, QuestionMeta};
