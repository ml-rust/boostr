//! The decision run over one shared prefix and many rows: the request
//! (`plan`), the model and scoring traits (`backbone`, `head`), rows into
//! passes of right-padded lanes (`packing`), the gathered hidden rows
//! (`gather`), the run (`runner`), and the tempered-softmax answer stage
//! (`finish`).

pub mod backbone;
pub(crate) mod finish;
pub(crate) mod gather;
pub mod head;
pub(crate) mod packing;
pub mod plan;
pub mod runner;

pub use backbone::{DecisionBackbone, PrefixState};
pub use head::{DecisionHead, OptionLogits};
pub use packing::{DEFAULT_MAX_LANES, DEFAULT_PASS_TOKENS, RunnerConfig};
pub use plan::{DecisionPlan, Gather, PrefixInput};
pub use runner::{prefill_prefix, run_decision};
