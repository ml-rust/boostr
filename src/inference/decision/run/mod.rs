//! The decision run over one shared prefix and many rows: the request
//! (`plan`), the model and scoring traits (`backbone`, `head`), rows into
//! passes of right-padded lanes (`packing`), the stop flag (`cancel`), the
//! gathered hidden rows (`gather`), the run (`runner`), prefix states kept
//! across requests (`state_cache`), and the tempered-softmax answer stage
//! (`finish`).

pub mod backbone;
pub mod cancel;
pub(crate) mod finish;
pub(crate) mod gather;
pub mod head;
pub(crate) mod packing;
pub mod plan;
pub mod runner;
pub mod state_cache;

pub use backbone::{DecisionBackbone, PrefixState};
pub use cancel::CancelToken;
pub use head::{DecisionHead, OptionLogits};
pub use packing::{DEFAULT_MAX_LANES, DEFAULT_PASS_TOKENS, RunnerConfig, peak_lane_bytes};
pub use plan::{DecisionPlan, Gather, PrefixInput};
pub use runner::{prefill_prefix, run_decision};
pub use state_cache::{CachedPrefix, PrefixStateCache, StateCacheStats, cached_prefix};
