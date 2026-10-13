//! Generate stage of the VoxCPM2 end-to-end orchestrator: the per-patch generation
//! loop that drives a finished [`PrefillState`](super::PrefillState) — one row or a
//! left-padded batch — until every row has hit a stop token or its cap.
//!
//! The VAE decode and the wav wrapper are a LATER unit: this module returns patches and
//! nothing else. `step` documents one iteration and the traps this module exists to get
//! right. `validate` documents the one device read.
//!
//! `types`: public types and construction. `validate`: shape checks and the stop decision
//! every entry point shares. `step`: one iteration and the noise draw. `run`: the whole-run
//! driver. `capture`: the capturing variant and its shared inner body. `slots`: slot seating
//! (`open`/`admit`/`release`). `teacher_forced`: the batched training-time counterpart.

mod capture;
mod prelude;
mod run;
mod slots;
mod step;
mod teacher_forced;
#[cfg(test)]
pub(crate) mod test_support;
mod types;
mod validate;

pub use capture::StepIntermediates;
pub use teacher_forced::TeacherForcedConditioning;
pub(crate) use types::STOP_CLASS;
pub use types::{
    GenerateOptions, GenerateOutcome, GenerateState, PatchGenerator, RowOptions, StepOutcome,
};
