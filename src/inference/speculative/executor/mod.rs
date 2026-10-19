//! The speculative generator: start, step and the generation loop.

pub(crate) mod accept;
pub mod generate;
pub mod source;
pub mod step;

pub use generate::SpeculativeGenerator;
pub use source::DraftSource;
pub use step::StepOutput;
