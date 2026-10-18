//! The speculative generator: start, step and the generation loop.

pub mod generate;
pub mod step;

pub use generate::SpeculativeGenerator;
pub use step::StepOutput;
