// Shared infrastructure
pub mod comm_utils;
pub mod parallel_embedding;
pub mod tensor_parallel;

// Training distributed
pub mod training;

// Inference distributed (requires nexar)
#[cfg(feature = "distributed")]
pub mod inference;

// Re-exports — shared
pub use comm_utils::*;
pub use parallel_embedding::*;
pub use tensor_parallel::*;

// Re-exports — training (preserves backwards compat)
pub use training::*;
