//! Hadamard-rotated layers: the rotation basis, the rotated embedding, and the rotated MLP.

pub mod hadamard;
pub mod rotated_embedding;
pub mod rotated_mlp;

pub use hadamard::HadamardRotation;
pub use rotated_embedding::{MaybeRotatedEmbedding, RotatedEmbedding};
pub use rotated_mlp::{MlpProjection, RotatedMlp};
