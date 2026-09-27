//! [`NeuralG2pFallback`]: misaki's out-of-vocabulary hook backed by g2p_lm,
//! a char-level prefix-LM oxidizr trains. See `model.rs` for the sequence
//! format and inference path, `assets/g2p_lm/build_dataset.py` for the
//! vocab and training data, and `assets/g2p_lm/pack_model.py` for the
//! embedded bundle the `g2p-neural` feature compiles in.

mod checkpoint;
mod config;
mod normalize;
mod vocab;

#[cfg(feature = "g2p-neural")]
mod embedded;

mod model;

pub use model::NeuralG2pFallback;
