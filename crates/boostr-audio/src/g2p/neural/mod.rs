//! [`NeuralG2pFallback`]: misaki's out-of-vocabulary hook backed by g2p_lm,
//! a char-level prefix-LM oxidizr trains. See `model.rs` for the sequence
//! format and inference path, and `assets/g2p_lm/build_dataset.py` for the
//! vocab and training data. `NeuralG2pFallback::from_dir` loads the model
//! from a checkpoint directory at run time; no weights are compiled in.

mod checkpoint;
mod config;
mod normalize;
mod vocab;

mod model;

pub use model::NeuralG2pFallback;
