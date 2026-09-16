//! Kokoro `ProsodyPredictor` — duration + F0 + energy head.
//!
//! Replaces the speculative `DurationPredictor` / `FramePredictor` placeholders
//! with the reference Kokoro implementation's actual architecture:
//!
//! ```text
//! ProsodyPredictor
//! ├── text_encoder: DurationEncoder
//! │     └── lstms: [LSTM, AdaLayerNorm] × nlayers  (alternating)
//! ├── lstm:         BiLSTM(d_hid + style_dim, d_hid/2)
//! ├── duration_proj: Linear(d_hid, max_dur)       # 50-class classifier, NOT scalar
//! ├── shared:       BiLSTM(d_hid + style_dim, d_hid/2)
//! ├── F0:  [AdainResBlk1d × 3]                    # 512→512, 512→256 (upsample), 256→256
//! ├── N:   [AdainResBlk1d × 3]                    # same shape as F0
//! ├── F0_proj: Conv1d(d_hid/2, 1, 1)
//! └── N_proj:  Conv1d(d_hid/2, 1, 1)
//! ```
//!
//! Inference-only (no padding masks, no dropout). The generic
//! `DurationPredictor` / `FramePredictor` structs from earlier turns still
//! exist but are not used on the Kokoro path.

mod decode;
mod duration_encoder;
mod predictor;
#[cfg(test)]
mod test_support;

pub use decode::decode_prosody_durations;
pub use duration_encoder::DurationEncoder;
pub use predictor::{ProsodyBranch, ProsodyPredictor};
