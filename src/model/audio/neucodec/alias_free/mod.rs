//! Alias-free activation primitives for NeuCodec's acoustic encoder
//! (BigCodec lineage, ultimately StyleGAN3's anti-aliased nonlinearity).
//!
//! A pointwise nonlinearity applied at the signal's own rate creates harmonics
//! above Nyquist that fold back as aliasing. `Activation1d` avoids that by
//! upsampling ×2, applying the nonlinearity at the higher rate, and filtering
//! back down:
//!
//! ```text
//! x -> UpSample1d(2) -> SnakeBeta -> DownSample1d(2) -> y   (same length as x)
//! ```
//!
//! Both resamplers use a 12-tap Kaiser-windowed sinc. **The filter taps are NOT
//! in the checkpoint** — the reference NeuCodec implementation registers them
//! as non-persistent buffers, so
//! they must be recomputed here exactly, or every activation in the encoder is
//! subtly wrong.
//!
//! Everything is composed from tracked `var_*` ops, so the encoder stays
//! differentiable (needed for any future codec finetune) and backend-generic.

mod filter;
mod resample;
mod snake;

pub use filter::{kaiser_sinc_filter1d, replicate_pad_1d};
pub use resample::{Activation1d, DownSample1d, RESAMPLE_KERNEL_SIZE, RESAMPLE_RATIO, UpSample1d};
pub use snake::SnakeBeta;
