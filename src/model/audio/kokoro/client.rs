//! [`KokoroClient`]: the client bound Kokoro's synthesis path composes
//! across every submodule it calls (ALBERT, text encoder, prosody
//! predictor, decoder, the seeded harmonic source, the excitation STFT and
//! the final iSTFT).

use crate::model::traits::ModelClient;
use numr::algorithm::fft::FftAlgorithms;
use numr::ops::{ComplexOps, RandomOps};
use numr::runtime::Runtime;

/// The ops Kokoro's full synthesis path needs on one backend.
///
/// [`ModelClient`] carries the convolution, normalization, matmul, type
/// conversion and utility ops. [`RandomOps`] draws the harmonic source's
/// phase and noise. [`ComplexOps`] and [`FftAlgorithms`] run the excitation
/// STFT and the output iSTFT. Every numr backend client (CPU, CUDA, WebGPU)
/// implements all of them, so the blanket impl covers each one.
pub trait KokoroClient<R: Runtime>:
    ModelClient<R> + RandomOps<R> + ComplexOps<R> + FftAlgorithms<R>
{
}

impl<R, C> KokoroClient<R> for C
where
    R: Runtime,
    C: ModelClient<R> + RandomOps<R> + ComplexOps<R> + FftAlgorithms<R>,
{
}
