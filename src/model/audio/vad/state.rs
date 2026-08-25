//! Per-stream state for Silero VAD: the LSTM `(h, c)` and the sample context
//! the next chunk is prefixed with.

use crate::error::{Error, Result};
use crate::model::audio::vad::config::{HIDDEN_SIZE, VadConfig};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Streaming state for ONE audio stream.
///
/// Create one per stream and thread it through every
/// [`SileroVad::chunk_probability`](super::SileroVad::chunk_probability) call.
/// Dropping it mid-stream and starting a fresh one restarts the model, which is
/// a behavior change, not a no-op: the context is what lets the network see the
/// samples immediately before the current chunk.
pub struct VadState<R: Runtime> {
    pub(super) h: Tensor<R>,
    pub(super) c: Tensor<R>,
    pub(super) context: Vec<f32>,
}

impl<R: Runtime<DType = DType>> VadState<R> {
    /// Zero state: `h = c = 0` and a zero-filled context, matching a stream
    /// that has not seen any audio.
    pub fn new(config: &VadConfig, device: &R::Device) -> Result<Self> {
        let h = Tensor::<R>::zeros(&[1, HIDDEN_SIZE], DType::F32, device).map_err(Error::Numr)?;
        let c = Tensor::<R>::zeros(&[1, HIDDEN_SIZE], DType::F32, device).map_err(Error::Numr)?;
        Ok(Self {
            h,
            c,
            context: vec![0.0; config.context_samples],
        })
    }

    /// Return to the zero state, ending the current stream.
    pub fn reset(&mut self, config: &VadConfig, device: &R::Device) -> Result<()> {
        *self = Self::new(config, device)?;
        Ok(())
    }

    /// LSTM hidden state, `[1, 128]`.
    pub fn hidden(&self) -> &Tensor<R> {
        &self.h
    }

    /// LSTM cell state, `[1, 128]`.
    pub fn cell(&self) -> &Tensor<R> {
        &self.c
    }

    /// The samples that will be prefixed to the next chunk: the tail of the
    /// previous chunk, or zeros at the start of a stream.
    pub fn context(&self) -> &[f32] {
        &self.context
    }

    /// Prime the context directly, for resuming a stream whose earlier chunks
    /// were processed elsewhere. Normal streaming never needs this — the
    /// context advances on its own with every chunk.
    ///
    /// Errors when `samples` is not exactly `config.context_samples` long: a
    /// short or long context silently shifts every STFT frame.
    pub fn set_context(&mut self, samples: &[f32]) -> Result<()> {
        if samples.len() != self.context.len() {
            return Err(Error::InvalidArgument {
                arg: "samples",
                reason: format!(
                    "context must be exactly {} samples, got {}",
                    self.context.len(),
                    samples.len()
                ),
            });
        }
        self.context.copy_from_slice(samples);
        Ok(())
    }
}
