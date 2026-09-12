//! The per-chunk forward pass of [`SileroVad`]: window assembly, the STFT and
//! encoder stack, the LSTM step, and the sigmoid head.
//!
//! Split from [`super::model`] so the public surface (construction, streaming
//! entry points) reads separately from the graph transcription below. Every
//! function here is crate-private; callers reach it through
//! [`SileroVad::chunk_probability`].

use crate::error::{Error, Result};
use crate::model::audio::vad::config::HIDDEN_SIZE;
use crate::model::audio::vad::model::SileroVad;
use numr::dtype::DType;
use numr::ops::{ConvOps, PaddingMode, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> SileroVad<R> {
    /// Assemble the network's input window: `context ++ chunk`, then a
    /// reflection pad of `context_samples` on the tail.
    ///
    /// Silero's STFT front end is `nn.ReflectionPad1d((0, context))`, which
    /// the ONNX graph exports as a `Pad` node in "reflect" mode. Zero-padding
    /// here instead still runs and still looks plausible, but moves every
    /// probability by up to 0.2.
    pub(super) fn window(&self, context: &[f32], chunk: &[f32]) -> Result<Vec<f32>> {
        let context_len = self.config.context_samples;
        let body_len = context.len() + chunk.len();
        if body_len < context_len + 2 {
            return Err(Error::ModelError {
                reason: format!(
                    "a {body_len}-sample window is too short to reflect-pad by {context_len}"
                ),
            });
        }
        let mut buffer = Vec::with_capacity(self.config.window_samples());
        buffer.extend_from_slice(context);
        buffer.extend_from_slice(chunk);
        for k in 0..context_len {
            // PyTorch's reflect excludes the boundary sample, so the mirror
            // starts at body_len - 2, not body_len - 1.
            let mirrored = buffer[body_len - 2 - k];
            buffer.push(mirrored);
        }
        Ok(buffer)
    }

    /// STFT magnitude -> encoder -> one LSTM step. `input` is the already
    /// assembled `[1, 1, window_samples]` buffer.
    pub(super) fn forward_window<C>(
        &self,
        client: &C,
        input: &Tensor<R>,
        h: &Tensor<R>,
        c: &Tensor<R>,
    ) -> Result<(Tensor<R>, Tensor<R>)>
    where
        C: RuntimeClient<R> + TensorOps<R> + ConvOps<R>,
    {
        let bins = self.config.freq_bins();

        // STFT as a strided convolution against the stored basis. No bias.
        let spectrum = client
            .conv1d(
                input,
                &self.stft_basis,
                None,
                self.config.hop(),
                PaddingMode::Valid,
                1,
                1,
            )
            .map_err(Error::Numr)?;

        // Channels [0, bins) are real, [bins, 2*bins) imaginary.
        let real = spectrum
            .narrow(1, 0, bins)
            .map_err(Error::Numr)?
            .contiguous()
            .map_err(Error::Numr)?;
        let imag = spectrum
            .narrow(1, bins, bins)
            .map_err(Error::Numr)?
            .contiguous()
            .map_err(Error::Numr)?;
        let re2 = client.square(&real).map_err(Error::Numr)?;
        let im2 = client.square(&imag).map_err(Error::Numr)?;
        let power = client.add(&re2, &im2).map_err(Error::Numr)?;
        let mut x = client.sqrt(&power).map_err(Error::Numr)?;

        for conv in &self.encoder {
            x = conv.forward_inference(client, &x)?;
            x = client.relu(&x).map_err(Error::Numr)?;
        }

        // [1, 128, 1] -> [1, 128]: the encoder's stride schedule always
        // collapses the time axis to one frame for a single chunk.
        let time = x.shape()[2];
        if time != 1 {
            return Err(Error::ModelError {
                reason: format!("encoder produced {time} frames, expected 1"),
            });
        }
        let flat = x.reshape(&[1, HIDDEN_SIZE]).map_err(Error::Numr)?;

        self.rnn.step(client, &flat, h, c)
    }

    /// ReLU on the hidden state, a 1x1 conv down to one channel, then sigmoid.
    pub(super) fn head_probability<C>(&self, client: &C, h: &Tensor<R>) -> Result<f32>
    where
        C: RuntimeClient<R> + TensorOps<R> + ConvOps<R>,
    {
        // ReLU comes BEFORE the 1x1 conv, applied to the LSTM hidden state.
        let activated = client.relu(h).map_err(Error::Numr)?;
        let shaped = activated
            .reshape(&[1, HIDDEN_SIZE, 1])
            .map_err(Error::Numr)?;
        let logit = self.head.forward_inference(client, &shaped)?;
        let prob = client.sigmoid(&logit).map_err(Error::Numr)?;
        match prob.to_vec::<f32>().first() {
            Some(p) => Ok(*p),
            None => Err(Error::ModelError {
                reason: "VAD head produced an empty output".to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::model::audio::vad::model::tests::{chunk, model};
    use crate::test_utils::cpu_setup;

    /// The window is `context ++ chunk ++ reflect(context)`, with the mirror
    /// starting one sample before the last: PyTorch's reflect excludes the
    /// boundary sample.
    #[test]
    fn window_reflects_the_tail_without_the_boundary_sample() {
        let (_client, device) = cpu_setup();
        let vad = model(&device);
        let ctx_len = vad.config().context_samples;
        let len = vad.config().chunk_samples;
        let context = chunk(1.0, ctx_len);
        let body = chunk(2.0, len);

        let window = vad.window(&context, &body).expect("window");
        assert_eq!(window.len(), vad.config().window_samples());
        assert_eq!(&window[..ctx_len], &context[..]);
        assert_eq!(&window[ctx_len..ctx_len + len], &body[..]);
        for k in 0..ctx_len {
            assert_eq!(window[ctx_len + len + k], body[len - 2 - k]);
        }
    }

    /// A body shorter than the pad plus two samples cannot be mirrored and is
    /// refused rather than indexed out of bounds.
    #[test]
    fn window_too_short_to_reflect_is_an_error() {
        let (_client, device) = cpu_setup();
        let vad = model(&device);
        let ctx_len = vad.config().context_samples;
        let err = vad
            .window(&[], &chunk(0.0, ctx_len + 1))
            .expect_err("a body of context_samples + 1 cannot reflect-pad");
        assert!(err.to_string().contains("too short"), "{err}");
    }
}
