//! The Silero VAD network and its forward pass, read off the upstream ONNX
//! graph node by node.
//!
//! # The input contract, which is the part that silently produces garbage
//!
//! The network consumes `context_samples + chunk_samples` samples per step: the
//! LAST `context_samples` of the PREVIOUS chunk, concatenated in FRONT of the
//! current chunk. At 16 kHz that is 64 + 512 = 576. The context is zeros at the
//! start of a stream and is NEVER reset between chunks of the same stream.
//!
//! Feeding a bare 512 samples does not error — it returns roughly 0 for every
//! chunk. Thirty seconds of loud speech then reports a max probability near
//! 0.03, and the VAD looks merely conservative while marking everything as
//! silence. [`VadState`] owns the context so a caller cannot skip it.
//!
//! # Forward pass
//!
//! 1. REFLECT-pad `context_samples` at the END of the buffer — upstream's
//!    `nn.ReflectionPad1d((0, context))`, exported as an ONNX `Pad` node in
//!    "reflect" mode. At 16 kHz: 576 -> 640. Zero-padding instead runs fine and
//!    shifts every probability by up to 0.2.
//! 2. STFT as a convolution: `conv1d(x[B, 1, 640], forward_basis_buffer,
//!    stride = n_fft / 2, padding = 0)` -> `[B, 258, 4]`. No bias.
//! 3. Magnitude: channels `0..129` are the real part, `129..258` the imaginary
//!    part. `mag = sqrt(re^2 + im^2)` -> `[B, 129, 4]`.
//! 4. Four `conv1d` + ReLU, every one `kernel_size = 3`, `padding = 1`, with
//!    strides `[1, 2, 2, 1]` -> `[B, 128, 1]`.
//! 5. Squeeze the trailing length-1 time axis -> `[B, 128]`.
//! 6. One LSTM cell step, hidden size 128.
//! 7. ReLU on the hidden state, then a `kernel_size = 1` conv down to one
//!    channel, then sigmoid. That scalar is the speech probability. (The
//!    graph's trailing `ReduceMean` over the time axis is identity here — the
//!    axis is always length 1 for a single chunk.)
//!
//! # Gate order
//!
//! `decoder.rnn.weight_{ih,hh}` are stored in PyTorch's `[i, f, g, o]` order.
//! The ONNX graph's slice/concat gymnastics exist only to convert that into
//! ONNX's `[i, o, f, g]` layout for its `LSTM` op, so [`crate::nn::Lstm`]
//! consumes these weights unchanged — no reordering.

use crate::error::{Error, Result};
use crate::model::audio::vad::config::{ENCODER_KERNEL, ENCODER_STRIDES, HIDDEN_SIZE, VadConfig};
use crate::model::audio::vad::state::VadState;
use crate::nn::{Conv1d, Lstm};
use numr::dtype::DType;
use numr::ops::{ConvOps, PaddingMode, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Raw checkpoint tensors, before they are wrapped into modules.
///
/// [`SileroVad::new`] validates every shape against a [`VadConfig`], so a
/// caller building these by hand cannot silently mismatch the sample rate.
pub struct SileroVadWeights<R: Runtime> {
    /// `stft.forward_basis_buffer` — `[2 * freq_bins, 1, n_fft]`.
    pub stft_basis: Tensor<R>,
    /// `encoder.{i}.reparam_conv.{weight,bias}`, in order.
    pub encoder: Vec<(Tensor<R>, Tensor<R>)>,
    /// `decoder.rnn.weight_ih` — `[4 * 128, 128]`, PyTorch `[i, f, g, o]` order.
    pub rnn_weight_ih: Tensor<R>,
    /// `decoder.rnn.weight_hh` — `[4 * 128, 128]`, same order.
    pub rnn_weight_hh: Tensor<R>,
    /// `decoder.rnn.bias_ih` — `[4 * 128]`.
    pub rnn_bias_ih: Tensor<R>,
    /// `decoder.rnn.bias_hh` — `[4 * 128]`.
    pub rnn_bias_hh: Tensor<R>,
    /// `decoder.decoder.2.weight` — `[1, 128, 1]`.
    pub head_weight: Tensor<R>,
    /// `decoder.decoder.2.bias` — `[1]`.
    pub head_bias: Tensor<R>,
}

/// The Silero VAD network.
pub struct SileroVad<R: Runtime> {
    config: VadConfig,
    stft_basis: Tensor<R>,
    encoder: Vec<Conv1d<R>>,
    rnn: Lstm<R>,
    head: Conv1d<R>,
}

impl<R: Runtime<DType = DType>> SileroVad<R> {
    /// Build from raw checkpoint tensors, validating every shape against
    /// `config`.
    pub fn new(config: VadConfig, weights: SileroVadWeights<R>) -> Result<Self> {
        let bins = config.freq_bins();
        check_shape(
            "stft.forward_basis_buffer",
            &weights.stft_basis,
            &[2 * bins, 1, config.n_fft],
        )?;

        let channels = config.encoder_channels();
        if weights.encoder.len() != channels.len() {
            return Err(Error::InvalidArgument {
                arg: "weights.encoder",
                reason: format!(
                    "expected {} encoder convolutions, got {}",
                    channels.len(),
                    weights.encoder.len()
                ),
            });
        }

        let mut encoder = Vec::with_capacity(channels.len());
        for (i, ((weight, bias), (in_c, out_c))) in
            weights.encoder.into_iter().zip(channels).enumerate()
        {
            check_shape(
                &format!("encoder.{i}.reparam_conv.weight"),
                &weight,
                &[out_c, in_c, ENCODER_KERNEL],
            )?;
            check_shape(&format!("encoder.{i}.reparam_conv.bias"), &bias, &[out_c])?;
            encoder.push(Conv1d::new(
                weight,
                Some(bias),
                ENCODER_STRIDES[i],
                PaddingMode::conv1d(1, 1),
                1,
                1,
                false,
            ));
        }

        // `Lstm::new` shape-checks the four RNN tensors itself, but it derives
        // the hidden size from them rather than pinning it, so check the width
        // here too — a mismatch would otherwise surface as a confusing matmul
        // error deep in the first chunk.
        check_shape(
            "decoder.rnn.weight_ih",
            &weights.rnn_weight_ih,
            &[4 * HIDDEN_SIZE, HIDDEN_SIZE],
        )?;
        let rnn = Lstm::new(
            weights.rnn_weight_ih,
            weights.rnn_weight_hh,
            weights.rnn_bias_ih,
            weights.rnn_bias_hh,
        )?;

        check_shape(
            "decoder.decoder.2.weight",
            &weights.head_weight,
            &[1, HIDDEN_SIZE, 1],
        )?;
        check_shape("decoder.decoder.2.bias", &weights.head_bias, &[1])?;
        let head = Conv1d::new(
            weights.head_weight,
            Some(weights.head_bias),
            1,
            PaddingMode::Valid,
            1,
            1,
            false,
        );

        Ok(Self {
            config,
            stft_basis: weights.stft_basis,
            encoder,
            rnn,
            head,
        })
    }

    /// The geometry this instance was built for.
    pub fn config(&self) -> &VadConfig {
        &self.config
    }

    /// A fresh zero state for one stream.
    pub fn new_state(&self, device: &R::Device) -> Result<VadState<R>> {
        VadState::new(&self.config, device)
    }

    /// Speech probability for one chunk of exactly
    /// [`VadConfig::chunk_samples`] samples, advancing `state`.
    ///
    /// The chunk is prefixed with `state`'s carried context and suffixed with a
    /// reflection pad before it reaches the network; the caller passes new
    /// audio only. A chunk of any other length is an error, never a silent pad
    /// — a padded final chunk would report a probability the upstream model
    /// never produces.
    pub fn chunk_probability<C>(
        &self,
        client: &C,
        state: &mut VadState<R>,
        chunk: &[f32],
    ) -> Result<f32>
    where
        C: RuntimeClient<R> + TensorOps<R> + ConvOps<R>,
    {
        let expected = self.config.chunk_samples;
        if chunk.len() != expected {
            return Err(Error::InvalidArgument {
                arg: "chunk",
                reason: format!("expected exactly {expected} samples, got {}", chunk.len()),
            });
        }
        let context_len = self.config.context_samples;
        if state.context.len() != context_len {
            return Err(Error::InvalidArgument {
                arg: "state",
                reason: format!(
                    "context is {} samples, this model needs {context_len} \
                     (state built for a different sample rate?)",
                    state.context.len()
                ),
            });
        }

        let buffer = self.window(&state.context, chunk)?;
        let device = client.device();
        let input =
            Tensor::<R>::from_slice(&buffer, &[1, 1, buffer.len()], device).map_err(Error::Numr)?;

        let (h_next, c_next) = self.forward_window(client, &input, &state.h, &state.c)?;
        let prob = self.head_probability(client, &h_next)?;

        state.h = h_next;
        state.c = c_next;
        // The next chunk's context is THIS chunk's tail — not the tail of the
        // padded window.
        state.context.clear();
        state
            .context
            .extend_from_slice(&chunk[expected - context_len..]);

        Ok(prob)
    }

    /// Probabilities for a whole signal, one per consecutive chunk, from a
    /// fresh zero state.
    ///
    /// Trailing samples that do not fill a whole chunk are dropped, matching
    /// upstream's own chunking.
    pub fn probabilities<C>(&self, client: &C, samples: &[f32]) -> Result<Vec<f32>>
    where
        C: RuntimeClient<R> + TensorOps<R> + ConvOps<R>,
    {
        let mut state = self.new_state(client.device())?;
        self.probabilities_with(client, &mut state, samples)
    }

    /// Same as [`SileroVad::probabilities`] but continues an existing stream,
    /// so a caller can feed audio in arbitrary blocks.
    pub fn probabilities_with<C>(
        &self,
        client: &C,
        state: &mut VadState<R>,
        samples: &[f32],
    ) -> Result<Vec<f32>>
    where
        C: RuntimeClient<R> + TensorOps<R> + ConvOps<R>,
    {
        let chunk = self.config.chunk_samples;
        let mut out = Vec::with_capacity(samples.len() / chunk);
        for window in samples.chunks_exact(chunk) {
            out.push(self.chunk_probability(client, state, window)?);
        }
        Ok(out)
    }

    /// Assemble the network's input window: `context ++ chunk`, then a
    /// reflection pad of `context_samples` on the tail.
    ///
    /// Upstream's STFT front end is `nn.ReflectionPad1d((0, context))`, which
    /// the ONNX graph exports as a `Pad` node in "reflect" mode. Zero-padding
    /// here instead still runs and still looks plausible, but moves every
    /// probability by up to 0.2.
    fn window(&self, context: &[f32], chunk: &[f32]) -> Result<Vec<f32>> {
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
    fn forward_window<C>(
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
    fn head_probability<C>(&self, client: &C, h: &Tensor<R>) -> Result<f32>
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

fn check_shape<R: Runtime>(name: &str, tensor: &Tensor<R>, expected: &[usize]) -> Result<()> {
    if tensor.shape() != expected {
        return Err(Error::ModelError {
            reason: format!(
                "{name}: expected shape {expected:?}, checkpoint has {:?}",
                tensor.shape()
            ),
        });
    }
    Ok(())
}
