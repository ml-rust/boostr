//! The Silero VAD network: construction, streaming entry points, and the
//! input contract. The forward pass itself, read off Silero's ONNX graph node
//! by node, lives in the sibling `forward` module.
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
//! 1. REFLECT-pad `context_samples` at the END of the buffer — Silero's
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
///
/// Fields are visible to the sibling `forward` module, which holds the graph
/// transcription; nothing outside `vad` reads them.
pub struct SileroVad<R: Runtime> {
    pub(super) config: VadConfig,
    pub(super) stft_basis: Tensor<R>,
    pub(super) encoder: Vec<Conv1d<R>>,
    pub(super) rnn: Lstm<R>,
    pub(super) head: Conv1d<R>,
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
    /// — a padded final chunk would report a probability the Silero model
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
    /// Silero's own chunking.
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

/// Unit tests for the parts of the network that do not need the checkpoint:
/// the input contract (chunk length, carried context) and weight validation.
///
/// Numerical parity against the Silero ONNX model lives in
/// `tests/silero_vad_parity.rs`, which needs the real weights. The synthetic
/// builders are visible across `vad` so the sibling `forward` tests share them.
#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    pub(in crate::model::audio::vad) fn patterned(
        shape: &[usize],
        scale: f32,
        device: &CpuDevice,
    ) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| scale * ((i % 17) as f32 - 8.0)).collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device).expect("patterned tensor")
    }

    /// A structurally correct 16 kHz model with synthetic weights. Every shape
    /// matches the real checkpoint, so the whole forward pass runs; only the
    /// numbers are meaningless.
    pub(in crate::model::audio::vad) fn model(device: &CpuDevice) -> SileroVad<CpuRuntime> {
        let config = VadConfig::silero_16k();
        let bins = config.freq_bins();
        let encoder: Vec<_> = config
            .encoder_channels()
            .iter()
            .map(|&(in_c, out_c)| {
                (
                    patterned(&[out_c, in_c, ENCODER_KERNEL], 0.001, device),
                    patterned(&[out_c], 0.01, device),
                )
            })
            .collect();
        let weights = SileroVadWeights {
            stft_basis: patterned(&[2 * bins, 1, config.n_fft], 0.002, device),
            encoder,
            rnn_weight_ih: patterned(&[4 * HIDDEN_SIZE, HIDDEN_SIZE], 0.001, device),
            rnn_weight_hh: patterned(&[4 * HIDDEN_SIZE, HIDDEN_SIZE], 0.001, device),
            rnn_bias_ih: patterned(&[4 * HIDDEN_SIZE], 0.01, device),
            rnn_bias_hh: patterned(&[4 * HIDDEN_SIZE], 0.01, device),
            head_weight: patterned(&[1, HIDDEN_SIZE, 1], 0.01, device),
            head_bias: patterned(&[1], 0.02, device),
        };
        SileroVad::new(config, weights).expect("synthetic weights are shape-correct")
    }

    /// A deterministic, chunk-length signal that is different from every other
    /// chunk `seed` produces.
    pub(in crate::model::audio::vad) fn chunk(seed: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (seed + i as f32 * 0.01).sin() * 0.5)
            .collect()
    }

    #[test]
    fn context_carries_the_previous_chunks_tail() {
        let (client, device) = cpu_setup();
        let vad = model(&device);
        let mut state = vad.new_state(&device).expect("state");
        let ctx = vad.config().context_samples;
        let len = vad.config().chunk_samples;

        // Chunk 0 sees a zero context.
        assert!(state.context().iter().all(|&v| v == 0.0));

        let first = chunk(0.0, len);
        vad.chunk_probability(&client, &mut state, &first)
            .expect("chunk 0");
        assert_eq!(state.context(), &first[len - ctx..]);

        let second = chunk(3.0, len);
        vad.chunk_probability(&client, &mut state, &second)
            .expect("chunk 1");
        // The tail of chunk 1, NOT of the padded window and not of chunk 0.
        assert_eq!(state.context(), &second[len - ctx..]);
    }

    #[test]
    fn short_chunk_is_an_error_not_a_silent_pad() {
        let (client, device) = cpu_setup();
        let vad = model(&device);
        let mut state = vad.new_state(&device).expect("state");
        let err = vad
            .chunk_probability(&client, &mut state, &chunk(0.0, 511))
            .expect_err("511 samples must be rejected");
        assert!(matches!(err, Error::InvalidArgument { arg: "chunk", .. }));
    }

    #[test]
    fn long_chunk_is_an_error() {
        let (client, device) = cpu_setup();
        let vad = model(&device);
        let mut state = vad.new_state(&device).expect("state");
        let err = vad
            .chunk_probability(&client, &mut state, &chunk(0.0, 513))
            .expect_err("513 samples must be rejected");
        assert!(matches!(err, Error::InvalidArgument { arg: "chunk", .. }));
    }

    #[test]
    fn empty_chunk_is_an_error() {
        let (client, device) = cpu_setup();
        let vad = model(&device);
        let mut state = vad.new_state(&device).expect("state");
        let err = vad
            .chunk_probability(&client, &mut state, &[])
            .expect_err("an empty chunk must be rejected");
        assert!(matches!(err, Error::InvalidArgument { arg: "chunk", .. }));
    }

    #[test]
    fn probabilities_drops_the_trailing_partial_chunk() {
        let (client, device) = cpu_setup();
        let vad = model(&device);
        let len = vad.config().chunk_samples;
        let samples = chunk(0.0, 3 * len + 100);
        let probs = vad.probabilities(&client, &samples).expect("probabilities");
        assert_eq!(probs.len(), 3);
        assert!(probs.iter().all(|p| (0.0..=1.0).contains(p)));
    }

    #[test]
    fn probabilities_with_continues_an_existing_stream() {
        // Feeding one block of two chunks must equal feeding two blocks of one,
        // which is only true if the state (h, c AND context) survives the call.
        let (client, device) = cpu_setup();
        let vad = model(&device);
        let len = vad.config().chunk_samples;
        let mut samples = chunk(0.0, len);
        samples.extend(chunk(7.0, len));

        let one_shot = vad.probabilities(&client, &samples).expect("one shot");

        let mut state = vad.new_state(&device).expect("state");
        let mut split = vad
            .probabilities_with(&client, &mut state, &samples[..len])
            .expect("first block");
        split.extend(
            vad.probabilities_with(&client, &mut state, &samples[len..])
                .expect("second block"),
        );

        assert_eq!(one_shot.len(), 2);
        assert_eq!(split.len(), 2);
        for (a, b) in one_shot.iter().zip(split.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn wrong_sample_rate_weights_are_rejected() {
        let (_client, device) = cpu_setup();
        // 8 kHz-shaped first encoder conv against the 16 kHz config.
        let config = VadConfig::silero_16k();
        let bins = config.freq_bins();
        let encoder: Vec<_> = VadConfig::silero_8k()
            .encoder_channels()
            .iter()
            .map(|&(in_c, out_c)| {
                (
                    patterned(&[out_c, in_c, ENCODER_KERNEL], 0.001, &device),
                    patterned(&[out_c], 0.01, &device),
                )
            })
            .collect();
        let weights = SileroVadWeights {
            stft_basis: patterned(&[2 * bins, 1, config.n_fft], 0.002, &device),
            encoder,
            rnn_weight_ih: patterned(&[4 * HIDDEN_SIZE, HIDDEN_SIZE], 0.001, &device),
            rnn_weight_hh: patterned(&[4 * HIDDEN_SIZE, HIDDEN_SIZE], 0.001, &device),
            rnn_bias_ih: patterned(&[4 * HIDDEN_SIZE], 0.01, &device),
            rnn_bias_hh: patterned(&[4 * HIDDEN_SIZE], 0.01, &device),
            head_weight: patterned(&[1, HIDDEN_SIZE, 1], 0.01, &device),
            head_bias: patterned(&[1], 0.02, &device),
        };
        // `SileroVad` holds tensors and is not `Debug`, so `expect_err` is unusable.
        let Err(err) = SileroVad::<CpuRuntime>::new(config, weights) else {
            panic!("8 kHz encoder must not load as 16 kHz");
        };
        assert!(matches!(err, Error::ModelError { .. }));
    }

    #[test]
    fn a_state_from_the_other_sample_rate_is_rejected() {
        let (client, device) = cpu_setup();
        let vad = model(&device);
        let mut state =
            VadState::<CpuRuntime>::new(&VadConfig::silero_8k(), &device).expect("state");
        let err = vad
            .chunk_probability(&client, &mut state, &chunk(0.0, 512))
            .expect_err("a 32-sample context must be rejected by the 16 kHz model");
        assert!(matches!(err, Error::InvalidArgument { arg: "state", .. }));
    }
}
