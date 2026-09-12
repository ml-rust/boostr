//! Run the network, then segment: the whole `get_speech_timestamps` in one call.
//!
//! A free function rather than a method: [`SileroVad`] is boostr's type and
//! the segmenter lives here, so the orphan rule leaves no inherent impl to
//! extend.

use crate::error::{Error, Result};
use crate::vad::options::{SpeechSegment, VadSegmentOptions};
use crate::vad::segment::segments_from_probabilities;
use boostr::model::audio::vad::SileroVad;
use numr::dtype::DType;
use numr::ops::{ConvOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

/// Run `vad` over `samples` from a fresh state, then segment the
/// probabilities with [`segments_from_probabilities`].
///
/// The final partial chunk is ZERO-PADDED to a full chunk and evaluated,
/// because Silero's `get_speech_timestamps` scores `ceil(n / chunk)`
/// chunks and every boundary is measured off that grid. This is the one
/// deliberate difference from [`SileroVad::probabilities`], which DROPS a
/// trailing partial chunk to stay bit-comparable with the ONNX reference
/// run; dropping it here would shift boundaries on any signal whose length
/// is not a multiple of the chunk size.
pub fn speech_timestamps<R, C>(
    vad: &SileroVad<R>,
    client: &C,
    samples: &[f32],
    opts: &VadSegmentOptions,
) -> Result<Vec<SpeechSegment>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + ConvOps<R>,
{
    let config = *vad.config();
    let window = config.chunk_samples;
    if window == 0 {
        return Err(Error::InvalidArgument {
            arg: "config.chunk_samples",
            reason: "must be non-zero".to_string(),
        });
    }

    let mut state = vad.new_state(client.device())?;
    let chunks = samples.len().div_ceil(window);
    let mut probs = Vec::with_capacity(chunks);
    let mut padded = vec![0.0f32; window];
    for i in 0..chunks {
        let start = i * window;
        let end = (start + window).min(samples.len());
        let chunk = &samples[start..end];
        let prob = if chunk.len() == window {
            vad.chunk_probability(client, &mut state, chunk)?
        } else {
            padded[..chunk.len()].copy_from_slice(chunk);
            for sample in &mut padded[chunk.len()..] {
                *sample = 0.0;
            }
            vad.chunk_probability(client, &mut state, &padded)?
        };
        probs.push(prob);
    }

    segments_from_probabilities(&probs, samples.len(), config.sample_rate, window, opts)
}

/// Tests with synthetic weights: every shape matches the real checkpoint so
/// the forward pass runs, but the probabilities mean nothing. What is pinned
/// is the chunk grid — that the trailing partial chunk is evaluated, not
/// dropped — and that the output is the segmenter's output over that grid.
#[cfg(test)]
mod tests {
    use super::*;
    use boostr::model::audio::vad::{
        ENCODER_KERNEL, HIDDEN_SIZE, SileroVad, SileroVadWeights, VadConfig,
    };
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    fn patterned(shape: &[usize], scale: f32, device: &CpuDevice) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| scale * ((i % 17) as f32 - 8.0)).collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device).expect("patterned tensor")
    }

    fn model(device: &CpuDevice) -> SileroVad<CpuRuntime> {
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

    fn signal(len: usize) -> Vec<f32> {
        (0..len).map(|i| (i as f32 * 0.01).sin() * 0.5).collect()
    }

    /// The wrapper must equal "pad the tail chunk with zeros, score every
    /// chunk, segment" — the grid Silero measures boundaries on.
    #[test]
    fn matches_the_segmenter_over_the_zero_padded_chunk_grid() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let vad = model(&device);
        let window = vad.config().chunk_samples;
        let samples = signal(3 * window + 100);

        let mut padded = samples.clone();
        padded.resize(4 * window, 0.0);
        let probs = vad.probabilities(&client, &padded).expect("probabilities");
        assert_eq!(probs.len(), 4, "the partial chunk must be scored");

        // Thresholds low enough that synthetic probabilities can cross them,
        // and no duration rules, so any difference in the grid shows up.
        let opts = VadSegmentOptions {
            threshold: 0.02,
            min_speech_duration_ms: 0,
            min_silence_duration_ms: 0,
            speech_pad_ms: 0,
            ..VadSegmentOptions::default()
        };
        let expected = segments_from_probabilities(&probs, samples.len(), 16000, window, &opts)
            .expect("segment");
        let got = speech_timestamps(&vad, &client, &samples, &opts).expect("timestamps");
        assert_eq!(got, expected);
        for seg in &got {
            assert!(seg.end <= samples.len(), "segment past the signal: {seg:?}");
        }
    }

    /// An empty signal has no chunks and no segments, not an error.
    #[test]
    fn an_empty_signal_yields_no_segments() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let vad = model(&device);
        let got = speech_timestamps(&vad, &client, &[], &VadSegmentOptions::default())
            .expect("empty signal");
        assert!(got.is_empty());
    }
}
