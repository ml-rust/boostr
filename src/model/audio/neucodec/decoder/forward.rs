use super::NeuCodecDecoder;
use crate::error::Result;
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::model::audio::stft::{IStftClient, IStftOptions, IStftPadding, hann_window, istft};
use crate::nn::TrainMode;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> TrainMode for NeuCodecDecoder<R> {
    fn set_training(&mut self, training: bool) {
        self.set_training_mode(training);
    }

    fn is_training(&self) -> bool {
        self.is_training_mode()
    }
}

impl<R: Runtime<DType = DType>> NeuCodecDecoder<R> {
    /// Full forward: `x [B, T, fc_in_dim] -> waveform [B, T * hop_length]`.
    ///
    /// Runs on any backend: the iSTFT tail is generic and numr's Bluestein path
    /// covers this vocoder's non-power-of-two `n_fft = 1920`.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<numr::tensor::Tensor<R>>
    where
        C: NeuCodecClient<R> + IStftClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let (mag, phase) = self.forward_features(client, x)?;
        let window = hann_window(self.config().n_fft, x.tensor().device())?;
        istft(
            client,
            mag.tensor(),
            phase.tensor(),
            &window,
            IStftOptions {
                hop_length: self.config().hop_length,
                // Vocos `padding="same"`, NOT torch's `center=True`: the
                // reference implementation trims `(n_fft - hop)/2 = 720` per
                // end, not `n_fft/2 = 960`.
                // This sets both the output length (`T*hop`, one hop per input
                // frame) and the alignment, so the two are not interchangeable.
                padding: IStftPadding::Same,
                eps: 1e-8,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::make_decoder;
    use numr::autograd::Var;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    /// Full decoder: input `[batch, frames, fc_in_dim]` -> waveform
    /// `[batch, samples]`.
    ///
    /// Derivation (see `crate::model::audio::stft::istft`): overlap-add
    /// builds `raw_len = (frames-1)*hop + n_fft`, then Vocos `padding="same"`
    /// trims `(n_fft - hop)/2` from each end, leaving
    /// `raw_len - (n_fft - hop) = frames * hop` samples.
    ///
    /// So one input frame yields exactly `hop_length` output samples, which is
    /// what makes the 50 Hz latent rate line up with 24 kHz audio
    /// (`50 * 480 = 24000`). An earlier version of this port used
    /// `torch.istft`-style `center=true` trimming of `n_fft/2` per end and got
    /// `(frames-1)*hop` — one hop short, and misaligned by 240 samples.
    #[test]
    fn forward_waveform_sample_count_matches_vocos_same_padding() {
        let (decoder, client, device, config) = make_decoder(0.01);
        let batch = 2;
        let frames = 7;
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.05f32; batch * frames * config.fc_in_dim],
                &[batch, frames, config.fc_in_dim],
                &device,
            )
            .unwrap(),
            false,
        );
        let waveform = decoder.forward(&client, &x).unwrap();
        let expected_samples = frames * config.hop_length;
        assert_eq!(waveform.shape(), &[batch, expected_samples]);
    }

    #[test]
    fn forward_waveform_is_finite() {
        let (decoder, client, device, config) = make_decoder(0.02);
        let batch = 1;
        let frames = 6;
        let x_data: Vec<f32> = (0..(frames * config.fc_in_dim))
            .map(|i| (i as f32 * 0.017).sin())
            .collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_data, &[batch, frames, config.fc_in_dim], &device)
                .unwrap(),
            false,
        );
        let waveform = decoder.forward(&client, &x).unwrap();
        for v in waveform.contiguous().unwrap().to_vec::<f32>() {
            assert!(v.is_finite(), "waveform sample is not finite: {v}");
        }
    }
}
