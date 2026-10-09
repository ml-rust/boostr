//! [`ClapFeatureExtractor`]: transformers' `ClapFeatureExtractor` for an
//! unfused checkpoint (`truncation="rand_trunc"`): each clip padded or cut
//! to the model's span, then its log-mel spectrogram in dB.
//!
//! The spectrogram is transformers' `spectrogram()` with a periodic Hann
//! window, centred reflect padding, power 2, the Slaney-normalized Slaney
//! filterbank, a `1e-10` floor and `10 · log10`. It runs in `f64` on the
//! CPU through [`stft_complex`] and [`mel_filterbank`], with each
//! spectrum bin rounded to `complex64` as transformers stores it, and
//! leaves as `f32` on the caller's device.
//!
//! A clip longer than the span: transformers keeps a window at a random
//! offset. This port keeps the window at offset 0, one of its draws, so a
//! run repeats.

use std::f64::consts::PI;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::model::audio::mel::{MelNorm, MelScale, mel_filterbank};
use crate::model::audio::stft::{CenterPad, StftOptions, stft_complex};
use numr::dtype::DType;
use numr::ops::{BinaryOps, MatmulOps, ScalarOps, TypeConversionOps, UnaryOps, UtilityOps};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// How a clip shorter than the span is filled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClapPadding {
    /// Whole repeats of the clip, then zeros.
    RepeatPad,
    /// Repeats of the clip, the last one cut at the span.
    Repeat,
    /// Zeros.
    Pad,
}

/// `preprocessor_config.json` as read, with transformers' defaults.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct PreprocessorFile {
    feature_size: usize,
    sampling_rate: u32,
    hop_length: usize,
    max_length_s: usize,
    fft_window_size: usize,
    frequency_min: f64,
    frequency_max: f64,
    truncation: String,
    padding: ClapPadding,
}

impl Default for PreprocessorFile {
    fn default() -> Self {
        Self {
            feature_size: 64,
            sampling_rate: 48_000,
            hop_length: 480,
            max_length_s: 10,
            fft_window_size: 1024,
            frequency_min: 0.0,
            frequency_max: 14_000.0,
            truncation: "fusion".to_string(),
            padding: ClapPadding::RepeatPad,
        }
    }
}

/// The log-mel front end of a CLAP checkpoint.
#[derive(Debug, Clone)]
pub struct ClapFeatureExtractor {
    mels: usize,
    sampling_rate: u32,
    hop: usize,
    n_fft: usize,
    span: usize,
    padding: ClapPadding,
    /// `[n_fft / 2 + 1, mels]`, row-major: the filterbank transposed for a
    /// right-hand matmul.
    filters_t: Vec<f64>,
}

impl ClapFeatureExtractor {
    /// Parses `preprocessor_config.json` text.
    ///
    /// Returns [`Error::ModelError`] for malformed JSON, a `fusion`
    /// truncation (fused checkpoints are not supported), or a zero size.
    pub fn from_json(text: &str) -> Result<Self> {
        let file: PreprocessorFile = serde_json::from_str(text).map_err(|e| Error::ModelError {
            reason: format!("CLAP preprocessor_config.json: {e}"),
        })?;
        if file.truncation != "rand_trunc" {
            return Err(Error::ModelError {
                reason: format!(
                    "CLAP preprocessor_config.json sets truncation {:?}; this port runs \
                     \"rand_trunc\" checkpoints only, such as laion/larger_clap_general",
                    file.truncation
                ),
            });
        }
        let span = file.max_length_s * file.sampling_rate as usize;
        if file.feature_size == 0 || file.hop_length == 0 || file.fft_window_size < 2 || span == 0 {
            return Err(Error::ModelError {
                reason: "CLAP preprocessor_config.json holds a zero size".to_string(),
            });
        }
        let filters = mel_filterbank(
            file.feature_size,
            file.fft_window_size,
            f64::from(file.sampling_rate),
            file.frequency_min,
            file.frequency_max,
            MelScale::Slaney,
            MelNorm::Slaney,
        );
        let bins = file.fft_window_size / 2 + 1;
        let mut filters_t = vec![0.0f64; bins * file.feature_size];
        for m in 0..file.feature_size {
            for k in 0..bins {
                filters_t[k * file.feature_size + m] = filters[m * bins + k];
            }
        }
        Ok(Self {
            mels: file.feature_size,
            sampling_rate: file.sampling_rate,
            hop: file.hop_length,
            n_fft: file.fft_window_size,
            span,
            padding: file.padding,
            filters_t,
        })
    }

    /// The rate clips must be at, in Hz.
    pub fn sampling_rate(&self) -> u32 {
        self.sampling_rate
    }

    /// Frames per clip: `span / hop + 1`.
    pub fn frames(&self) -> usize {
        self.span / self.hop + 1
    }

    /// Mel bins per frame.
    pub fn mels(&self) -> usize {
        self.mels
    }

    /// `clip` padded or cut to the span, in `f64`.
    ///
    /// Returns [`Error::InvalidArgument`] for an empty clip, which no
    /// padding fills.
    pub fn fit(&self, clip: &[f32]) -> Result<Vec<f64>> {
        if clip.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "clip",
                reason: "an empty clip has nothing to repeat or pad".to_string(),
            });
        }
        let mut wave: Vec<f64> = clip.iter().map(|&v| f64::from(v)).collect();
        if wave.len() >= self.span {
            wave.truncate(self.span);
            return Ok(wave);
        }
        let repeats = self.span / wave.len();
        let mut out: Vec<f64> = match self.padding {
            ClapPadding::RepeatPad => wave.repeat(repeats),
            ClapPadding::Repeat => {
                let mut v = wave.repeat(repeats + 1);
                v.truncate(self.span);
                v
            }
            ClapPadding::Pad => wave,
        };
        out.resize(self.span, 0.0);
        Ok(out)
    }

    /// `np.hanning(n + 1)[:-1]`: the periodic Hann window, as NumPy
    /// evaluates it.
    fn window(&self) -> Vec<f64> {
        let m = self.n_fft + 1;
        let denom = (m - 1) as f64;
        (0..self.n_fft)
            .map(|i| {
                let n = (2 * i) as f64 - (m - 1) as f64;
                0.5 + 0.5 * (PI * n / denom).cos()
            })
            .collect()
    }

    /// The log-mel features of one clip, `[frames, mels]` row-major.
    fn clip_features(
        &self,
        client: &CpuClient,
        device: &CpuDevice,
        clip: &[f32],
    ) -> Result<Vec<f32>> {
        let wave = self.fit(clip)?;
        let wave = Tensor::<CpuRuntime>::from_slice(&wave, &[1, self.span], device)?;
        let window = Tensor::<CpuRuntime>::from_slice(&self.window(), &[self.n_fft], device)?;
        let opts = StftOptions {
            n_fft: self.n_fft,
            hop_length: self.hop,
            center: true,
            pad: CenterPad::Reflect,
        };
        let (real, imag) = stft_complex(client, &wave, &window, opts)?;
        // transformers stores the spectrum as complex64.
        let as_c64 = |t: &Tensor<CpuRuntime>| -> Result<Tensor<CpuRuntime>> {
            Ok(client.cast(&client.cast(t, DType::F32)?, DType::F64)?)
        };
        let (real, imag) = (as_c64(&real)?, as_c64(&imag)?);
        let power = client.add(&client.mul(&real, &real)?, &client.mul(&imag, &imag)?)?;
        let bins = self.n_fft / 2 + 1;
        let frames = self.frames();
        let power = power
            .reshape(&[bins, frames])?
            .transpose(0, 1)?
            .contiguous()?;
        let filters =
            Tensor::<CpuRuntime>::from_slice(&self.filters_t, &[bins, self.mels], device)?;
        let mel = client.matmul(&power, &filters)?;
        let mel = client.clamp(&mel, 1e-10, f64::INFINITY)?;
        let db = client.mul_scalar(&client.log10(&mel)?, 10.0)?;
        Ok(client.cast(&db, DType::F32)?.to_vec())
    }

    /// The features of `clips` (mono, at [`Self::sampling_rate`]) on
    /// `device`: `[clips, 1, frames, mels]`, `f32`.
    ///
    /// Returns [`Error::InvalidArgument`] when `clips` is empty or holds an
    /// empty clip.
    pub fn features<R: Runtime<DType = DType>>(
        &self,
        clips: &[Vec<f32>],
        device: &R::Device,
    ) -> Result<Tensor<R>> {
        if clips.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "clips",
                reason: "no clips to featurize".to_string(),
            });
        }
        let cpu = CpuDevice::new();
        let client = CpuClient::new(cpu.clone());
        let mut all = Vec::with_capacity(clips.len() * self.frames() * self.mels);
        for clip in clips {
            all.extend(self.clip_features(&client, &cpu, clip)?);
        }
        Ok(Tensor::<R>::from_slice(
            &all,
            &[clips.len(), 1, self.frames(), self.mels],
            device,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LARGER_CLAP_GENERAL: &str = r#"{"feature_size": 64, "fft_window_size": 1024,
        "frequency_max": 14000, "frequency_min": 50, "hop_length": 480, "max_length_s": 10,
        "padding": "repeatpad", "sampling_rate": 48000, "truncation": "rand_trunc"}"#;

    #[test]
    fn padding_modes_fill_the_span() {
        let mut fx = ClapFeatureExtractor::from_json(LARGER_CLAP_GENERAL).expect("config");
        fx.span = 10;
        let clip = [1.0f32, 2.0, 3.0];
        assert_eq!(
            fx.fit(&clip).expect("repeatpad"),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 0.0]
        );
        fx.padding = ClapPadding::Repeat;
        assert_eq!(
            fx.fit(&clip).expect("repeat"),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]
        );
        fx.padding = ClapPadding::Pad;
        assert_eq!(fx.fit(&clip).expect("pad")[3..], [0.0; 7]);
        assert_eq!(fx.fit(&[0.5f32; 12]).expect("cut"), vec![0.5; 10]);
        assert!(fx.fit(&[]).is_err());
    }

    #[test]
    fn fusion_is_refused() {
        assert!(ClapFeatureExtractor::from_json(r#"{"truncation": "fusion"}"#).is_err());
    }

    #[test]
    fn silence_floors_at_minus_100_db() {
        let fx = ClapFeatureExtractor::from_json(LARGER_CLAP_GENERAL).expect("config");
        assert_eq!(fx.frames(), 1001);
        let t = fx
            .features::<CpuRuntime>(&[vec![0.0f32; 4800]], &CpuDevice::new())
            .expect("features");
        assert_eq!(t.shape(), &[1, 1, 1001, 64]);
        let v: Vec<f32> = t.to_vec();
        assert!(v.iter().all(|&x| x == -100.0));
    }

    #[test]
    fn window_is_numpy_hanning() {
        let fx = ClapFeatureExtractor::from_json(LARGER_CLAP_GENERAL).expect("config");
        let w = fx.window();
        assert_eq!(w.len(), 1024);
        assert_eq!(w[0], 0.0);
        assert_eq!(w[512], 1.0);
    }
}
