//! Spectral-gating denoiser for reference recordings.
//!
//! The noise profile comes from **silence** — either a clip the caller points
//! at ([`denoise_with_profile`]) or, failing that, the quietest frames of the
//! take itself ([`denoise`]), which in real speech are the pauses between
//! phrases.
//!
//! It must come from silence and cannot come from an order statistic taken
//! per frequency bin across the whole take. A sustained vowel is stationary in
//! its harmonic bins, so a per-bin percentile reads the voice itself as the
//! floor and the gate then removes the voice. Measured on a held 150 Hz tone,
//! that mistake cost 28 dB of fundamental.
//!
//! A take with no pause in it therefore has no measurable noise profile. When
//! the quietest frames are not clearly below the median frame, [`denoise`]
//! returns the input untouched rather than gate a signal against itself.
//!
//! **CPU only, by design.** This is offline preparation run once per take, and
//! the mask is built from an order statistic over the whole spectrogram — a
//! sort per frequency bin, which is host work. Precedent:
//! [`crate::model::audio::mel`] builds its filterbank the same way.
//!
//! Subtracting too hard produces "musical noise" — isolated surviving cells
//! that warble. Two defenses are built in: a gain floor, so a suppressed cell
//! is attenuated rather than zeroed, and a box smoothing of the mask across
//! time, so a cell cannot switch on and off between adjacent frames. Smoothing
//! across *frequency* is available but off by default; see
//! [`DenoiseOptions::freq_smooth_bins`] for the measurement that decided it.

use super::super::kokoro::{IStftOptions, IStftPadding, istft};
use super::super::stft::{StftOptions, stft};
use crate::error::{Error, Result};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// Controls for [`denoise`].
#[derive(Debug, Clone, Copy)]
pub struct DenoiseOptions {
    /// Transform size. 1024 at 48 kHz is ~21 ms — long enough to resolve the
    /// pitch harmonics the mask must not eat, short enough to track plosives.
    pub n_fft: usize,
    /// Frame hop. Must divide `n_fft` evenly enough for the Hann window to sum
    /// to a constant; `n_fft / 4` does.
    pub hop_length: usize,
    /// Fraction of the quietest frames taken as the noise profile when no
    /// noise clip is supplied. 0.10 = the quietest tenth.
    pub noise_frame_fraction: f64,
    /// How far below the loudest frames the quiet frames must sit, in dB, for
    /// them to count as silence. Below this the take has no pause in it, no
    /// noise profile can be measured, and [`denoise`] does nothing.
    pub min_dynamic_range_db: f64,
    /// How far above the estimated noise a cell must sit to pass untouched,
    /// in dB. Higher removes more noise and more of the voice with it.
    pub over_subtraction_db: f64,
    /// Smallest gain any cell is reduced to, in dB. `-30.0` leaves a quiet
    /// residual floor rather than digital silence, which is what stops the
    /// surviving cells from sounding like isolated tones.
    pub gain_floor_db: f64,
    /// Half-width, in bins, of the box filter applied to the mask across
    /// frequency. **Default 0 — off.**
    ///
    /// Blurring a gain mask across frequency pulls the near-zero gains of the
    /// empty bins between harmonics into the harmonics themselves. Measured
    /// against a clean reference at 48 kHz with `n_fft = 1024`, where a 150 Hz
    /// voice leaves roughly two empty bins between partials: half-width 0 wins
    /// 15 dB, half-width 1 wins 3.4 dB, half-width 2 *loses* 9 dB. Real speech
    /// has the same harmonic spacing, so this is not an artifact of the test
    /// signal. Raise it only for a noise that is genuinely broadband and a
    /// voice that is not.
    pub freq_smooth_bins: usize,
    /// Half-width, in frames, of the box filter applied to the mask across
    /// time. Costs nothing measurable and stops a cell from switching on and
    /// off between adjacent frames, which is what musical noise is.
    pub time_smooth_frames: usize,
}

impl Default for DenoiseOptions {
    fn default() -> Self {
        Self {
            n_fft: 1024,
            hop_length: 256,
            noise_frame_fraction: 0.10,
            min_dynamic_range_db: 6.0,
            over_subtraction_db: 6.0,
            gain_floor_db: -30.0,
            freq_smooth_bins: 0,
            time_smooth_frames: 2,
        }
    }
}

/// Periodic Hann window — the analysis/synthesis pair `istft` normalizes for.
fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos()) as f32)
        .collect()
}

/// Estimated broadband noise floor of a recording, in dBFS.
///
/// Reported as the 10th percentile of 20 ms frame RMS. The minimum frame would
/// be a truer floor only in a take with real digital silence in it; in a live
/// room the minimum is dominated by whichever frame happened to fall between
/// two breaths, so an order statistic is steadier.
///
/// Returns `-inf` for silence and for input shorter than one frame.
pub fn noise_floor_dbfs(samples: &[f32], rate: u32) -> f64 {
    if rate == 0 {
        return f64::NEG_INFINITY;
    }
    let frame = (rate as usize / 50).max(1);
    if samples.len() < frame {
        return f64::NEG_INFINITY;
    }
    let mut rms: Vec<f64> = samples
        .chunks_exact(frame)
        .map(|c| (c.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>() / frame as f64).sqrt())
        .collect();
    rms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((rms.len() as f64 * 0.10) as usize).min(rms.len() - 1);
    let v = rms[idx];
    if v > 0.0 {
        20.0 * v.log10()
    } else {
        f64::NEG_INFINITY
    }
}

/// Remove broadband noise from `samples`, using its own pauses as the profile.
///
/// Returns the same number of samples. Phase is left untouched: a gate that
/// alters phase smears transients, and the mask here only ever scales
/// magnitude, so the original phase is still the correct one to resynthesize
/// with.
///
/// Returns the input unchanged when the take has no pause quiet enough to
/// measure a floor from (see [`DenoiseOptions::min_dynamic_range_db`]). Pass a
/// known silent region to [`denoise_with_profile`] to gate such a take.
pub fn denoise(samples: &[f32], opts: DenoiseOptions) -> Result<Vec<f32>> {
    run(samples, None, opts)
}

/// Remove broadband noise from `samples`, using `noise_clip` as the profile.
///
/// `noise_clip` is a stretch of the same recording chain with no voice in it —
/// room tone before the first word, or the gap between takes. It must be at
/// least `n_fft` samples long and must come from the same microphone, preamp
/// and room, because what is being measured is that chain's floor.
///
/// This is the accurate form. [`denoise`] has to find the pauses itself and
/// can only work when the take has some.
pub fn denoise_with_profile(
    samples: &[f32],
    noise_clip: &[f32],
    opts: DenoiseOptions,
) -> Result<Vec<f32>> {
    if noise_clip.len() < opts.n_fft {
        return Err(Error::InvalidArgument {
            arg: "noise_clip",
            reason: format!(
                "need at least n_fft ({}) samples of silence, got {}",
                opts.n_fft,
                noise_clip.len()
            ),
        });
    }
    run(samples, Some(noise_clip), opts)
}

/// One signal's spectrogram, kept together because the mask needs the
/// magnitude while resynthesis needs the phase and the original shape.
struct Analysis {
    mag: Vec<f32>,
    phase: Tensor<CpuRuntime>,
    shape: Vec<usize>,
    f_bins: usize,
    t_frames: usize,
}

fn run(samples: &[f32], noise_clip: Option<&[f32]>, opts: DenoiseOptions) -> Result<Vec<f32>> {
    validate(&opts)?;
    // Too short to frame at all: nothing to estimate a floor from, so
    // returning the input unchanged is the only honest answer.
    if samples.len() < opts.n_fft {
        return Ok(samples.to_vec());
    }

    let device = CpuDevice::default();
    let client = CpuClient::new(device.clone());
    let window = Tensor::<CpuRuntime>::from_slice(&hann(opts.n_fft), &[opts.n_fft], &device)?;
    let st = StftOptions {
        n_fft: opts.n_fft,
        hop_length: opts.hop_length,
        center: true,
    };

    let analyze = |sig: &[f32]| -> Result<Analysis> {
        let wave = Tensor::<CpuRuntime>::from_slice(sig, &[1, sig.len()], &device)?;
        let (mag, phase) = stft(&client, &wave, &window, st)?;
        let shape = mag.shape().to_vec();
        Ok(Analysis {
            f_bins: shape[1],
            t_frames: shape[2],
            mag: mag.contiguous()?.to_vec(),
            phase,
            shape,
        })
    };

    let signal = analyze(samples)?;
    let (f_bins, t_frames) = (signal.f_bins, signal.t_frames);

    let noise = match noise_clip {
        Some(clip) => {
            let n = analyze(clip)?;
            debug_assert_eq!(n.f_bins, f_bins);
            let all: Vec<usize> = (0..n.t_frames).collect();
            mean_spectrum(&n.mag, n.f_bins, n.t_frames, &all)
        }
        None => match quiet_frame_profile(&signal.mag, f_bins, t_frames, &opts) {
            Some(profile) => profile,
            // No pause in the take: no floor can be measured, and gating
            // against the signal itself would remove the voice.
            None => return Ok(samples.to_vec()),
        },
    };

    let mask = build_mask(&signal.mag, &noise, f_bins, t_frames, &opts);
    let masked: Vec<f32> = signal
        .mag
        .iter()
        .zip(mask.iter())
        .map(|(&m, &g)| m * g)
        .collect();
    let masked = Tensor::<CpuRuntime>::from_slice(&masked, &signal.shape, &device)?;

    let back = istft(
        &client,
        &masked,
        &signal.phase,
        &window,
        IStftOptions {
            hop_length: opts.hop_length,
            padding: IStftPadding::Center,
            eps: 1e-8,
        },
    )?;

    // Center padding returns `(T_frames - 1) * hop` samples, which is within
    // one hop of the input length. Pad or trim so callers get back exactly
    // what they handed in and can zip the result against the original.
    let mut out: Vec<f32> = back.to_vec();
    out.resize(samples.len(), 0.0);
    Ok(out)
}

fn validate(opts: &DenoiseOptions) -> Result<()> {
    if opts.n_fft == 0 || opts.hop_length == 0 {
        return Err(Error::InvalidArgument {
            arg: "opts",
            reason: "n_fft and hop_length must be > 0".into(),
        });
    }
    if !(0.0..=1.0).contains(&opts.noise_frame_fraction) || opts.noise_frame_fraction == 0.0 {
        return Err(Error::InvalidArgument {
            arg: "noise_frame_fraction",
            reason: format!("must be in (0, 1], got {}", opts.noise_frame_fraction),
        });
    }
    Ok(())
}

/// Mean magnitude spectrum over the given frames, laid out per bin.
fn mean_spectrum(mag: &[f32], f_bins: usize, t_frames: usize, frames: &[usize]) -> Vec<f32> {
    let mut out = vec![0.0f32; f_bins];
    if frames.is_empty() {
        return out;
    }
    for k in 0..f_bins {
        let row = &mag[k * t_frames..(k + 1) * t_frames];
        let sum: f64 = frames.iter().map(|&t| row[t] as f64).sum();
        out[k] = (sum / frames.len() as f64) as f32;
    }
    out
}

/// Noise profile taken from the quietest frames, or `None` when the take has
/// no pause in it.
fn quiet_frame_profile(
    mag: &[f32],
    f_bins: usize,
    t_frames: usize,
    opts: &DenoiseOptions,
) -> Option<Vec<f32>> {
    let mut energy: Vec<(f64, usize)> = (0..t_frames)
        .map(|t| {
            let e: f64 = (0..f_bins)
                .map(|k| {
                    let m = mag[k * t_frames + t] as f64;
                    m * m
                })
                .sum();
            (e, t)
        })
        .collect();
    energy.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let take = ((t_frames as f64 * opts.noise_frame_fraction) as usize).clamp(1, t_frames);
    let quiet_mean: f64 = energy[..take].iter().map(|e| e.0).sum::<f64>() / take as f64;
    // Compared against the LOUD frames, not the median. In a take that is
    // mostly silence — a single sentence with the recorder left running, which
    // is most reference takes — the median frame *is* the noise floor, and a
    // median comparison reads a perfectly gateable recording as pause-free. A
    // real 30 s take measured 1.9 dB against the median and 24.2 dB against
    // the 90th percentile.
    let loud = energy[((t_frames as f64 * 0.9) as usize).min(t_frames - 1)].0;

    // A pause is only a pause if it is meaningfully quieter than the speech.
    if quiet_mean <= 0.0 || loud <= 0.0 {
        return None;
    }
    let range_db = 10.0 * (loud / quiet_mean).log10();
    if range_db < opts.min_dynamic_range_db {
        return None;
    }

    let frames: Vec<usize> = energy[..take].iter().map(|e| e.1).collect();
    Some(mean_spectrum(mag, f_bins, t_frames, &frames))
}

/// Per-cell gain in `[floor, 1]`, laid out `[F, T]` to match the spectrogram.
fn build_mask(
    mag: &[f32],
    noise: &[f32],
    f_bins: usize,
    t_frames: usize,
    opts: &DenoiseOptions,
) -> Vec<f32> {
    let threshold_scale = 10f64.powf(opts.over_subtraction_db / 20.0);
    let floor = 10f32.powf(opts.gain_floor_db as f32 / 20.0);

    let mut raw = vec![0.0f32; f_bins * t_frames];
    for k in 0..f_bins {
        let row = &mag[k * t_frames..(k + 1) * t_frames];
        let threshold = noise[k] as f64 * threshold_scale;
        for (t, &m) in row.iter().enumerate() {
            // Spectral subtraction in the magnitude domain, expressed as a
            // gain so the smoothing below has something continuous to work on.
            let g = if m as f64 > 0.0 {
                ((m as f64 - threshold) / m as f64).clamp(0.0, 1.0)
            } else {
                0.0
            };
            raw[k * t_frames + t] = g as f32;
        }
    }

    box_smooth(
        &raw,
        f_bins,
        t_frames,
        opts.freq_smooth_bins,
        opts.time_smooth_frames,
    )
    .into_iter()
    .map(|g| g.max(floor))
    .collect()
}

/// Separable box filter over a `[F, T]` grid, clamped at the edges.
fn box_smooth(src: &[f32], f_bins: usize, t_frames: usize, fw: usize, tw: usize) -> Vec<f32> {
    let mut tmp = vec![0.0f32; src.len()];
    // Across time.
    for k in 0..f_bins {
        let row = &src[k * t_frames..(k + 1) * t_frames];
        for t in 0..t_frames {
            let lo = t.saturating_sub(tw);
            let hi = (t + tw + 1).min(t_frames);
            let span = &row[lo..hi];
            tmp[k * t_frames + t] = span.iter().sum::<f32>() / span.len() as f32;
        }
    }
    // Across frequency.
    let mut out = vec![0.0f32; src.len()];
    for t in 0..t_frames {
        for k in 0..f_bins {
            let lo = k.saturating_sub(fw);
            let hi = (k + fw + 1).min(f_bins);
            let mut sum = 0.0f32;
            for kk in lo..hi {
                sum += tmp[kk * t_frames + t];
            }
            out[k * t_frames + t] = sum / (hi - lo) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests;
