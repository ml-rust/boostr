//! Sample-rate conversion for mono `f32` audio.
//!
//! Recorded corpora arrive at 44.1 or 48 kHz; [`super::neucodec`] and the mel
//! front-end demand 16 kHz. [`resample`] is that missing link: a polyphase
//! windowed-sinc FIR converter between any two rates.
//!
//! **CPU-only.** Resampling runs once per file at load time, on a host-side
//! `Vec<f32>` that has just come out of the WAV decoder and has never touched a
//! device. Uploading it to run a strided, phase-dependent gather would cost more
//! than the filter itself. Stays here (not in numr) because it is an audio
//! specific composition, not a core numerical op.
//!
//! # Design
//!
//! `from_rate / to_rate` reduces by GCD to a coprime fraction `down / up`.
//! Conceptually the signal is upsampled by `up` (zero stuffing), low-pass
//! filtered, then decimated by `down`. The `up`-times longer signal is never
//! materialised: the prototype filter is split into `up` phases, and each output
//! sample reads one phase against the input samples it actually overlaps.
//!
//! The prototype is `h(n) = up * fc * sinc(fc * n) * kaiser(n)` on the upsampled
//! grid, with `fc = 0.95 * min(1/up, 1/down)`. Its cutoff is therefore
//! `min(from_rate, to_rate) / 2` — the lower of the two Nyquist limits — scaled
//! by [`ROLLOFF`] to leave a transition band. That cutoff is the anti-aliasing:
//! content that the output rate cannot represent is stopped before decimation
//! folds it back into the audible band. The `up` factor restores the gain that
//! zero stuffing loses, so a DC input keeps its amplitude.
//!
//! The Kaiser window ([`KAISER_BETA`] = 8.6, about 80 dB of stopband rejection)
//! suppresses the ringing a rectangular truncation of the sinc would leave.
//!
//! # Edges
//!
//! Input indices outside `0..samples.len()` read as zero (implicit zero
//! padding). The first and last few hundred output samples therefore ramp in and
//! out over the filter's half-width instead of holding steady amplitude. For
//! speech corpora that region is leading and trailing silence; callers that need
//! exact edge amplitude must pad the signal themselves before calling.

use crate::error::{Error, Result};

use super::wav_decode::{WavData, to_mono};

/// Filter taps per polyphase branch, and so the filter's half-width in input
/// samples on either side of an output sample. 32 puts the transition band
/// near 8% of the lower sample rate.
pub const DEFAULT_TAPS_PER_PHASE: usize = 32;

/// Kaiser shape parameter. 8.6 gives roughly 80 dB of stopband rejection.
const KAISER_BETA: f64 = 8.6;

/// Cutoff as a fraction of the lower Nyquist limit, leaving a transition band
/// below it rather than aliasing right at the edge.
const ROLLOFF: f64 = 0.95;

/// Largest prototype filter [`resample`] builds, in taps on the upsampled grid.
///
/// The prototype spans `2 * taps_per_phase * max(up, down) + 1` taps, so a rate
/// pair that barely reduces — 44100 to 16001, which is already coprime — would
/// demand millions of taps and tens of megabytes. Such a pair is a typo, not a
/// workload, so it is rejected instead of allocated.
pub const MAX_FILTER_TAPS: usize = 1 << 20;

fn bad(arg: &'static str, reason: String) -> Error {
    Error::InvalidArgument { arg, reason }
}

fn gcd(a: u32, b: u32) -> u32 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Number of output samples for `in_len` input samples: `ceil(in_len * to / from)`.
///
/// Computed in `u64` so a long recording at a high ratio cannot wrap.
fn output_len(in_len: usize, from_rate: u32, to_rate: u32) -> Result<usize> {
    let scaled = (in_len as u64).checked_mul(to_rate as u64).ok_or_else(|| {
        bad(
            "samples",
            format!("{in_len} samples at {to_rate} Hz overflows a u64 output length"),
        )
    })?;
    let from = from_rate as u64;
    let out = scaled.div_ceil(from);
    usize::try_from(out).map_err(|_| {
        bad(
            "samples",
            format!("output length {out} does not fit in a usize"),
        )
    })
}

/// One polyphase branch: the taps `h(r + j * up)` for `j` in `j_min..`.
struct Phase {
    j_min: i64,
    taps: Vec<f32>,
}

/// The prototype low-pass split into `up` interleaved branches.
struct PolyphaseFilter {
    up: i64,
    down: i64,
    phases: Vec<Phase>,
}

impl PolyphaseFilter {
    /// Build the branch table for the coprime ratio `up / down`.
    ///
    /// `from_rate` and `to_rate` are carried only to name the offending pair in
    /// the [`MAX_FILTER_TAPS`] error.
    fn design(
        up: u64,
        down: u64,
        taps_per_phase: usize,
        from_rate: u32,
        to_rate: u32,
    ) -> Result<Self> {
        let half_len = (taps_per_phase as u64)
            .checked_mul(up.max(down))
            .filter(|half| *half <= (MAX_FILTER_TAPS as u64 - 1) / 2)
            .ok_or_else(|| {
                bad(
                    "from_rate",
                    format!(
                        "resampling {from_rate} Hz to {to_rate} Hz reduces to {up}/{down}, \
                         needing more than {MAX_FILTER_TAPS} filter taps at \
                         {taps_per_phase} taps per phase; pick rates with a larger common \
                         factor, or lower taps_per_phase"
                    ),
                )
            })?;

        let up_i = up as i64;
        let half_i = half_len.max(1) as i64;
        let half = half_i as f64;
        let fc = ROLLOFF * (1.0 / up as f64).min(1.0 / down as f64);
        // Zero stuffing by `up` divides the signal's energy across `up` slots;
        // the same factor here restores unity passband gain.
        let gain = up as f64 * fc;
        let i0_beta = bessel_i0(KAISER_BETA);

        let mut phases = Vec::with_capacity(up as usize);
        for r in 0..up_i {
            // Taps live at offsets `r + j * up` inside `[-half_i, half_i]`.
            let j_min = -((half_i + r) / up_i);
            let j_max = (half_i - r) / up_i;
            let mut taps = Vec::with_capacity((j_max - j_min + 1).max(0) as usize);
            for j in j_min..=j_max {
                let n = (r + j * up_i) as f64;
                let t = n / half;
                let window = bessel_i0(KAISER_BETA * (1.0 - t * t).max(0.0).sqrt()) / i0_beta;
                taps.push((gain * sinc(fc * n) * window) as f32);
            }
            phases.push(Phase { j_min, taps });
        }
        Ok(Self {
            up: up_i,
            down: down as i64,
            phases,
        })
    }

    /// Filter and resample `x` into exactly `out_len` samples.
    ///
    /// Output sample `m` sits at position `m * down` on the upsampled grid. That
    /// position selects a branch (`p mod up`) and an input anchor (`p div up`);
    /// only the taps of that one branch are touched.
    fn apply(&self, x: &[f32], out_len: usize) -> Vec<f32> {
        let n = x.len() as i64;
        let mut out = Vec::with_capacity(out_len);
        for m in 0..out_len {
            let p = (m as i64).saturating_mul(self.down);
            let r = p.rem_euclid(self.up) as usize;
            let anchor = p.div_euclid(self.up);
            let mut acc = 0.0f32;
            if let Some(phase) = self.phases.get(r) {
                // Clamp the tap range to the taps whose input index is in
                // bounds; everything outside reads as zero, so it is skipped.
                let last = phase.j_min + phase.taps.len() as i64 - 1;
                let lo = phase.j_min.max(anchor - n + 1);
                let hi = last.min(anchor);
                if lo <= hi {
                    let start = (lo - phase.j_min) as usize;
                    let end = (hi - phase.j_min) as usize + 1;
                    if let Some(taps) = phase.taps.get(start..end) {
                        for (t, &tap) in taps.iter().enumerate() {
                            let idx = anchor - lo - t as i64;
                            if let Some(&sample) = usize::try_from(idx).ok().and_then(|k| x.get(k))
                            {
                                acc += tap * sample;
                            }
                        }
                    }
                }
            }
            out.push(acc);
        }
        out
    }
}

/// Normalised sinc, `sin(pi x) / (pi x)`, with the removable singularity filled.
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        return 1.0;
    }
    let px = std::f64::consts::PI * x;
    px.sin() / px
}

/// Modified Bessel function of the first kind, order zero.
///
/// Sums `sum_k (x/2)^(2k) / (k!)^2`. Every term is positive and the factorial
/// square dominates, so for the `x <= KAISER_BETA` this module uses it settles
/// to f64 precision in well under 64 terms.
fn bessel_i0(x: f64) -> f64 {
    let quarter = (x / 2.0) * (x / 2.0);
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    for k in 1..64 {
        term *= quarter / ((k * k) as f64);
        sum += term;
        if term < 1e-18 * sum {
            break;
        }
    }
    sum
}

/// Resample mono f32 samples from `from_rate` to `to_rate`.
///
/// Uses [`DEFAULT_TAPS_PER_PHASE`]; see [`resample_with_taps`] for the rest of
/// the contract.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    resample_with_taps(samples, from_rate, to_rate, DEFAULT_TAPS_PER_PHASE)
}

/// Resample mono f32 samples, choosing the filter's taps per polyphase branch.
///
/// Longer branches narrow the transition band and cost proportionally more
/// arithmetic per output sample.
///
/// Returns exactly `ceil(samples.len() * to_rate / from_rate)` samples. An empty
/// input gives an empty output. `from_rate == to_rate` returns the input
/// unchanged, bit for bit, with no filtering. Out-of-range input indices read as
/// zero, so the edges ramp — see the module docs.
///
/// Returns [`Error::InvalidArgument`] when either rate is 0, when
/// `taps_per_phase` is 0, when the rate pair needs more than
/// [`MAX_FILTER_TAPS`] taps, or when the output length overflows.
pub fn resample_with_taps(
    samples: &[f32],
    from_rate: u32,
    to_rate: u32,
    taps_per_phase: usize,
) -> Result<Vec<f32>> {
    if from_rate == 0 {
        return Err(bad("from_rate", "source sample rate is 0 Hz".to_string()));
    }
    if to_rate == 0 {
        return Err(bad("to_rate", "target sample rate is 0 Hz".to_string()));
    }
    if taps_per_phase == 0 {
        return Err(bad(
            "taps_per_phase",
            "filter length is 0 taps per phase".to_string(),
        ));
    }
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }
    let out_len = output_len(samples.len(), from_rate, to_rate)?;
    if samples.is_empty() || out_len == 0 {
        return Ok(Vec::new());
    }

    let divisor = gcd(from_rate, to_rate).max(1);
    let up = (to_rate / divisor) as u64;
    let down = (from_rate / divisor) as u64;
    let filter = PolyphaseFilter::design(up, down, taps_per_phase, from_rate, to_rate)?;
    Ok(filter.apply(samples, out_len))
}

/// Downmix to mono and resample to `target_rate` in one step.
///
/// Averages the channels with [`to_mono`], then applies [`resample`]. Returns
/// [`Error::InvalidArgument`] for anything either of those rejects.
pub fn to_mono_at_rate(data: &WavData, target_rate: u32) -> Result<Vec<f32>> {
    let mono = to_mono(&data.samples, data.channels)?;
    if data.sample_rate == target_rate {
        return Ok(mono);
    }
    resample(&mono, data.sample_rate, target_rate)
}

#[cfg(test)]
mod tests;
