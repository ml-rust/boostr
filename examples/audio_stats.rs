//! Report level, floor, pitch and harmonicity for one or more audio files.
//!
//! ```text
//! cargo run --release --example audio_stats -- a.wav b.wav ...
//! ```
//!
//! Built for A/B-ing takes and generated samples side by side. `signal` is the
//! column to read: loudness minus noise floor, which gain cannot change and
//! which therefore says whether a recording is actually cleaner or merely
//! louder.

use boostr::model::audio::enhance::{integrated_lufs, noise_floor_dbfs, peak_dbfs};
use boostr::model::audio::{PitchOptions, decode_wav, estimate_pitch, to_mono};

/// Last two path components, so sibling directories stay distinguishable
/// without printing the whole path.
fn short_name(path: &str) -> String {
    let parts: Vec<String> = std::path::Path::new(path)
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    parts[parts.len().saturating_sub(2)..].join("/")
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: audio_stats FILE.wav [FILE.wav ...]");
        std::process::exit(2);
    }

    println!(
        "{:<36} {:>6} {:>8} {:>8} {:>8} {:>7} {:>7} {:>6}",
        "file", "secs", "LUFS", "peak", "floor", "signal", "F0", "HNR"
    );
    for path in &paths {
        let wav = decode_wav(&std::fs::read(path)?)?;
        let samples = to_mono(&wav.samples, wav.channels)?;
        let rate = wav.sample_rate;

        let lufs = integrated_lufs(&samples, rate)?;
        let floor = noise_floor_dbfs(&samples, rate);
        let track = estimate_pitch(&samples, rate, PitchOptions::default())?;
        let voiced: Vec<f64> = track.f0.iter().filter_map(|&v| v).collect();

        println!(
            "{:<36} {:>6.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>7.1} {:>6.2}",
            short_name(path),
            samples.len() as f64 / rate as f64,
            lufs,
            peak_dbfs(&samples),
            floor,
            lufs - floor,
            median(&voiced),
            track.mean_hnr_db.unwrap_or(f64::NAN),
        );
    }
    Ok(())
}
