//! Bring a raw recording to reference quality and report what changed.
//!
//! ```text
//! cargo run --release --example enhance_reference -- IN.wav OUT.wav [NOISE.wav]
//! ```
//!
//! `NOISE.wav` is optional room tone from the same chain — a few seconds with
//! nobody speaking. Supply it when the take has no pause in it; without a
//! pause and without a clip there is no floor to measure and the denoiser
//! correctly leaves the take alone.

use boostr::model::audio::enhance::{
    EnhanceOptions, EnhanceReport, enhance_with_noise_profile, integrated_lufs, noise_floor_dbfs,
    peak_dbfs,
};
use boostr::model::audio::{decode_wav, encode_wav_pcm16, to_mono};

fn load(path: &str) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let wav = decode_wav(&std::fs::read(path)?)?;
    let mono = to_mono(&wav.samples, wav.channels)?;
    Ok((mono, wav.sample_rate))
}

fn line(label: &str, before: f64, after: f64, unit: &str) {
    println!(
        "  {label:<18} {before:>8.2} -> {after:>8.2} {unit}   ({:+.2})",
        after - before
    );
}

fn report(r: &EnhanceReport) {
    line("loudness", r.input_lufs, r.output_lufs, "LUFS");
    line("peak", r.input_peak_dbfs, r.output_peak_dbfs, "dBFS");
    line(
        "noise floor",
        r.input_noise_floor_dbfs,
        r.output_noise_floor_dbfs,
        "dBFS",
    );
    let before = r.input_lufs - r.input_noise_floor_dbfs;
    let after = r.output_lufs - r.output_noise_floor_dbfs;
    line("signal above floor", before, after, "dB");
    println!("  {:<18} {:>8.2} dB", "gain applied", r.applied_gain_db);
    println!(
        "  {:<18} {:>8.2} dB sustained, {:.2} dB peak",
        "limiting", r.limiter_reduction_db, r.limiter_peak_reduction_db
    );
    if !r.reached_target {
        println!(
            "  note: stopped {:.2} LU short of the target — more would need limiting past the cap",
            -18.0 - r.output_lufs
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} IN.wav OUT.wav [NOISE.wav]", args[0]);
        std::process::exit(2);
    }

    let (samples, rate) = load(&args[1])?;
    println!(
        "{}: {:.2}s at {} Hz",
        args[1],
        samples.len() as f64 / rate as f64,
        rate
    );

    let noise = match args.get(3) {
        Some(p) => {
            let (n, nr) = load(p)?;
            if nr != rate {
                return Err(format!("noise clip is {nr} Hz, take is {rate} Hz").into());
            }
            println!(
                "{p}: {:.2}s of room tone, floor {:.2} dBFS",
                n.len() as f64 / nr as f64,
                noise_floor_dbfs(&n, nr)
            );
            Some(n)
        }
        None => None,
    };

    let (out, r) =
        enhance_with_noise_profile(&samples, noise.as_deref(), rate, EnhanceOptions::default())?;
    report(&r);

    // A denoiser that found no floor to work from leaves the take alone. Say
    // so rather than letting the caller assume it ran.
    if noise.is_none()
        && (r.output_noise_floor_dbfs - r.input_noise_floor_dbfs - r.applied_gain_db).abs() < 0.5
    {
        println!(
            "  note: the floor moved only by the applied gain — this take has no pause \
             the gate could measure. Pass a room-tone clip as the third argument."
        );
    }

    std::fs::write(&args[2], encode_wav_pcm16(&out, rate)?)?;
    println!(
        "wrote {} ({:.2} LUFS, peak {:.2} dBFS)",
        args[2],
        integrated_lufs(&out, rate)?,
        peak_dbfs(&out)
    );
    Ok(())
}
