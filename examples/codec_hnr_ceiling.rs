//! Does NeuCodec itself cap harmonicity, making corpus noise policy moot?
//!
//! Background: a probe over 60 generations found that this TTS base model's
//! `Audio quality:` description does NOT move the noise in its output. That
//! leaves denoising the corpus as the remaining lever — but the generations
//! measured only ~11-12 dB harmonic-to-noise ratio, where clean speech normally
//! sits at 15-25 dB. If the CODEC is what caps HNR, then neither labelling nor
//! denoising the training audio can change the result, because everything the
//! model can emit passes through that codec first.
//!
//! This settles it by measuring a real recording before and after a NeuCodec
//! round trip.
//!
//! # The rate control matters
//!
//! NeuCodec encodes 16 kHz and reconstructs 24 kHz, so a naive before/after
//! comparison confounds the codec with the rate change. This measures three
//! things: the source at 16 kHz, the source resampled to 24 kHz, and the codec
//! round trip at 24 kHz. If the first two agree, the rate is not a factor and
//! the third difference is the codec's alone.
//!
//! Usage:
//!   cargo run --release --features cuda --example codec_hnr_ceiling -- AUDIO_FILE NEUCODEC_DIR

use boostr::Runtime;
use boostr::model::audio::neucodec::{NeuCodec, NeuCodecEncoder};
use boostr::model::audio::{
    PitchOptions, decode_audio_file_mono_at, estimate_pitch, measure_quality, resample,
};
use boostr::runtime::cpu::{CpuDevice, CpuRuntime};

/// Encoder input rate, fixed by the checkpoint's fbank frontend.
const ENCODE_RATE: u32 = 16_000;
/// Decoder output rate, fixed by the checkpoint.
const DECODE_RATE: u32 = 24_000;
/// Seconds of audio to measure. Long enough for a stable HNR average, short
/// enough to stay well inside the encoder's quadratic-attention limit.
const CLIP_SECS: usize = 20;

fn report(label: &str, samples: &[f32], rate: u32) -> Option<f64> {
    let q = measure_quality(samples, rate).ok()?;
    let track = estimate_pitch(samples, rate, PitchOptions::default()).ok()?;
    let hnr = track.mean_hnr_db;
    println!(
        "{label:<34} {:>6.2}s  {:>7.1} dBFS floor  {:>5.1}% voiced  {:>7}",
        q.duration_s,
        q.floor_dbfs,
        track.voiced_fraction * 100.0,
        hnr.map_or("-".to_string(), |h| format!("{h:.2} dB"))
    );
    hnr
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let audio_path = args
        .next()
        .ok_or("usage: codec_hnr_ceiling AUDIO_FILE NEUCODEC_DIR")?;
    let codec_dir = args
        .next()
        .ok_or("usage: codec_hnr_ceiling AUDIO_FILE NEUCODEC_DIR")?;

    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let all = decode_audio_file_mono_at(std::path::Path::new(&audio_path), ENCODE_RATE)?;
    let want = CLIP_SECS * ENCODE_RATE as usize;
    if all.len() < want {
        return Err(format!(
            "need at least {CLIP_SECS}s at {ENCODE_RATE} Hz, file has {:.1}s",
            all.len() as f64 / ENCODE_RATE as f64
        )
        .into());
    }
    // Skip the first 30 s: recordings usually open with silence or an intro,
    // and an unvoiced stretch would make the HNR average meaningless.
    let skip = (30 * ENCODE_RATE as usize).min(all.len() - want);
    let clip = &all[skip..skip + want];

    println!(
        "{:<34} {:>8}  {:>17}  {:>13}  {:>7}",
        "stage", "duration", "noise floor", "voiced", "HNR"
    );
    println!("{}", "-".repeat(88));

    let source_16k = report("1. source @ 16 kHz", clip, ENCODE_RATE);

    let upsampled = resample(clip, ENCODE_RATE, DECODE_RATE)?;
    let source_24k = report("2. source resampled to 24 kHz", &upsampled, DECODE_RATE);

    println!("\nloading NeuCodec from {codec_dir} ...");
    let encoder = NeuCodecEncoder::<CpuRuntime>::from_safetensors(&codec_dir, &device)?;
    let codec = NeuCodec::<CpuRuntime>::from_safetensors(&codec_dir, &device)?;

    let codes = encoder.encode(&client, clip, &device)?;
    let n_codes = codes.numel();
    let indices = codes.reshape(&[1, n_codes])?;
    let waveform = codec.decode(&client, &indices)?;
    let round_trip: Vec<f32> = waveform.contiguous()?.to_vec();
    println!();
    let coded = report("3. NeuCodec round trip @ 24 kHz", &round_trip, DECODE_RATE);

    println!(
        "\n{n_codes} codes for {CLIP_SECS}s = {:.1} tokens/s",
        n_codes as f64 / CLIP_SECS as f64
    );

    match (source_16k, source_24k, coded) {
        (Some(a), Some(b), Some(c)) => {
            let rate_effect = b - a;
            let codec_effect = c - b;
            println!(
                "\nresample effect: {rate_effect:+.2} dB   codec effect: {codec_effect:+.2} dB"
            );
            if rate_effect.abs() > 1.0 {
                println!(
                    "⚠ the resample alone moved HNR by {rate_effect:+.2} dB; the codec figure \
                     below is confounded."
                );
            }
            // The generations measured ~11.5 dB. If the codec lands a clean
            // source near that number, the codec is the ceiling and no corpus
            // treatment can raise it.
            if c < 13.0 {
                println!(
                    "=> codec output HNR {c:.2} dB is at the level the model's own generations \
                     reach.\n   THE CODEC IS THE CEILING: denoising the corpus cannot raise it."
                );
            } else {
                println!(
                    "=> codec preserves {c:.2} dB, above what the generations reach.\n   \
                     The ceiling is NOT the codec, so corpus quality still controls the output."
                );
            }
        }
        _ => println!("\nHNR unavailable for at least one stage — nothing voiced?"),
    }

    Ok(())
}
