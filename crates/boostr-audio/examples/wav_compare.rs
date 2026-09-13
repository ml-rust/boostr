//! Compare two wav files: SNR, multi-resolution log-magnitude L1, spectral
//! convergence, max absolute sample difference, and whether their lengths
//! match.
//!
//! ```text
//! cargo run --release --example wav_compare -- reference.wav test.wav
//! ```
//!
//! `reference.wav` is the ground truth (e.g. an F32 `AudioVAE` decode);
//! `test.wav` is what is being checked against it (e.g. the same latent
//! decoded at F16). A channel count above 1 is downmixed to mono; a sample
//! rate mismatch resamples `test.wav` to `reference.wav`'s rate before
//! comparison, since [`multi_resolution_stft_distance`] and [`snr_db`] both
//! assume a shared timebase.

use boostr_audio::{decode_wav, multi_resolution_stft_distance, snr_db, to_mono, to_mono_at_rate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [reference_path, test_path] = args.as_slice() else {
        eprintln!("usage: wav_compare REFERENCE.wav TEST.wav");
        std::process::exit(2);
    };

    let reference_wav = decode_wav(&std::fs::read(reference_path)?)?;
    let test_wav = decode_wav(&std::fs::read(test_path)?)?;

    let reference = to_mono(&reference_wav.samples, reference_wav.channels)?;
    let test = if test_wav.sample_rate == reference_wav.sample_rate {
        to_mono(&test_wav.samples, test_wav.channels)?
    } else {
        eprintln!(
            "resampling {test_path} from {} Hz to {} Hz to match {reference_path}",
            test_wav.sample_rate, reference_wav.sample_rate
        );
        to_mono_at_rate(&test_wav, reference_wav.sample_rate)?
    };

    let length_match = reference.len() == test.len();
    let n = reference.len().min(test.len());
    let max_abs_diff = reference[..n]
        .iter()
        .zip(test[..n].iter())
        .fold(0.0f32, |acc, (&r, &t)| acc.max((r - t).abs()));

    let snr = snr_db(&reference, &test)?;
    let distance = multi_resolution_stft_distance(&reference, &test, reference_wav.sample_rate)?;

    println!("reference: {reference_path} ({} samples)", reference.len());
    println!("test:      {test_path} ({} samples)", test.len());
    println!("length match:          {length_match}");
    println!("SNR:                   {snr:.2} dB");
    println!("log-magnitude L1:      {:.6}", distance.log_mag_l1);
    println!(
        "spectral convergence:  {:.6}",
        distance.spectral_convergence
    );
    println!("max abs sample diff:   {max_abs_diff:.6}");

    Ok(())
}
