//! Decode the noise-conditioning probe's token ids and measure their noise floor.
//!
//! Second half of the experiment started by
//! `audio/pipeline/noise_conditioning_probe.py`. That script generates audio
//! token ids under three prompts that differ ONLY in the checkpoint's
//! `Audio quality:` description line; this one decodes them through NeuCodec and
//! measures each take with the same `measure_quality` the corpus pipeline uses.
//!
//! What the numbers mean: if the `clean` arm's noise floor sits meaningfully
//! below the `noisy` arm's, the description genuinely controls output noise, so a
//! home corpus can be recorded raw and LABELLED rather than denoised. If the two
//! distributions overlap, the field is decorative and the corpus must be denoised
//! before training.
//!
//! Usage:
//!   cargo run --example noise_probe_decode --release -- probe.json NEUCODEC_DIR [OUT_DIR]

use boostr::Runtime;
use boostr::model::audio::kokoro::IStftClient;
use boostr::model::audio::neucodec::{NeuCodec, NeuCodecClient};
use boostr::model::audio::{PitchOptions, encode_wav_pcm16, estimate_pitch, measure_quality};
use boostr::runtime::cpu::{CpuDevice, CpuRuntime};
use boostr::tensor::Tensor;
use numr::dtype::DType;

#[derive(serde::Deserialize)]
struct Probe {
    sample_rate: u32,
    results: Vec<Take>,
}

#[derive(serde::Deserialize)]
struct Take {
    arm: String,
    seed: u64,
    codes: Vec<i32>,
    complete: bool,
}

/// Mean and SAMPLE standard deviation (n-1), or `None` when there is nothing
/// to summarize. `n-1` because these are samples from the model's output
/// distribution, not the whole population.
fn mean_sd(xs: &[f64]) -> Option<(f64, f64)> {
    if xs.is_empty() {
        return None;
    }
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    if xs.len() < 2 {
        return Some((mean, f64::NAN));
    }
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    Some((mean, var.sqrt()))
}

/// Welch's t statistic and degrees of freedom for the difference of two means.
///
/// The comparison that matters is the separation of the two MEANS against the
/// standard error of that difference — not against the within-arm spread. An
/// earlier version of this example compared the delta to a pooled standard
/// deviation, which is a effect-size measure, not a test of whether the means
/// differ, and it called a p ~= 0.05 separation "within sampling spread".
fn welch(a: &[f64], b: &[f64]) -> Option<(f64, f64, f64)> {
    if a.len() < 2 || b.len() < 2 {
        return None;
    }
    let (ma, sa) = mean_sd(a)?;
    let (mb, sb) = mean_sd(b)?;
    let (va, vb) = (sa * sa / a.len() as f64, sb * sb / b.len() as f64);
    let se = (va + vb).sqrt();
    if se == 0.0 {
        return None;
    }
    let t = (mb - ma) / se;
    let df =
        (va + vb).powi(2) / (va.powi(2) / (a.len() - 1) as f64 + vb.powi(2) / (b.len() - 1) as f64);
    Some((t, df, se))
}

/// Decode every take and print the per-take table plus a per-arm summary.
///
/// Generic over the runtime so the same decode can be run on CPU and on CUDA
/// and the two compared — the iSTFT vocoder tail only became backend-agnostic
/// recently, and matching floors on real generated audio is a stronger check of
/// that than a synthetic parity test.
#[allow(clippy::type_complexity)]
fn decode_all<R, C>(
    client: &C,
    device: &R::Device,
    codec: &NeuCodec<R>,
    probe: &Probe,
    out_dir: Option<&String>,
) -> Result<std::collections::BTreeMap<String, Vec<f64>>, Box<dyn std::error::Error>>
where
    R: Runtime<DType = DType>,
    C: NeuCodecClient<R> + IStftClient<R>,
    R::Client: NeuCodecClient<R>,
{
    println!(
        "\n{:<16} {:>5} {:>7} {:>9} {:>9} {:>9} {:>8} {:>8} {:>8} {:>7}",
        "arm",
        "seed",
        "dur_s",
        "peak_dB",
        "rms_dB",
        "floor_dB",
        "snr_dB",
        "clipped",
        "hnr_dB",
        "f0_Hz"
    );
    println!("{}", "-".repeat(90));

    let mut by_arm: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    let mut snr_by_arm: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    let mut hnr_by_arm: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    let mut f0_by_arm: std::collections::BTreeMap<String, Vec<f64>> = Default::default();

    for take in &probe.results {
        if take.codes.is_empty() {
            println!("{:<16} {:>5} {:>7}", take.arm, take.seed, "EMPTY");
            continue;
        }
        let frames = take.codes.len();
        let indices = Tensor::<R>::from_slice(&take.codes, &[1, frames], device)?;
        let waveform = codec.decode(client, &indices)?;
        let samples: Vec<f32> = waveform.contiguous()?.to_vec();
        let q = measure_quality(&samples, probe.sample_rate)?;

        // HNR measures noise DURING speech, which the floor cannot: the floor
        // is set by the pauses, and this model's pauses are digitally silent
        // under every prompt.
        let track = estimate_pitch(&samples, probe.sample_rate, PitchOptions::default()).ok();
        let hnr = track.as_ref().and_then(|t| t.mean_hnr_db);
        // Mean F0 over voiced frames. Male speech typically sits 85-180 Hz and
        // female 165-255 Hz, so this is what says whether a base model's
        // speaker prior can carry the voice being cloned.
        let f0 = track.as_ref().and_then(|t| t.mean_hz);

        let flag = if take.complete { "" } else { "  TRUNCATED" };
        println!(
            "{:<16} {:>5} {:>7.2} {:>9.1} {:>9.1} {:>9.1} {:>8.1} {:>8} {:>8} {:>7}{}",
            take.arm,
            take.seed,
            q.duration_s,
            q.peak_dbfs,
            q.rms_dbfs,
            q.floor_dbfs,
            q.snr_db,
            q.clipped_samples,
            hnr.map_or("-".to_string(), |h| format!("{h:.1}")),
            f0.map_or("-".to_string(), |v| format!("{v:.0}")),
            flag
        );

        if take.complete {
            by_arm
                .entry(take.arm.clone())
                .or_default()
                .push(q.floor_dbfs);
            snr_by_arm
                .entry(take.arm.clone())
                .or_default()
                .push(q.snr_db);
            if let Some(h) = hnr {
                hnr_by_arm.entry(take.arm.clone()).or_default().push(h);
            }
            if let Some(v) = f0 {
                f0_by_arm.entry(take.arm.clone()).or_default().push(v);
            }
        }

        if let Some(dir) = out_dir {
            let path = std::path::Path::new(dir).join(format!("{}_{}.wav", take.arm, take.seed));
            std::fs::write(path, encode_wav_pcm16(&samples, probe.sample_rate)?)?;
        }
    }

    println!("\nComplete takes only, floor_dBFS (lower = quieter background):");
    println!(
        "{:<16} {:>3} {:>12} {:>10} {:>12} {:>10} {:>8} {:>9} {:>7}",
        "arm", "n", "floor mean", "floor sd", "snr mean", "hnr mean", "hnr sd", "f0 mean", "f0 sd"
    );
    println!("{}", "-".repeat(78));
    for (arm, floors) in &by_arm {
        let Some((fm, fsd)) = mean_sd(floors) else {
            continue;
        };
        let snr_mean = snr_by_arm
            .get(arm)
            .and_then(|s| mean_sd(s))
            .map_or(f64::NAN, |(m, _)| m);
        let (hnr_mean, hnr_sd) = hnr_by_arm
            .get(arm)
            .and_then(|h| mean_sd(h))
            .unwrap_or((f64::NAN, f64::NAN));
        let (f0_mean, f0_sd) = f0_by_arm
            .get(arm)
            .and_then(|f| mean_sd(f))
            .unwrap_or((f64::NAN, f64::NAN));
        println!(
            "{arm:<16} {:>3} {fm:>12.1} {fsd:>10.2} {snr_mean:>12.1} {hnr_mean:>10.2} \
             {hnr_sd:>8.2} {f0_mean:>9.0} {f0_sd:>7.1}",
            floors.len()
        );
    }

    // The HNR comparison is the one that can actually see noise under the
    // voice; the floor map is returned alongside it for the pause-level view.
    if let (Some(c), Some(n)) = (hnr_by_arm.get("clean"), hnr_by_arm.get("noisy")) {
        let (cm, _) = mean_sd(c).unwrap_or((f64::NAN, f64::NAN));
        let (nm, _) = mean_sd(n).unwrap_or((f64::NAN, f64::NAN));
        if let Some((t, df, se)) = welch(c, n) {
            // Signed so a POSITIVE delta means the noisy prompt produced LESS
            // harmonic audio, i.e. the description worked in the stated direction.
            println!(
                "\nclean - noisy HNR delta: {:+.2} dB   SE {se:.2}   t {:.2}   df {df:.1}",
                cm - nm,
                -t
            );
            if -t > 2.0 && (cm - nm) > 1.0 {
                println!("=> the description MOVES noise under the voice. Label, do not denoise.");
            } else {
                println!("=> HNR does not separate the arms either.");
            }
        }
    }

    Ok(by_arm)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let probe_path = args
        .next()
        .ok_or("usage: noise_probe_decode PROBE_JSON NEUCODEC_DIR [OUT_DIR]")?;
    let codec_dir = args
        .next()
        .ok_or("usage: noise_probe_decode PROBE_JSON NEUCODEC_DIR [OUT_DIR]")?;
    let out_dir = args.next();

    let probe: Probe = serde_json::from_str(&std::fs::read_to_string(&probe_path)?)?;
    let use_cuda = std::env::var("PROBE_DEVICE").as_deref() == Ok("cuda");

    if let Some(dir) = &out_dir {
        std::fs::create_dir_all(dir)?;
    }

    println!("loading NeuCodec from {codec_dir} ...");
    let by_arm = if use_cuda {
        #[cfg(feature = "cuda")]
        {
            use boostr::runtime::cuda::{CudaDevice, CudaRuntime};
            let device = CudaDevice::new(0);
            let client = CudaRuntime::default_client(&device);
            let codec = NeuCodec::<CudaRuntime>::from_safetensors(&codec_dir, &device)?;
            println!("device  : CUDA");
            decode_all(&client, &device, &codec, &probe, out_dir.as_ref())?
        }
        #[cfg(not(feature = "cuda"))]
        {
            return Err(
                "PROBE_DEVICE=cuda but this binary was built without --features cuda".into(),
            );
        }
    } else {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let codec = NeuCodec::<CpuRuntime>::from_safetensors(&codec_dir, &device)?;
        println!("device  : CPU");
        decode_all(&client, &device, &codec, &probe, out_dir.as_ref())?
    };

    match (by_arm.get("clean"), by_arm.get("noisy")) {
        (Some(c), Some(n)) => {
            let (cm, _) = mean_sd(c).unwrap_or((f64::NAN, f64::NAN));
            let (nm, _) = mean_sd(n).unwrap_or((f64::NAN, f64::NAN));
            let delta = nm - cm;
            match welch(c, n) {
                Some((t, df, se)) => {
                    println!(
                        "\nnoisy - clean floor delta: {delta:+.2} dB   SE {se:.2}   \
                         t {t:.2}   df {df:.1}"
                    );
                    if t > 2.0 && delta > 1.0 {
                        println!(
                            "=> the description MOVES the noise floor. Label the corpus \
                             rather than denoising it."
                        );
                    } else if t > 2.0 {
                        println!(
                            "=> separation is statistically real but under 1 dB: too small to act on."
                        );
                    } else {
                        println!("=> not separated at this n. Either the field is decorative, or");
                        println!("   the floor metric is measuring the wrong thing (see below).");
                    }
                }
                None => {
                    println!("\nnoisy - clean floor delta: {delta:+.2} dB (n too small to test)")
                }
            }
        }
        _ => println!("\nclean/noisy arms missing — cannot compare."),
    }

    let quietest = by_arm
        .values()
        .flatten()
        .fold(f64::INFINITY, |a: f64, &b| a.min(b));
    if quietest < -70.0 {
        println!(
            "\nNOTE: floors reach {quietest:.0} dBFS — effectively digital silence in the pauses."
        );
        println!("      This metric probes PAUSES. Hiss during SPEECH would not show up here.");
    }

    Ok(())
}
