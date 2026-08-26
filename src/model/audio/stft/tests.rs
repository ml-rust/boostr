#![allow(clippy::useless_vec)]

use super::*;
use crate::model::audio::kokoro::{IStftOptions, IStftPadding, istft};
use crate::test_utils::cpu_setup;
use numr::runtime::cpu::CpuRuntime;

fn tensor(
    data: &[f32],
    shape: &[usize],
    device: &<CpuRuntime as Runtime>::Device,
) -> Tensor<CpuRuntime> {
    Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap()
}

/// Periodic Hann window, the analysis/synthesis pair `istft` normalizes for.
fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos() as f32)
        .collect()
}

#[test]
fn output_shape_follows_formula_without_center() {
    let (client, device) = cpu_setup();
    let n_fft = 8;
    let wave = tensor(&vec![0.0f32; 16], &[1, 16], &device);
    let win = tensor(&vec![1.0f32; n_fft], &[n_fft], &device);
    let (mag, phase) = stft(
        &client,
        &wave,
        &win,
        StftOptions {
            n_fft,
            hop_length: 4,
            center: false,
        },
    )
    .unwrap();
    // T_spec = (16 - 8)/4 + 1 = 3, F = 5.
    assert_eq!(mag.shape(), &[1, 5, 3]);
    assert_eq!(phase.shape(), &[1, 5, 3]);
}

#[test]
fn output_shape_includes_center_padding() {
    let (client, device) = cpu_setup();
    let n_fft = 8;
    let wave = tensor(&vec![0.0f32; 16], &[1, 16], &device);
    let win = tensor(&vec![1.0f32; n_fft], &[n_fft], &device);
    let (mag, _) = stft(
        &client,
        &wave,
        &win,
        StftOptions {
            n_fft,
            hop_length: 4,
            center: true,
        },
    )
    .unwrap();
    // padded = 16 + 2*4 = 24; T_spec = (24-8)/4 + 1 = 5.
    assert_eq!(mag.shape(), &[1, 5, 5]);
}

#[test]
fn zero_signal_produces_zero_magnitude() {
    let (client, device) = cpu_setup();
    let wave = tensor(&vec![0.0f32; 32], &[1, 32], &device);
    let win = tensor(&vec![1.0f32; 8], &[8], &device);
    let (mag, _) = stft(
        &client,
        &wave,
        &win,
        StftOptions {
            n_fft: 8,
            hop_length: 4,
            center: false,
        },
    )
    .unwrap();
    for v in mag.to_vec::<f32>() {
        assert!(v.abs() < 1e-5);
    }
}

#[test]
fn constant_signal_concentrates_at_dc() {
    let (client, device) = cpu_setup();
    let wave = tensor(&vec![1.0f32; 16], &[1, 16], &device);
    let win = tensor(&vec![1.0f32; 4], &[4], &device);
    let (mag, _) = stft(
        &client,
        &wave,
        &win,
        StftOptions {
            n_fft: 4,
            hop_length: 2,
            center: false,
        },
    )
    .unwrap();
    let v: Vec<f32> = mag.to_vec();
    let t_spec = 7; // (16-4)/2+1
    // Bin 0 at each time equals the window sum, 4.
    for (t, &dc) in v.iter().take(t_spec).enumerate() {
        assert!((dc - 4.0).abs() < 1e-4, "DC bin at t={t}: {dc}");
    }
    for k in 1..3 {
        for t in 0..t_spec {
            let v = v[k * t_spec + t];
            assert!(v.abs() < 1e-4, "bin {k}, t {t}: {v}");
        }
    }
}

#[test]
fn a_pure_tone_lands_in_its_own_bin() {
    // The test the shape checks cannot do: proves the transform is a DFT and
    // not merely the right size. A sinusoid at exactly bin k has all its
    // energy there.
    let (client, device) = cpu_setup();
    let n_fft = 64;
    let bin = 8usize;
    let samples: Vec<f32> = (0..n_fft * 4)
        .map(|i| (std::f64::consts::TAU * bin as f64 * i as f64 / n_fft as f64).sin() as f32)
        .collect();
    let wave = tensor(&samples, &[1, samples.len()], &device);
    let win = tensor(&vec![1.0f32; n_fft], &[n_fft], &device);
    let (mag, _) = stft(
        &client,
        &wave,
        &win,
        StftOptions {
            n_fft,
            hop_length: n_fft,
            center: false,
        },
    )
    .unwrap();
    let f = n_fft / 2 + 1;
    let t_spec = 4;
    let v: Vec<f32> = mag.to_vec();
    let peak = (0..f)
        .max_by(|&a, &b| v[a * t_spec].partial_cmp(&v[b * t_spec]).unwrap())
        .unwrap();
    assert_eq!(peak, bin, "peak landed in bin {peak}, expected {bin}");
    // Amplitude 1.0 over n_fft samples with a rectangular window gives n_fft/2.
    assert!(
        (v[bin * t_spec] - n_fft as f32 / 2.0).abs() < 1e-2,
        "peak magnitude {}",
        v[bin * t_spec]
    );
}

#[test]
fn non_power_of_two_n_fft_works() {
    // Kokoro's generator uses n_fft = 20. The direct-DFT implementation this
    // replaced existed because `rfft` once rejected such sizes; numr's
    // Bluestein path now takes them, and this test is what pins that.
    let (client, device) = cpu_setup();
    let n_fft = 20usize;
    let samples: Vec<f32> = (0..200)
        .map(|i| (std::f64::consts::TAU * 4.0 * i as f64 / n_fft as f64).sin() as f32)
        .collect();
    let wave = tensor(&samples, &[1, samples.len()], &device);
    let win = tensor(&vec![1.0f32; n_fft], &[n_fft], &device);
    let (mag, _) = stft(
        &client,
        &wave,
        &win,
        StftOptions {
            n_fft,
            hop_length: n_fft,
            center: false,
        },
    )
    .unwrap();
    let f = n_fft / 2 + 1;
    let t_spec = 10;
    assert_eq!(mag.shape(), &[1, f, t_spec]);
    let v: Vec<f32> = mag.to_vec();
    let peak = (0..f)
        .max_by(|&a, &b| v[a * t_spec].partial_cmp(&v[b * t_spec]).unwrap())
        .unwrap();
    assert_eq!(peak, 4, "peak landed in bin {peak}, expected 4");
}

#[test]
fn stft_then_istft_reconstructs_the_waveform() {
    // The strongest available check: the forward transform must be the exact
    // inverse of the one already tested in `kokoro::istft`, phase included. A
    // magnitude-only bug or an off-by-one in the framing shows up here and
    // nowhere else.
    let (client, device) = cpu_setup();
    let n_fft = 32usize;
    let hop = 8usize;
    let samples: Vec<f32> = (0..512)
        .map(|i| {
            let t = i as f64;
            (0.6 * (t * 0.11).sin() + 0.3 * (t * 0.37 + 1.0).sin()) as f32
        })
        .collect();
    let wave = tensor(&samples, &[1, samples.len()], &device);
    let win = tensor(&hann(n_fft), &[n_fft], &device);

    let (mag, phase) = stft(
        &client,
        &wave,
        &win,
        StftOptions {
            n_fft,
            hop_length: hop,
            center: true,
        },
    )
    .unwrap();
    let back = istft(
        &client,
        &mag,
        &phase,
        &win,
        IStftOptions {
            hop_length: hop,
            padding: IStftPadding::Center,
            eps: 1e-8,
        },
    )
    .unwrap();

    let got: Vec<f32> = back.to_vec();
    // Center padding gives (T_spec - 1) * hop samples back.
    let n = got.len().min(samples.len());
    assert!(n >= samples.len() - hop, "reconstruction is {n} samples");
    // Skip the first and last frame: fewer windows overlap there, so those
    // samples are attenuated by design rather than reconstructed.
    for i in n_fft..n - n_fft {
        assert!(
            (got[i] - samples[i]).abs() < 1e-3,
            "sample {i}: {} vs {}",
            got[i],
            samples[i]
        );
    }
}

#[test]
fn rejects_wrong_window_size() {
    let (client, device) = cpu_setup();
    let wave = tensor(&vec![0.0f32; 16], &[1, 16], &device);
    let win = tensor(&vec![1.0f32; 5], &[5], &device);
    assert!(
        stft(
            &client,
            &wave,
            &win,
            StftOptions {
                n_fft: 8,
                hop_length: 4,
                center: false
            }
        )
        .is_err()
    );
}

#[test]
fn rejects_too_short_signal() {
    let (client, device) = cpu_setup();
    let wave = tensor(&vec![0.0f32; 4], &[1, 4], &device);
    let win = tensor(&vec![1.0f32; 8], &[8], &device);
    assert!(
        stft(
            &client,
            &wave,
            &win,
            StftOptions {
                n_fft: 8,
                hop_length: 4,
                center: false
            }
        )
        .is_err()
    );
}
