//! The iSTFT vocoder tail on CUDA, checked against CPU at the sizes that matter.
//!
//! Run with:
//!   `cd boostr && cargo test --release --features cuda --test istft_cuda_parity`
//!
//! Why this test exists: `istft` was `CpuRuntime`-only, on the grounds that numr
//! exposed no `scatter_add` for the overlap-add and no non-power-of-two `irfft`.
//! Both gaps are closed (`scatter_reduce` with `Sum`, and Bluestein on CUDA), so
//! the function is now generic — but "it compiles for CudaRuntime" is not
//! evidence that it computes the same waveform. The overlap-add is an
//! accumulating scatter with many colliding writes, which is precisely the shape
//! that produces a plausible-but-wrong result if the index arithmetic or the
//! accumulation order is off.
//!
//! `n_fft = 1920` is not an arbitrary choice: it is NeuCodec's real vocoder size
//! and it is not a power of two, so it exercises the Bluestein path rather than
//! the Stockham one.

#![cfg(feature = "cuda")]

use boostr::model::audio::kokoro::{IStftOptions, IStftPadding, hann_window, istft};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::runtime::cuda::{CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

/// Deterministic pseudo-random spectrogram: a fixed LCG rather than a constant,
/// because a constant magnitude makes every frame identical and would hide an
/// error in the per-frame indexing.
fn spectrogram(b: usize, f: usize, t: usize) -> (Vec<f32>, Vec<f32>) {
    let mut state = 0x2545_F491u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1 << 24) as f32
    };
    let n = b * f * t;
    let mag: Vec<f32> = (0..n).map(|_| next() * 2.0).collect();
    let phase: Vec<f32> = (0..n)
        .map(|_| (next() - 0.5) * 2.0 * std::f32::consts::PI)
        .collect();
    (mag, phase)
}

fn run_case(n_fft: usize, hop: usize, t_frames: usize, batch: usize, padding: IStftPadding) {
    let f = n_fft / 2 + 1;
    let (mag_data, phase_data) = spectrogram(batch, f, t_frames);
    let opts = IStftOptions {
        hop_length: hop,
        padding,
        eps: 1e-8,
    };

    let cpu_device = CpuDevice::new();
    let cpu_client = CpuClient::new(cpu_device.clone());
    let shape = [batch, f, t_frames];
    let cpu_mag = Tensor::<CpuRuntime>::from_slice(&mag_data, &shape, &cpu_device).unwrap();
    let cpu_phase = Tensor::<CpuRuntime>::from_slice(&phase_data, &shape, &cpu_device).unwrap();
    let cpu_window = hann_window::<CpuRuntime>(n_fft, &cpu_device).unwrap();
    let cpu_out = istft(&cpu_client, &cpu_mag, &cpu_phase, &cpu_window, opts)
        .expect("cpu istft must succeed");
    let cpu_shape = cpu_out.shape().to_vec();
    let cpu_samples: Vec<f32> = cpu_out.to_vec();

    let cuda_device = CudaDevice::new(0);
    let cuda_client = CudaRuntime::default_client(&cuda_device);
    let cuda_mag = Tensor::<CudaRuntime>::from_slice(&mag_data, &shape, &cuda_device).unwrap();
    let cuda_phase = Tensor::<CudaRuntime>::from_slice(&phase_data, &shape, &cuda_device).unwrap();
    let cuda_window = hann_window::<CudaRuntime>(n_fft, &cuda_device).unwrap();
    let cuda_out = istft(&cuda_client, &cuda_mag, &cuda_phase, &cuda_window, opts)
        .expect("cuda istft must succeed");

    assert_eq!(
        cuda_out.shape(),
        cpu_shape.as_slice(),
        "n_fft={n_fft} shape mismatch"
    );
    let cuda_samples: Vec<f32> = cuda_out.to_vec();

    // Scale the tolerance to the signal: the waveform amplitude here is O(1),
    // but the un-normalized overlap-add sums n_fft/hop frames of an inverse
    // transform of n_fft terms, so absolute error grows with n_fft.
    let tol = 1e-5 * (n_fft as f32).sqrt();
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (i, (c, g)) in cpu_samples.iter().zip(cuda_samples.iter()).enumerate() {
        let d = (c - g).abs();
        if d > worst {
            worst = d;
            worst_at = i;
        }
    }
    assert!(
        worst < tol,
        "n_fft={n_fft} hop={hop} pad={padding:?}: worst |cpu-cuda| = {worst} at sample \
         {worst_at} (cpu {}, cuda {}), tol {tol}",
        cpu_samples[worst_at],
        cuda_samples[worst_at]
    );

    // A tolerance check passes trivially if both sides are silent. The signal
    // must actually be there.
    let peak = cpu_samples.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(
        peak > 1e-3,
        "n_fft={n_fft}: CPU output is silent, test is inert"
    );
}

#[test]
fn neucodec_vocoder_size_matches_cpu() {
    // NeuCodec's real geometry: n_fft = 1920 (NOT a power of two), hop = 480,
    // Vocos `padding="same"`.
    run_case(1920, 480, 6, 1, IStftPadding::Same);
}

#[test]
fn neucodec_vocoder_size_matches_cpu_batched() {
    // A batch dimension catches per-row index arithmetic in the scatter.
    run_case(1920, 480, 4, 3, IStftPadding::Same);
}

#[test]
fn kokoro_vocoder_size_matches_cpu() {
    // Kokoro's n_fft = 20, also non-power-of-two, at the opposite extreme of
    // size — small enough that M = 64 and the shared-memory FFT path is used.
    run_case(20, 5, 12, 2, IStftPadding::Center);
}

#[test]
fn power_of_two_size_matches_cpu() {
    // The Stockham path, so a Bluestein-specific bug cannot mask a general
    // overlap-add bug (or vice versa).
    run_case(256, 64, 8, 2, IStftPadding::Center);
}

#[test]
fn untrimmed_output_matches_cpu() {
    // `None` padding returns the full (T-1)*hop + n_fft overlap-add, including
    // the boundary region where the window-square sum approaches zero and the
    // eps mask decides between a division and a hard zero.
    run_case(64, 16, 5, 1, IStftPadding::None);
}
