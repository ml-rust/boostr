//! Analysis/synthesis windows for Kokoro's STFT/iSTFT vocoder paths.

use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// Periodic Hann window of length `n` on the given device.
///
/// Uses the periodic (DFT-even) definition `0.5 - 0.5·cos(2π·i/n)`, matching
/// PyTorch's `torch.hann_window(n, periodic=True)` — the convention Kokoro's
/// `TorchSTFT` uses for both analysis and synthesis.
pub fn hann_window(n: usize, device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
    use std::f32::consts::PI;
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let ratio = i as f32 / n.max(1) as f32;
            0.5 - 0.5 * (2.0 * PI * ratio).cos()
        })
        .collect();
    Tensor::<CpuRuntime>::from_slice(&data, &[n], device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;

    #[test]
    fn hann_window_endpoints_are_zero() {
        let (_client, device) = cpu_setup();
        let w = hann_window(8, &device);
        let v: Vec<f32> = w.to_vec();
        assert!(v[0].abs() < 1e-6);
        // Hann is symmetric around the midpoint; the mid value peaks near 1.
        assert!(v[4] > 0.9);
    }
}
