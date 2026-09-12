//! The [`Init`] enum and PyTorch's `fan_in` convention.

/// PyTorch's `fan_in` for a weight tensor, matching
/// `torch.nn.init._calculate_fan_in_and_fan_out`.
///
/// PyTorch computes `fan_in = tensor.size(1) * prod(tensor.shape[2:])`, i.e.
/// the product of every dimension EXCEPT the leading output dimension.
///
/// - `Linear` weight `[out_features, in_features]` → `in_features`. This layout
///   is fixed by [`crate::nn::Linear::forward`], which computes
///   `input @ weight^T` and reads `out_features` from `shape[0]`.
/// - `Conv1d` weight `[out_channels, in_channels / groups, kernel]` →
///   `in_channels / groups * kernel`. The division by `groups` is already
///   baked into the stored `shape[1]`, so a depthwise conv `[C, 1, K]`
///   correctly yields `K`, not `C` and not `1`.
///
/// PyTorch rejects tensors with fewer than 2 dimensions. A 1-D tensor has no
/// separate input dimension, so its own length is the only defensible fan_in.
pub(super) fn pytorch_fan_in(shape: &[usize]) -> usize {
    if shape.len() < 2 {
        return shape.first().copied().unwrap_or(1);
    }
    shape[1..].iter().product()
}

/// Initialization strategy for new tensors.
#[derive(Debug, Clone, Copy)]
pub enum Init {
    /// All zeros
    Zeros,
    /// All ones
    Ones,
    /// Constant value
    Const(f32),
    /// Uniform random in `[-bound, bound]`
    Uniform(f32),
    /// Kaiming uniform (PyTorch `Linear`/`Conv` default):
    /// U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
    ///
    /// `fan_in` follows PyTorch exactly — see `pytorch_fan_in`. For a
    /// `[out_features, in_features]` weight that is `in_features`, NOT
    /// `out_features`: the leading dimension is the output side.
    PyTorchLinear,
    /// PyTorch Embedding default: N(0, 1)
    PyTorchEmbedding,
    /// Kaiming (He) normal: N(0, sqrt(2 / fan_in))
    ///
    /// Standard initialization for ReLU networks. `fan_in` follows
    /// `pytorch_fan_in`, the same convention as [`Init::PyTorchLinear`]: for
    /// a `[out_features, in_features]` weight it is `in_features`.
    ///
    /// This previously read `fan_in` off the OPPOSITE end of the shape, so for
    /// the `[out, in]` layout every `Linear` in this workspace stores it
    /// scaled by `fan_out` instead. That inflates or deflates the initial
    /// variance by `out/in` — invisible on a square weight, and wrong on every
    /// other one.
    Kaiming,
    /// Xavier (Glorot) normal: N(0, sqrt(2 / (fan_in + fan_out)))
    ///
    /// Standard initialization for Sigmoid/Tanh networks. Uses the same
    /// `[out_features, in_features]` convention as [`Init::PyTorchLinear`].
    ///
    /// Xavier is symmetric in `fan_in + fan_out`, so a 2-D weight was already
    /// unaffected by the layout mix-up this shares with [`Init::Kaiming`]. A
    /// 3-D or higher weight was not: `fan_in` and `fan_out` split the
    /// dimensions differently, so their sum changed.
    Xavier,
    /// Normal distribution with given mean and standard deviation.
    Randn { mean: f64, stdev: f64 },
    /// Truncated normal: N(mean, stdev) clamped to [mean - 2*stdev, mean + 2*stdev]
    ///
    /// Used by GPT-2, BERT, and most modern LLMs for training stability.
    TruncatedNormal { mean: f64, stdev: f64 },
}

/// Xavier's `(fan_in, fan_out)` split: leading dim is the output side,
/// matching `pytorch_fan_in`. A 1-D shape uses its own length for both.
pub(super) fn xavier_fans(shape: &[usize]) -> (usize, usize) {
    if shape.len() >= 2 {
        (pytorch_fan_in(shape), shape[0])
    } else {
        let n = shape.first().copied().unwrap_or(1);
        (n, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::dtype::DType;
    use numr::runtime::Runtime;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn device() -> CpuDevice {
        CpuDevice::new()
    }

    fn client() -> numr::runtime::cpu::CpuClient {
        let d = device();
        CpuRuntime::default_client(&d)
    }

    // ===== `Init::PyTorchLinear` fan_in =====
    //
    // PyTorch's `_calculate_fan_in_and_fan_out` uses
    // `fan_in = size(1) * prod(shape[2:])` — every dimension except the leading
    // output dimension. `Linear` stores its weight `[out_features, in_features]`
    // (`Linear::forward` computes `input @ weight^T`), so the fan_in of a Linear
    // weight is `shape[1]`.

    /// The largest absolute value in a uniform draw of `n` samples over
    /// `[-b, b]` lands within `b * (1 - 2/n)` of `b` in expectation, so at
    /// 8192 samples the observed maximum pins `b` to well under a percent.
    fn max_abs(data: &[f32]) -> f32 {
        data.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    }

    /// A non-square weight must be bounded by `1/sqrt(shape[1])`.
    ///
    /// `[8192, 2048]` is the real `gate_proj` shape, whose two dimensions differ
    /// by 4x. The two candidate bounds are therefore 2x apart and cannot be
    /// confused: correct is `1/sqrt(2048) = 0.02210`, the defect gives
    /// `1/sqrt(8192) = 0.01105`.
    ///
    /// Drawing the full 16.7M-element tensor is wasteful, so this uses
    /// `[128, 32]` — the same 4x ratio, the same 2x bound separation — and
    /// checks the observed maximum sits just under the correct bound and far
    /// above the wrong one.
    #[test]
    fn test_pytorch_linear_fan_in_is_the_trailing_dim() {
        let d = device();
        let c = client();

        let shape = &[128, 32];
        let correct_bound = 1.0f32 / 32.0f32.sqrt(); // 0.176777
        let wrong_bound = 1.0f32 / 128.0f32.sqrt(); // 0.088388

        let t = Init::PyTorchLinear
            .init_tensor(shape, DType::F32, &d, &c)
            .unwrap();
        let data: Vec<f32> = t.to_vec();
        assert_eq!(data.len(), 4096);

        let observed = max_abs(&data);
        assert!(
            observed <= correct_bound,
            "value {observed} exceeds the fan_in={} bound {correct_bound}",
            shape[1]
        );
        // 4096 samples over [-b, b]: P(max < 0.95*b) = 0.95^4096, i.e. zero.
        assert!(
            observed > 0.95 * correct_bound,
            "observed max {observed} is far below the fan_in={} bound \
             {correct_bound}; it looks bounded by the fan_in={} value {wrong_bound}",
            shape[1],
            shape[0]
        );
    }

    /// A square weight is bounded identically before and after the fix — this is
    /// why the defect survived. `q_proj [2048, 2048]` is the real case; `[64, 64]`
    /// is the same situation at a testable size.
    #[test]
    fn test_pytorch_linear_square_weight_bound_is_unchanged() {
        let d = device();
        let c = client();

        let shape = &[64, 64];
        let bound = 1.0f32 / 64.0f32.sqrt(); // shape[0] and shape[1] agree

        let t = Init::PyTorchLinear
            .init_tensor(shape, DType::F32, &d, &c)
            .unwrap();
        let observed = max_abs(&t.to_vec::<f32>());
        assert!(
            observed <= bound && observed > 0.95 * bound,
            "square max {observed} does not match bound {bound}"
        );
    }

    /// A depthwise `Conv1d` weight is stored `[channels, 1, kernel]`, and every
    /// Mamba layer initializes `conv1d.weight` at that rank with
    /// `Init::PyTorchLinear`. PyTorch's fan_in there is
    /// `in_channels/groups * kernel = 1 * kernel`, which `shape[1..].product()`
    /// gives. A literal `shape[1]` would return 1 (bound 1.0, ~7x too wide) and
    /// `shape[0]` returns the channel count (far too narrow), so this pins the
    /// only correct reading.
    #[test]
    fn test_pytorch_linear_conv1d_fan_in_is_kernel_size() {
        let d = device();
        let c = client();

        let shape = &[256, 1, 4]; // channels=256, depthwise, d_conv=4
        let correct_bound = 1.0f32 / 4.0f32.sqrt(); // 0.5

        let t = Init::PyTorchLinear
            .init_tensor(shape, DType::F32, &d, &c)
            .unwrap();
        let observed = max_abs(&t.to_vec::<f32>());
        assert!(
            observed <= correct_bound && observed > 0.95 * correct_bound,
            "conv1d max {observed} does not match kernel-size bound {correct_bound}"
        );
    }

    /// Non-square shape whose two fan directions differ by 4x, so a convention
    /// mix-up moves the standard deviation by a factor of 2.
    const OUT_FEATURES: usize = 8;
    const IN_FEATURES: usize = 32;

    #[test]
    fn kaiming_reads_fan_in_from_the_trailing_dimensions() {
        // `[out_features, in_features]` is what every Linear in this workspace
        // stores, so fan_in must be in_features. Reading it off the leading
        // dimension instead gives sqrt(2/8) rather than sqrt(2/32) — twice the
        // spread, and silently so on any square weight.
        let d = device();
        let c = client();
        let shape = [OUT_FEATURES, IN_FEATURES];
        let t = Init::Kaiming
            .init_tensor(&shape, DType::F32, &d, &c)
            .expect("kaiming init");

        let v: Vec<f32> = t.to_vec();
        let n = v.len() as f64;
        let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
        let sd = (v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();

        let expected = (2.0 / IN_FEATURES as f64).sqrt();
        let wrong = (2.0 / OUT_FEATURES as f64).sqrt();
        // Sample sd of 256 draws sits well inside 25% of the true sd, and the two
        // candidates differ by 2x, so this cannot confuse them.
        assert!(
            (sd - expected).abs() < 0.25 * expected,
            "sd {sd:.4} should be near {expected:.4} (fan_in = in_features), not {wrong:.4}"
        );
    }

    #[test]
    fn xavier_agrees_with_kaiming_about_which_dimension_is_fan_in() {
        // Xavier is symmetric in fan_in + fan_out, so a 2-D weight cannot detect a
        // swap. A 3-D weight can: fan_in collapses the trailing dims while fan_out
        // stays the leading one, so the sum differs.
        let d = device();
        let c = client();
        let shape = [4usize, 8, 16]; // fan_in = 8*16 = 128, fan_out = 4
        let t = Init::Xavier
            .init_tensor(&shape, DType::F32, &d, &c)
            .expect("xavier init");

        let v: Vec<f32> = t.to_vec();
        let n = v.len() as f64;
        let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
        let sd = (v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();

        let expected = (2.0f64 / (128.0 + 4.0)).sqrt();
        let swapped = (2.0f64 / (32.0 + 16.0)).sqrt(); // the old split: 4*8 and 16
        assert!(
            (sd - expected).abs() < 0.25 * expected,
            "sd {sd:.4} should be near {expected:.4}, not the swapped {swapped:.4}"
        );
    }
}
