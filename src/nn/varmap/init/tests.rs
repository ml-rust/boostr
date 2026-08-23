use super::*;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};

fn device() -> CpuDevice {
    CpuDevice::new()
}

fn client() -> numr::runtime::cpu::CpuClient {
    let d = device();
    CpuRuntime::default_client(&d)
}

#[test]
fn test_init_zeros() {
    let d = device();
    let c = client();
    let t = Init::Zeros
        .init_tensor(&[2, 3], DType::F32, &d, &c)
        .unwrap();
    assert_eq!(t.shape(), &[2, 3]);
    let data: Vec<f32> = t.to_vec();
    assert!(data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_init_kaiming() {
    let d = device();
    let c = client();
    // [out=64, in=128] → fan_in=128, std=sqrt(2/128)≈0.125
    let t = Init::Kaiming
        .init_tensor(&[64, 128], DType::F32, &d, &c)
        .unwrap();
    assert_eq!(t.shape(), &[64, 128]);
    let data: Vec<f32> = t.to_vec();
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    // Mean should be close to 0
    assert!(mean.abs() < 0.1, "Kaiming mean too large: {mean}");
    // Std should be close to sqrt(2/128) ≈ 0.125
    let var: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    let std = var.sqrt();
    // fan_in = product of all dims except last = 64
    let expected_std = (2.0f32 / 64.0).sqrt();
    assert!(
        (std - expected_std).abs() < 0.05,
        "Kaiming std {std} vs expected {expected_std}"
    );
}

#[test]
fn test_init_xavier() {
    let d = device();
    let c = client();
    // [256, 512] → fan_in=256, fan_out=512, std=sqrt(2/768)≈0.051
    let t = Init::Xavier
        .init_tensor(&[256, 512], DType::F32, &d, &c)
        .unwrap();
    assert_eq!(t.shape(), &[256, 512]);
    let data: Vec<f32> = t.to_vec();
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!(mean.abs() < 0.05, "Xavier mean too large: {mean}");
}

#[test]
fn test_init_randn() {
    let d = device();
    let c = client();
    let t = Init::Randn {
        mean: 5.0,
        stdev: 0.1,
    }
    .init_tensor(&[1000], DType::F32, &d, &c)
    .unwrap();
    let data: Vec<f32> = t.to_vec();
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!((mean - 5.0).abs() < 0.1, "Randn mean {mean} should be ~5.0");
}

#[test]
fn test_init_truncated_normal() {
    let d = device();
    let c = client();
    let t = Init::TruncatedNormal {
        mean: 0.0,
        stdev: 0.02,
    }
    .init_tensor(&[10000], DType::F32, &d, &c)
    .unwrap();
    let data: Vec<f32> = t.to_vec();
    // All values should be within [-0.04, 0.04] (2*stdev)
    for &v in &data {
        assert!(
            (-0.04..=0.04).contains(&v),
            "Truncated normal value {v} out of range [-0.04, 0.04]"
        );
    }
}

// ===== Seeded-path tests =====
//
// Shape is [4096] (16384 bytes of f32) for every random variant below: at
// that size, two independent unseeded draws colliding bit-for-bit by chance
// is not a real possibility, so an equality assertion there is a genuine
// discriminator, not a coincidence.
const BIG: &[usize] = &[4096];

#[test]
fn test_seeded_uniform_same_seed_bit_identical() {
    let d = device();
    let c = client();
    let a = Init::Uniform(0.5)
        .init_tensor_seeded(BIG, DType::F32, &d, &c, 42)
        .unwrap();
    let b = Init::Uniform(0.5)
        .init_tensor_seeded(BIG, DType::F32, &d, &c, 42)
        .unwrap();
    assert_eq!(a.to_vec::<f32>(), b.to_vec::<f32>());
}

#[test]
fn test_seeded_uniform_different_seed_differs() {
    let d = device();
    let c = client();
    let a = Init::Uniform(0.5)
        .init_tensor_seeded(BIG, DType::F32, &d, &c, 1)
        .unwrap();
    let b = Init::Uniform(0.5)
        .init_tensor_seeded(BIG, DType::F32, &d, &c, 2)
        .unwrap();
    assert_ne!(a.to_vec::<f32>(), b.to_vec::<f32>());
}

#[test]
fn test_seeded_pytorch_linear_same_seed_bit_identical() {
    let d = device();
    let c = client();
    let shape = &[128, 64];
    let a = Init::PyTorchLinear
        .init_tensor_seeded(shape, DType::F32, &d, &c, 7)
        .unwrap();
    let b = Init::PyTorchLinear
        .init_tensor_seeded(shape, DType::F32, &d, &c, 7)
        .unwrap();
    assert_eq!(a.to_vec::<f32>(), b.to_vec::<f32>());
}

#[test]
fn test_seeded_pytorch_linear_different_seed_differs() {
    let d = device();
    let c = client();
    let shape = &[128, 64];
    let a = Init::PyTorchLinear
        .init_tensor_seeded(shape, DType::F32, &d, &c, 7)
        .unwrap();
    let b = Init::PyTorchLinear
        .init_tensor_seeded(shape, DType::F32, &d, &c, 8)
        .unwrap();
    assert_ne!(a.to_vec::<f32>(), b.to_vec::<f32>());
}

#[test]
fn test_seeded_pytorch_embedding_same_seed_bit_identical() {
    let d = device();
    let c = client();
    let a = Init::PyTorchEmbedding
        .init_tensor_seeded(BIG, DType::F32, &d, &c, 3)
        .unwrap();
    let b = Init::PyTorchEmbedding
        .init_tensor_seeded(BIG, DType::F32, &d, &c, 3)
        .unwrap();
    assert_eq!(a.to_vec::<f32>(), b.to_vec::<f32>());
}

#[test]
fn test_seeded_pytorch_embedding_different_seed_differs() {
    let d = device();
    let c = client();
    let a = Init::PyTorchEmbedding
        .init_tensor_seeded(BIG, DType::F32, &d, &c, 3)
        .unwrap();
    let b = Init::PyTorchEmbedding
        .init_tensor_seeded(BIG, DType::F32, &d, &c, 4)
        .unwrap();
    assert_ne!(a.to_vec::<f32>(), b.to_vec::<f32>());
}

#[test]
fn test_seeded_kaiming_same_seed_bit_identical() {
    let d = device();
    let c = client();
    let shape = &[64, 128];
    let a = Init::Kaiming
        .init_tensor_seeded(shape, DType::F32, &d, &c, 11)
        .unwrap();
    let b = Init::Kaiming
        .init_tensor_seeded(shape, DType::F32, &d, &c, 11)
        .unwrap();
    assert_eq!(a.to_vec::<f32>(), b.to_vec::<f32>());
}

#[test]
fn test_seeded_kaiming_different_seed_differs() {
    let d = device();
    let c = client();
    let shape = &[64, 128];
    let a = Init::Kaiming
        .init_tensor_seeded(shape, DType::F32, &d, &c, 11)
        .unwrap();
    let b = Init::Kaiming
        .init_tensor_seeded(shape, DType::F32, &d, &c, 12)
        .unwrap();
    assert_ne!(a.to_vec::<f32>(), b.to_vec::<f32>());
}

#[test]
fn test_seeded_xavier_same_seed_bit_identical() {
    let d = device();
    let c = client();
    let shape = &[256, 512];
    let a = Init::Xavier
        .init_tensor_seeded(shape, DType::F32, &d, &c, 13)
        .unwrap();
    let b = Init::Xavier
        .init_tensor_seeded(shape, DType::F32, &d, &c, 13)
        .unwrap();
    assert_eq!(a.to_vec::<f32>(), b.to_vec::<f32>());
}

#[test]
fn test_seeded_xavier_different_seed_differs() {
    let d = device();
    let c = client();
    let shape = &[256, 512];
    let a = Init::Xavier
        .init_tensor_seeded(shape, DType::F32, &d, &c, 13)
        .unwrap();
    let b = Init::Xavier
        .init_tensor_seeded(shape, DType::F32, &d, &c, 14)
        .unwrap();
    assert_ne!(a.to_vec::<f32>(), b.to_vec::<f32>());
}

#[test]
fn test_seeded_randn_same_seed_bit_identical() {
    let d = device();
    let c = client();
    let init = Init::Randn {
        mean: 5.0,
        stdev: 0.1,
    };
    let a = init
        .init_tensor_seeded(BIG, DType::F32, &d, &c, 21)
        .unwrap();
    let b = init
        .init_tensor_seeded(BIG, DType::F32, &d, &c, 21)
        .unwrap();
    assert_eq!(a.to_vec::<f32>(), b.to_vec::<f32>());
}

#[test]
fn test_seeded_randn_different_seed_differs() {
    let d = device();
    let c = client();
    let init = Init::Randn {
        mean: 5.0,
        stdev: 0.1,
    };
    let a = init
        .init_tensor_seeded(BIG, DType::F32, &d, &c, 21)
        .unwrap();
    let b = init
        .init_tensor_seeded(BIG, DType::F32, &d, &c, 22)
        .unwrap();
    assert_ne!(a.to_vec::<f32>(), b.to_vec::<f32>());
}

#[test]
fn test_seeded_truncated_normal_same_seed_bit_identical() {
    let d = device();
    let c = client();
    let init = Init::TruncatedNormal {
        mean: 0.0,
        stdev: 0.02,
    };
    let a = init
        .init_tensor_seeded(BIG, DType::F32, &d, &c, 31)
        .unwrap();
    let b = init
        .init_tensor_seeded(BIG, DType::F32, &d, &c, 31)
        .unwrap();
    assert_eq!(a.to_vec::<f32>(), b.to_vec::<f32>());
}

#[test]
fn test_seeded_truncated_normal_different_seed_differs() {
    let d = device();
    let c = client();
    let init = Init::TruncatedNormal {
        mean: 0.0,
        stdev: 0.02,
    };
    let a = init
        .init_tensor_seeded(BIG, DType::F32, &d, &c, 31)
        .unwrap();
    let b = init
        .init_tensor_seeded(BIG, DType::F32, &d, &c, 32)
        .unwrap();
    assert_ne!(a.to_vec::<f32>(), b.to_vec::<f32>());
}

/// The assertion that proves the seeded path does something the unseeded
/// path does not: two unseeded draws of the same `Init` variant must NOT
/// collide. `BIG` is large enough (4096 f32 values) that a coincidental
/// bit-for-bit match is not a real possibility, so a failure here means the
/// unseeded RNG stream is broken (e.g. accidentally reusing a fixed seed),
/// not bad luck.
#[test]
fn test_unseeded_init_tensor_twice_differs() {
    let d = device();
    let c = client();
    let a = Init::Randn {
        mean: 0.0,
        stdev: 1.0,
    }
    .init_tensor(BIG, DType::F32, &d, &c)
    .unwrap();
    let b = Init::Randn {
        mean: 0.0,
        stdev: 1.0,
    }
    .init_tensor(BIG, DType::F32, &d, &c)
    .unwrap();
    assert_ne!(a.to_vec::<f32>(), b.to_vec::<f32>());
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

/// The seeded twin must agree with the unseeded path on the distribution.
/// They draw different numbers, so this compares the bound each respects,
/// not the values. If only one path is fixed the bounds differ by 2x and
/// this fails.
#[test]
fn test_pytorch_linear_seeded_and_unseeded_share_a_distribution() {
    let d = device();
    let c = client();

    let shape = &[128, 32];
    let correct_bound = 1.0f32 / 32.0f32.sqrt();

    let unseeded = Init::PyTorchLinear
        .init_tensor(shape, DType::F32, &d, &c)
        .unwrap();
    let seeded = Init::PyTorchLinear
        .init_tensor_seeded(shape, DType::F32, &d, &c, 4242)
        .unwrap();

    let m_unseeded = max_abs(&unseeded.to_vec::<f32>());
    let m_seeded = max_abs(&seeded.to_vec::<f32>());

    for (label, observed) in [("unseeded", m_unseeded), ("seeded", m_seeded)] {
        assert!(
            observed <= correct_bound && observed > 0.95 * correct_bound,
            "{label} max {observed} does not match bound {correct_bound}"
        );
    }
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

/// Deterministic variants ignore the seed entirely: seeded and unseeded
/// paths must agree bit-for-bit.
#[test]
fn test_deterministic_variants_seeded_matches_unseeded() {
    let d = device();
    let c = client();
    for init in [Init::Zeros, Init::Ones, Init::Const(3.5)] {
        let unseeded = init.init_tensor(&[8, 8], DType::F32, &d, &c).unwrap();
        let seeded = init
            .init_tensor_seeded(&[8, 8], DType::F32, &d, &c, 999)
            .unwrap();
        assert_eq!(
            unseeded.to_vec::<f32>(),
            seeded.to_vec::<f32>(),
            "{init:?} must be identical whether seeded or not"
        );
    }
}
