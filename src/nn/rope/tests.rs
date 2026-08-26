//! Tests for RoPE frequency precomputation and scaling.

use super::*;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};

#[test]
fn test_rope_precompute_shape() {
    let device = CpuDevice::new();
    let rope = RoPE::<CpuRuntime>::precompute_freqs(128, 64, 10000.0, None, &device).unwrap();
    assert_eq!(rope.cos_cache().shape(), &[128, 32]);
    assert_eq!(rope.sin_cache().shape(), &[128, 32]);
}

#[test]
fn test_rope_precompute_values() {
    let device = CpuDevice::new();
    let rope = RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, None, &device).unwrap();

    let cos: Vec<f32> = rope.cos_cache().tensor().to_vec();
    let sin: Vec<f32> = rope.sin_cache().tensor().to_vec();

    // pos=0: all cos=1, sin=0
    for i in 0..4 {
        assert!((cos[i] - 1.0).abs() < 1e-6, "cos[0,{i}]={}", cos[i]);
        assert!(sin[i].abs() < 1e-6, "sin[0,{i}]={}", sin[i]);
    }
}

#[test]
fn test_rope_forward() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let rope = RoPE::<CpuRuntime>::precompute_freqs(8, 16, 10000.0, None, &device).unwrap();

    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32; 2 * 4 * 16], &[1, 2, 4, 16], &device).unwrap(),
        false,
    );
    let out = rope.forward(&client, &x).unwrap();
    assert_eq!(out.shape(), &[1, 2, 4, 16]);
}

#[test]
fn test_rope_linear_scaling() {
    let device = CpuDevice::new();
    let cfg = RopeScalingConfig {
        scaling_type: "linear".to_string(),
        factor: 2.0,
        original_max_position_embeddings: None,
        low_freq_factor: None,
        high_freq_factor: None,
        attention_factor: None,
        beta_fast: None,
        beta_slow: None,
        short_factor: None,
        long_factor: None,
    };

    let unscaled = RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, None, &device).unwrap();
    let scaled = RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, Some(&cfg), &device).unwrap();

    let cos_unscaled: Vec<f32> = unscaled.cos_cache().tensor().to_vec();
    let cos_scaled: Vec<f32> = scaled.cos_cache().tensor().to_vec();

    // At pos=0, both should be all 1s (cos(0)=1)
    assert!((cos_scaled[0] - 1.0).abs() < 1e-6);

    // At pos=2 scaled should match pos=1 unscaled (freq halved → angle halved)
    let half_dim = 4;
    for i in 0..half_dim {
        let expected = cos_unscaled[half_dim + i]; // pos=1 unscaled
        let actual = cos_scaled[2 * half_dim + i]; // pos=2 scaled
        assert!(
            (actual - expected).abs() < 1e-5,
            "dim {i}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn test_rope_llama3_scaling() {
    let device = CpuDevice::new();
    let cfg = RopeScalingConfig {
        scaling_type: "llama3".to_string(),
        factor: 8.0,
        original_max_position_embeddings: Some(8192),
        low_freq_factor: Some(1.0),
        high_freq_factor: Some(4.0),
        attention_factor: None,
        beta_fast: None,
        beta_slow: None,
        short_factor: None,
        long_factor: None,
    };

    let rope =
        RoPE::<CpuRuntime>::precompute_freqs(128, 64, 500000.0, Some(&cfg), &device).unwrap();
    assert_eq!(rope.cos_cache().shape(), &[128, 32]);
    // Verify it doesn't panic and produces valid values
    let cos: Vec<f32> = rope.cos_cache().tensor().to_vec();
    for &v in &cos {
        assert!(v.is_finite(), "non-finite cos value: {v}");
        assert!((-1.0..=1.0).contains(&v), "cos out of range: {v}");
    }
}

/// Recover the per-dim frequencies from the caches: at pos=1 the angle equals
/// the frequency, and `atan2` cancels any mscale applied to both cos and sin.
fn freqs_at_pos1(rope: &RoPE<CpuRuntime>, half_dim: usize) -> Vec<f32> {
    let cos: Vec<f32> = rope.cos_cache().tensor().to_vec();
    let sin: Vec<f32> = rope.sin_cache().tensor().to_vec();
    (0..half_dim)
        .map(|i| sin[half_dim + i].atan2(cos[half_dim + i]))
        .collect()
}

fn yarn_cfg() -> RopeScalingConfig {
    RopeScalingConfig {
        scaling_type: "yarn".to_string(),
        factor: 4.0,
        original_max_position_embeddings: Some(2048),
        low_freq_factor: None,
        high_freq_factor: None,
        attention_factor: None,
        beta_fast: Some(32.0),
        beta_slow: Some(1.0),
        short_factor: None,
        long_factor: None,
    }
}

#[test]
fn test_rope_yarn_frequencies() {
    // head_dim=8, base=10000, factor=4, original=2048, beta_fast=32, beta_slow=1.
    //   correction range: low = floor(1.008) = 1, high = ceil(2.513) = 3
    //   ramp over dim/2=4 entries: [0, 0, 0.5, 1] -> extrapolation [1, 1, 0.5, 0]
    //   inv_freq_extrapolation = 10000^(-i/4) = [1, 0.1, 0.01, 0.001]
    let device = CpuDevice::new();
    let cfg = yarn_cfg();
    let rope = RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, Some(&cfg), &device).unwrap();

    let expected = [1.0f32, 0.1, 0.00625, 0.00025];
    let got = freqs_at_pos1(&rope, 4);
    for (i, (&e, &g)) in expected.iter().zip(got.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-6,
            "yarn freq[{i}]: expected {e}, got {g}"
        );
    }
}

#[test]
fn test_rope_yarn_attention_factor_scales_caches() {
    let device = CpuDevice::new();
    let mut cfg = yarn_cfg();
    cfg.attention_factor = Some(1.0);
    let unit = RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, Some(&cfg), &device).unwrap();
    cfg.attention_factor = Some(0.25);
    let scaled = RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, Some(&cfg), &device).unwrap();

    let cos_unit: Vec<f32> = unit.cos_cache().tensor().to_vec();
    let sin_unit: Vec<f32> = unit.sin_cache().tensor().to_vec();
    let cos_scaled: Vec<f32> = scaled.cos_cache().tensor().to_vec();
    let sin_scaled: Vec<f32> = scaled.sin_cache().tensor().to_vec();

    for i in 0..cos_unit.len() {
        assert!(
            (cos_scaled[i] - cos_unit[i] * 0.25).abs() < 1e-6,
            "cos[{i}]: expected {}, got {}",
            cos_unit[i] * 0.25,
            cos_scaled[i]
        );
        assert!(
            (sin_scaled[i] - sin_unit[i] * 0.25).abs() < 1e-6,
            "sin[{i}]: expected {}, got {}",
            sin_unit[i] * 0.25,
            sin_scaled[i]
        );
    }
}

#[test]
fn test_rope_yarn_requires_original_max_position_embeddings() {
    let device = CpuDevice::new();
    let mut cfg = yarn_cfg();
    cfg.original_max_position_embeddings = None;
    let err = RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, Some(&cfg), &device)
        .err()
        .expect("yarn without original_max_position_embeddings must error");
    assert!(
        err.to_string().contains("original_max_position_embeddings"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_rope_dynamic_scaling_rejected() {
    let device = CpuDevice::new();
    let mut cfg = yarn_cfg();
    cfg.scaling_type = "dynamic".to_string();
    let err = RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, Some(&cfg), &device)
        .err()
        .expect("dynamic scaling must error");
    assert!(
        err.to_string().contains("dynamic"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_rope_unknown_scaling_rejected() {
    let device = CpuDevice::new();
    let mut cfg = yarn_cfg();
    cfg.scaling_type = "nope".to_string();
    let err = RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, Some(&cfg), &device)
        .err()
        .expect("unknown scaling type must error");
    let msg = err.to_string();
    assert!(
        msg.contains("nope"),
        "error must name the offending value: {msg}"
    );
    assert!(
        msg.contains("yarn"),
        "error must list supported types: {msg}"
    );
}

#[test]
fn test_rope_linear_llama3_unscaled_by_attention_factor() {
    // Regression guard: mscale applies to the yarn arm only, so linear and
    // llama3 caches keep cos(0)=1 even when attention_factor is present.
    let device = CpuDevice::new();
    for scaling_type in ["linear", "llama3"] {
        let cfg = RopeScalingConfig {
            scaling_type: scaling_type.to_string(),
            factor: 4.0,
            original_max_position_embeddings: Some(2048),
            low_freq_factor: Some(1.0),
            high_freq_factor: Some(4.0),
            attention_factor: Some(0.25),
            beta_fast: Some(32.0),
            beta_slow: Some(1.0),
            short_factor: None,
            long_factor: None,
        };
        let rope =
            RoPE::<CpuRuntime>::precompute_freqs(4, 8, 10000.0, Some(&cfg), &device).unwrap();
        let cos: Vec<f32> = rope.cos_cache().tensor().to_vec();
        for (i, c) in cos.iter().take(4).enumerate() {
            assert!((c - 1.0).abs() < 1e-6, "{scaling_type} cos[0,{i}]={c}");
        }
    }
}

fn longrope_cfg(
    short_factor: Vec<f32>,
    long_factor: Vec<f32>,
    original_max_position_embeddings: Option<usize>,
) -> RopeScalingConfig {
    RopeScalingConfig {
        scaling_type: "longrope".to_string(),
        factor: 1.0,
        original_max_position_embeddings,
        low_freq_factor: None,
        high_freq_factor: None,
        attention_factor: None,
        beta_fast: None,
        beta_slow: None,
        short_factor: Some(short_factor),
        long_factor: Some(long_factor),
    }
}

#[test]
fn test_rope_longrope_short_factor_frequencies() {
    // head_dim=8, half_dim=4, base=10000, original=8, max_seq_len=8 (<=
    // original -> short_factor path, and max_seq_len == original ->
    // attention_scaling == 1).
    //   inv_freq = 10000^(-2i/8) = [1, 0.1, 0.01, 0.001]
    //   short_factor = [2, 4, 5, 8] -> divided = [0.5, 0.025, 0.002, 0.000125]
    let device = CpuDevice::new();
    let cfg = longrope_cfg(vec![2.0, 4.0, 5.0, 8.0], vec![1.0; 4], Some(8));
    let rope = RoPE::<CpuRuntime>::precompute_freqs(8, 8, 10000.0, Some(&cfg), &device).unwrap();

    let expected = [0.5f32, 0.025, 0.002, 0.000125];
    let got = freqs_at_pos1(&rope, 4);
    for (i, (&e, &g)) in expected.iter().zip(got.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-6,
            "longrope short_factor freq[{i}]: expected {e}, got {g}"
        );
    }
}

#[test]
fn test_rope_longrope_long_factor_and_attention_scaling() {
    // max_seq_len=16 > original=8 selects long_factor (short_factor=999 would
    // give wildly different frequencies if wrongly selected, so this also
    // pins the selection, not just the scaling).
    //   inv_freq = 10000^(-2i/8) = [1, 0.1, 0.01, 0.001]
    //   long_factor = [2, 2, 2, 2] -> divided = [0.5, 0.05, 0.005, 0.0005]
    //   attention_scaling = sqrt(1 + ln(16/8) / ln(8)) ~= 1.1547005
    let device = CpuDevice::new();
    let cfg = longrope_cfg(vec![999.0; 4], vec![2.0; 4], Some(8));
    let rope = RoPE::<CpuRuntime>::precompute_freqs(16, 8, 10000.0, Some(&cfg), &device).unwrap();

    let expected = [0.5f32, 0.05, 0.005, 0.0005];
    let got = freqs_at_pos1(&rope, 4);
    for (i, (&e, &g)) in expected.iter().zip(got.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-6,
            "longrope long_factor freq[{i}]: expected {e}, got {g}"
        );
    }

    let cos: Vec<f32> = rope.cos_cache().tensor().to_vec();
    let expected_scaling = 1.154_700_5f32;
    for (i, &c) in cos.iter().take(4).enumerate() {
        assert!(
            (c - expected_scaling).abs() < 1e-5,
            "attention_scaling cos[0,{i}]: expected {expected_scaling}, got {c}"
        );
    }
}

#[test]
fn test_rope_longrope_wrong_length_factor_errors() {
    let device = CpuDevice::new();
    let cfg = longrope_cfg(vec![1.0; 3], vec![1.0; 4], Some(8));
    let err = RoPE::<CpuRuntime>::precompute_freqs(8, 8, 10000.0, Some(&cfg), &device)
        .err()
        .expect("wrong-length short_factor must error");
    let msg = err.to_string();
    assert!(msg.contains("expected 4"), "unexpected error: {msg}");
    assert!(msg.contains("got 3"), "unexpected error: {msg}");
}

#[test]
fn test_rope_longrope_requires_original_max_position_embeddings() {
    let device = CpuDevice::new();
    let cfg = longrope_cfg(vec![1.0; 4], vec![1.0; 4], None);
    let err = RoPE::<CpuRuntime>::precompute_freqs(8, 8, 10000.0, Some(&cfg), &device)
        .err()
        .expect("longrope without original_max_position_embeddings must error");
    assert!(
        err.to_string().contains("original_max_position_embeddings"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_rope_narrow_positions_keeps_matching_rows() {
    let device = CpuDevice::new();
    let rope = RoPE::<CpuRuntime>::precompute_freqs(128, 64, 10000.0, None, &device).unwrap();
    let narrowed = rope.narrow_positions(5).unwrap();
    assert_eq!(narrowed.cos_cache().shape(), &[5, 32]);
    assert_eq!(narrowed.sin_cache().shape(), &[5, 32]);

    let cos_full: Vec<f32> = rope.cos_cache().tensor().to_vec();
    let sin_full: Vec<f32> = rope.sin_cache().tensor().to_vec();
    let cos_narrow: Vec<f32> = narrowed.cos_cache().tensor().to_vec();
    let sin_narrow: Vec<f32> = narrowed.sin_cache().tensor().to_vec();
    assert_eq!(cos_narrow, cos_full[..5 * 32]);
    assert_eq!(sin_narrow, sin_full[..5 * 32]);
}

#[test]
fn test_rope_narrow_positions_rejects_zero() {
    let device = CpuDevice::new();
    let rope = RoPE::<CpuRuntime>::precompute_freqs(128, 64, 10000.0, None, &device).unwrap();
    let err = rope
        .narrow_positions(0)
        .err()
        .expect("num_positions=0 must error");
    assert!(
        err.to_string().contains("num_positions"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_rope_narrow_positions_rejects_over_length() {
    let device = CpuDevice::new();
    let rope = RoPE::<CpuRuntime>::precompute_freqs(128, 64, 10000.0, None, &device).unwrap();
    let err = rope
        .narrow_positions(129)
        .err()
        .expect("num_positions > cache length must error");
    let msg = err.to_string();
    assert!(msg.contains("128"), "unexpected error: {msg}");
    assert!(msg.contains("129"), "unexpected error: {msg}");
}

#[test]
fn test_rope_longrope_requires_short_factor() {
    let device = CpuDevice::new();
    let mut cfg = longrope_cfg(vec![1.0; 4], vec![1.0; 4], Some(8));
    cfg.short_factor = None;
    let err = RoPE::<CpuRuntime>::precompute_freqs(8, 8, 10000.0, Some(&cfg), &device)
        .err()
        .expect("longrope without short_factor must error");
    assert!(
        err.to_string().contains("short_factor"),
        "unexpected error: {err}"
    );
}
