//! Tests for [`super`] that need no checkpoint: the padding arithmetic (the
//! regression guard for the always-fires rule), the prior-projection shape
//! validation, and the min-length branch truncation.

use super::*;
use crate::test_utils::cpu_setup;
use numr::runtime::cpu::CpuRuntime;

fn var(data: &[f32], shape: &[usize], device: &<CpuRuntime as Runtime>::Device) -> Var<CpuRuntime> {
    Var::new(
        Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap(),
        false,
    )
}

#[test]
fn alignment_is_the_acoustic_stride() {
    assert_eq!(encode_alignment(), 320);
}

/// THE regression guard: an exact multiple of 320 still gets a FULL 320
/// samples of padding. `pad = 320 - (T % 320)` is unconditional upstream.
#[test]
fn exact_multiple_still_pads_a_full_stride() {
    assert_eq!(encode_padding(8000), 320);
    assert_eq!(8000 + encode_padding(8000), 8320);
    assert_eq!(encode_padding(8320), 320);
    assert_eq!(encode_padding(320), 320);
}

#[test]
fn partial_frame_pads_up_to_the_next_multiple() {
    assert_eq!(encode_padding(1), 319);
    assert_eq!(encode_padding(100), 220);
    assert_eq!(encode_padding(8321), 319);
    for len in [1usize, 100, 321, 8000, 8321, 16_000] {
        assert_eq!((len + encode_padding(len)) % encode_alignment(), 0);
    }
}

#[test]
fn fc_prior_must_be_2048_square_with_bias() {
    let (_client, device) = cpu_setup();

    let good = Linear::<CpuRuntime>::new(
        Tensor::from_slice(
            &vec![0.0f32; PRIOR_DIM * PRIOR_DIM],
            &[PRIOR_DIM, PRIOR_DIM],
            &device,
        )
        .unwrap(),
        Some(Tensor::from_slice(&vec![0.0f32; PRIOR_DIM], &[PRIOR_DIM], &device).unwrap()),
        false,
    );
    assert!(check_fc_prior(&good).is_ok());

    let no_bias = Linear::<CpuRuntime>::new(
        Tensor::from_slice(
            &vec![0.0f32; PRIOR_DIM * PRIOR_DIM],
            &[PRIOR_DIM, PRIOR_DIM],
            &device,
        )
        .unwrap(),
        None,
        false,
    );
    assert!(check_fc_prior(&no_bias).is_err());

    let wrong_shape = Linear::<CpuRuntime>::new(
        Tensor::from_slice(&[0.0f32; 4 * 8], &[4, 8], &device).unwrap(),
        Some(Tensor::from_slice(&[0.0f32; 4], &[4], &device).unwrap()),
        false,
    );
    assert!(check_fc_prior(&wrong_shape).is_err());
}

#[test]
fn time_len_rejects_non_rank_3() {
    let (_client, device) = cpu_setup();
    let x = var(&[0.0; 6], &[2, 3], &device);
    assert!(time_len(&x, "x").is_err());
    assert!(to_time_last(&x).is_err());
}

/// Both branches are cut to `min(Ts, Ta)`, keeping the EARLIEST frames.
#[test]
fn branches_truncate_to_the_shorter_one() {
    let (_client, device) = cpu_setup();

    // [1, 2, 3] semantic (Ts = 3) and [1, 2, 5] acoustic (Ta = 5).
    let semantic = var(
        &(0..6).map(|i| i as f32).collect::<Vec<_>>(),
        &[1, 2, 3],
        &device,
    );
    let acoustic = var(
        &(0..10).map(|i| 100.0 + i as f32).collect::<Vec<_>>(),
        &[1, 2, 5],
        &device,
    );

    let min_len = min_time(&semantic, &acoustic).expect("min_time");
    assert_eq!(min_len, 3);

    let sem_cut = narrow_time(&semantic, min_len).expect("narrow semantic");
    let aco_cut = narrow_time(&acoustic, min_len).expect("narrow acoustic");
    assert_eq!(sem_cut.shape(), &[1, 2, 3]);
    assert_eq!(aco_cut.shape(), &[1, 2, 3]);

    // Earliest frames kept, tail dropped: rows are 100..103 and 105..108.
    let values = aco_cut
        .tensor()
        .contiguous()
        .expect("contiguous")
        .to_vec::<f32>();
    assert_eq!(values, vec![100.0, 101.0, 102.0, 105.0, 106.0, 107.0]);
}

/// The channel-axis join is SEMANTIC FIRST: `[0, C)` semantic, `[C, 2C)`
/// acoustic. The reverse order is shape-identical and silently wrong.
#[test]
fn concat_puts_semantic_in_the_low_channels() {
    let (client, device) = cpu_setup();

    let semantic = var(&[1.0, 2.0], &[1, 1, 2], &device);
    let acoustic = var(&[3.0, 4.0], &[1, 1, 2], &device);
    let joined = var_cat(&[&semantic, &acoustic], 1, &client).expect("cat");

    assert_eq!(joined.shape(), &[1, 2, 2]);
    assert_eq!(
        joined
            .tensor()
            .contiguous()
            .expect("contiguous")
            .to_vec::<f32>(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn to_time_last_swaps_the_trailing_axes() {
    let (_client, device) = cpu_setup();
    let x = var(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], &device);
    let y = to_time_last(&x).expect("permute");
    assert_eq!(y.shape(), &[1, 3, 2]);
    assert_eq!(
        y.tensor().contiguous().expect("contiguous").to_vec::<f32>(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

/// The refusal is the entire reason [`MAX_ENCODE_SAMPLES`] exists, so it is
/// tested directly rather than through a model that needs a checkpoint.
/// Without the guard, an over-long clip dies as an allocation failure inside a
/// matmul, naming nothing the caller can act on.
#[test]
fn encode_len_guard_refuses_an_over_long_clip() {
    let over = MAX_ENCODE_SAMPLES + 1;
    let Err(err) = check_encode_len(over, MAX_ENCODE_SAMPLES) else {
        panic!("a clip one sample over the limit must be refused");
    };
    let msg = err.to_string();
    assert!(msg.contains(&over.to_string()), "{msg}");
    assert!(msg.contains("utterance"), "{msg}");
    // The corpus case: 26.8 minutes is ~80k frames of quadratic attention.
    assert!(check_encode_len(26 * 60 * SAMPLE_RATE, MAX_ENCODE_SAMPLES).is_err());
}

/// Exactly at the limit must pass — an off-by-one here would silently reject
/// the longest legitimate utterance.
#[test]
fn encode_len_guard_accepts_exactly_the_limit() {
    assert!(check_encode_len(MAX_ENCODE_SAMPLES, MAX_ENCODE_SAMPLES).is_ok());
    assert!(check_encode_len(1, 1).is_ok());
    assert!(check_encode_len(2, 1).is_err());
}

/// Empty input is refused separately from the length limit, so a caller that
/// hands over a zero-length decode result gets a distinct message.
#[test]
fn encode_len_guard_refuses_empty_input() {
    let Err(err) = check_encode_len(0, MAX_ENCODE_SAMPLES) else {
        panic!("an empty waveform must be refused");
    };
    assert!(err.to_string().contains("non-empty"), "{err}");
}
