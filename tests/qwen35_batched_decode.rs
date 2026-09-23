//! A lane of a batched `qwen35` decode must produce the bits its sequence
//! produces alone.
//!
//! Sequences of different lengths share one cache: each lane holds its own
//! tokens from position 0, `kv_start` names its first position and `kv_end`
//! its last, and each lane's step is written at that lane's own next
//! position. The test prefills each sequence on its own, admits it into a
//! lane, then decodes greedily and holds every step's logits to bit
//! equality with the same sequence decoded alone.
//!
//! Run with:
//!   cd boostr && cargo test --test qwen35_batched_decode

mod common;

use common::cpu_setup;
use common::qwen35_lanes::{Lanes, assert_bits, decode_solo};
use common::qwen35_tiny::tiny_model;
use numr::runtime::cpu::CpuRuntime;

const STEPS: usize = 4;

/// Greedy decode of several sequences at once, one lane each.
fn decode_batched(
    client: &numr::runtime::cpu::CpuClient,
    device: &numr::runtime::cpu::CpuDevice,
    model: &boostr::model::qwen35::Qwen35Model<CpuRuntime>,
    prompts: &[&[i64]],
) -> Vec<Vec<Vec<f32>>> {
    let batch = prompts.len();
    let mut lanes = Lanes::new(device, model, batch);
    for (idx, prompt) in prompts.iter().enumerate() {
        let lane = lanes.admit(client, device, model, idx as u64, prompt);
        assert_eq!(lane, idx, "lanes are handed out in order");
    }

    let mut steps = vec![Vec::new(); batch];
    for _ in 0..STEPS {
        for (lane, row) in lanes.step(client, device, model).into_iter().enumerate() {
            steps[lane].push(row);
        }
    }
    steps
}

/// Prompts of different lengths, so every lane ends at a different cache
/// position and each step writes to a different slot.
#[test]
fn lanes_match_the_same_sequence_alone() {
    let (client, device) = cpu_setup();
    let model = tiny_model::<CpuRuntime>(&device, 0x3535_0100);
    let prompts: [&[i64]; 3] = [&[1, 5, 9, 2, 7], &[3, 11], &[8, 4, 6, 15]];

    let batched = decode_batched(&client, &device, &model, &prompts);
    for (lane, prompt) in prompts.iter().enumerate() {
        let solo = decode_solo(&client, &device, &model, prompt, STEPS);
        for step in 0..STEPS {
            assert_bits(lane, step, &batched[lane][step], &solo[step]);
        }
    }
}

/// Equal-length prompts: every lane ends at the same position, so this
/// isolates the batch dimension itself.
#[test]
fn equal_length_lanes_match_solo() {
    let (client, device) = cpu_setup();
    let model = tiny_model::<CpuRuntime>(&device, 0x3535_0101);
    let prompts: [&[i64]; 2] = [&[2, 3, 5], &[13, 1, 4]];

    let batched = decode_batched(&client, &device, &model, &prompts);
    for (lane, prompt) in prompts.iter().enumerate() {
        let solo = decode_solo(&client, &device, &model, prompt, STEPS);
        for step in 0..STEPS {
            assert_bits(lane, step, &batched[lane][step], &solo[step]);
        }
    }
}

/// One lane is the whole batch: the batched entry at a single lane is the
/// solo path, so this pins the entry itself before the batch cases.
#[test]
fn single_lane_batch_matches_solo() {
    let (client, device) = cpu_setup();
    let model = tiny_model::<CpuRuntime>(&device, 0x3535_0102);
    let prompt: &[i64] = &[6, 2, 14, 9];

    let batched = decode_batched(&client, &device, &model, &[prompt]);
    let solo = decode_solo(&client, &device, &model, prompt, STEPS);
    for step in 0..STEPS {
        assert_bits(0, step, &batched[0][step], &solo[step]);
    }
}

/// A lane left free while another decodes changes nothing: the occupied
/// lane's logits are still its solo run's.
#[test]
fn a_free_lane_does_not_disturb_an_occupied_one() {
    let (client, device) = cpu_setup();
    let model = tiny_model::<CpuRuntime>(&device, 0x3535_0103);
    let prompt: &[i64] = &[4, 12, 7];

    let mut lanes = Lanes::new(&device, &model, 2);
    assert_eq!(lanes.admit(&client, &device, &model, 1, prompt), 0);
    assert!(lanes.table.is_free(1));

    let solo = decode_solo(&client, &device, &model, prompt, STEPS);
    for (step, want) in solo.iter().enumerate() {
        let rows = lanes.step(&client, &device, &model);
        assert_bits(0, step, &rows[0], want);
    }
}
