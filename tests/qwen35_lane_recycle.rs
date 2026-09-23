//! A lane freed mid-flight takes any next sequence, whatever the lanes
//! around it are doing.
//!
//! Every lane holds its own tokens from cache position 0 with its own end,
//! so the sequence that recycles a lane may be longer than the number of
//! steps the other lanes have already decoded. The test holds both lanes to
//! bit equality with their solo runs across the recycle.
//!
//! Run with:
//!   cd boostr && cargo test --test qwen35_lane_recycle

mod common;

use common::cpu_setup;
use common::qwen35_lanes::{Lanes, assert_bits, decode_solo};
use common::qwen35_tiny::tiny_model;
use numr::runtime::cpu::CpuRuntime;

/// Steps before the recycle, and after it.
const BEFORE: usize = 2;
const AFTER: usize = 3;

/// `A` and `B` decode together, `B`'s lane is freed, and `C` takes it with
/// a prompt longer than the steps already decoded.
#[test]
fn a_recycled_lane_takes_a_longer_sequence() {
    let (client, device) = cpu_setup();
    let model = tiny_model::<CpuRuntime>(&device, 0x3535_0200);
    let a: &[i64] = &[1, 5, 9, 2, 7];
    let b: &[i64] = &[3, 11];
    let c: &[i64] = &[8, 4, 6, 15, 10, 13];
    assert!(
        c.len() > BEFORE,
        "C's prompt must outlast the steps B already took, so the recycled lane starts longer \
         than its neighbor"
    );

    let mut lanes = Lanes::new(&device, &model, 2);
    assert_eq!(lanes.admit(&client, &device, &model, 1, a), 0);
    assert_eq!(lanes.admit(&client, &device, &model, 2, b), 1);

    let mut lane_a = Vec::new();
    let mut lane_b = Vec::new();
    for _ in 0..BEFORE {
        let rows = lanes.step(&client, &device, &model);
        lane_a.push(rows[0].clone());
        lane_b.push(rows[1].clone());
    }

    lanes.release(&client, 1);
    assert!(lanes.table.is_free(1));
    assert_eq!(lanes.table.occupant(0), Some(1));

    assert_eq!(lanes.admit(&client, &device, &model, 3, c), 1);
    assert_eq!(lanes.table.lane_len(1), c.len());
    assert_eq!(lanes.table.lane_len(0), a.len() + BEFORE);

    let mut lane_c = Vec::new();
    for _ in 0..AFTER {
        let rows = lanes.step(&client, &device, &model);
        lane_a.push(rows[0].clone());
        lane_c.push(rows[1].clone());
    }

    // Lane 0 ran through the release and the admit beside it.
    let solo_a = decode_solo(&client, &device, &model, a, BEFORE + AFTER);
    for step in 0..BEFORE + AFTER {
        assert_bits(0, step, &lane_a[step], &solo_a[step]);
    }

    // Lane 1 before the recycle, and after it.
    let solo_b = decode_solo(&client, &device, &model, b, BEFORE);
    for step in 0..BEFORE {
        assert_bits(1, step, &lane_b[step], &solo_b[step]);
    }
    let solo_c = decode_solo(&client, &device, &model, c, AFTER);
    for step in 0..AFTER {
        assert_bits(1, step, &lane_c[step], &solo_c[step]);
    }
}

/// The same sequence admitted into a lane a previous occupant left keys in
/// decodes exactly as it does into a lane that was never used: the stale
/// keys sit past the new occupant's end.
#[test]
fn stale_keys_past_the_new_end_are_never_read() {
    let (client, device) = cpu_setup();
    let model = tiny_model::<CpuRuntime>(&device, 0x3535_0201);
    let long: &[i64] = &[2, 14, 6, 1, 9, 12, 5];
    let short: &[i64] = &[7, 3];

    let mut lanes = Lanes::new(&device, &model, 1);
    lanes.admit(&client, &device, &model, 1, long);
    for _ in 0..BEFORE {
        lanes.step(&client, &device, &model);
    }
    lanes.release(&client, 0);
    assert_eq!(lanes.admit(&client, &device, &model, 2, short), 0);

    let solo = decode_solo(&client, &device, &model, short, AFTER);
    for (step, want) in solo.iter().enumerate() {
        let rows = lanes.step(&client, &device, &model);
        assert_bits(0, step, &rows[0], want);
    }
}
