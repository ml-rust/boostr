//! A graph-arm build recovers from a first capture arena that is too small.
//!
//! `build_lane_driver_with_first_arena_for_test` forces the first-capture
//! arena size instead of the free-memory probe, so the test can hand it an
//! arena of 1 byte: too small for the first intermediate allocation the
//! captured decode step makes. The build must still open a working session,
//! and it must have taken more than one capture attempt to get there —
//! without the retry in `build_graph`, this first capture would fail outright
//! and `LaneDecodeSession::open`/`build_lane_driver` would return an error
//! instead of a driver.
//!
//! Run with:
//!   cd boostr && cargo test --release --features cuda --test qwen35_arena_retry

#![cfg(feature = "cuda")]

mod common;

use boostr::inference::lane_decode::{
    LaneDecodeMode, LaneDriverSpec, build_lane_driver_with_first_arena_for_test,
};
use common::qwen35_cuda::{caches_batched, cuda_available, cuda_setup};
use common::qwen35_tiny::{MAX_POS, tiny_model};
use numr::runtime::cuda::CudaRuntime;

/// Too small for the first arena-internal intermediate the captured decode
/// step allocates, so the first attempt always exhausts the arena.
const TOO_SMALL_FIRST_ARENA_BYTES: usize = 1;

/// Lanes the forced-tiny-arena build captures for.
const LANES: usize = 2;

#[test]
fn a_too_small_first_arena_is_retried_and_recovers() {
    if !cuda_available() {
        return;
    }
    let (_client, device) = cuda_setup();
    let model = tiny_model::<CudaRuntime>(&device, 0x4152_454e);
    let (kv, gdn) = caches_batched(&device, &model, MAX_POS, LANES);

    let spec = LaneDriverSpec {
        model: &model,
        device: &device,
        kv,
        gdn,
        lanes: LANES,
        capacity: MAX_POS,
        mode: LaneDecodeMode::Graph,
    };

    let build = build_lane_driver_with_first_arena_for_test(spec, TOO_SMALL_FIRST_ARENA_BYTES)
        .expect(
            "a first capture arena too small to hold the step must still open a session via \
             retry",
        );

    assert_eq!(build.driver.batch(), LANES);
    assert!(
        build.graph_captures > 1,
        "a 1-byte first arena must force at least one retry beyond the first attempt, got {} \
         capture(s)",
        build.graph_captures
    );
}
