//! Unit tests for the no-reference (zero-shot) prefill path.
//!
//! The with-reference path is exercised end to end by
//! `examples/voxcpm/eval_common.rs`'s `build_prefill_and_target` and by
//! `SequenceLayout`'s own tests in `sequence.rs`. This file pins the
//! `ref_feat = None` branch specifically, since `prefill_capturing(..., None,
//! ...)` is the exact call the training loop makes for a manifest row with no
//! `ref_wav` (see `eval_common.rs`'s doc comment on `build_prefill_and_target`).

use crate::model::audio::voxcpm::model::config::AUDIO_START_ID;
use crate::model::audio::voxcpm::model::generate::tests::support::{fixture, model};
use crate::test_utils::cpu_setup;

/// `prefill_capturing(client, None, ...)` must succeed, land `position`
/// exactly at `text_token_ids.len()` (no reference prefix contributes rows),
/// and always populate `intermediates` — the training loop's
/// `cfm_loss`/`train_losses_with_noise` require `Some` there.
#[test]
fn no_reference_prefill_capturing_succeeds() {
    let (client, device) = cpu_setup();
    let m = model(fixture(false, &device), &device);

    let text_token_ids = [11u32, 22, AUDIO_START_ID];
    let prefill = m
        .prefill_capturing(&client, None, &text_token_ids, text_token_ids.len())
        .expect("no-reference prefill_capturing");

    assert_eq!(
        prefill.position,
        text_token_ids.len(),
        "S == text_token_ids.len() with no reference prefix"
    );
    assert!(
        prefill.intermediates.is_some(),
        "prefill_capturing must always populate intermediates, with or without a reference"
    );
}

/// The plain (non-capturing) path must agree on `position` and leave
/// `intermediates` empty, same as the with-reference path already does.
#[test]
fn no_reference_prefill_leaves_intermediates_empty() {
    let (client, device) = cpu_setup();
    let m = model(fixture(false, &device), &device);

    let text_token_ids = [7u32, AUDIO_START_ID];
    let prefill = m
        .prefill(&client, None, &text_token_ids, text_token_ids.len())
        .expect("no-reference prefill");

    assert_eq!(prefill.position, text_token_ids.len());
    assert!(prefill.intermediates.is_none());
}
