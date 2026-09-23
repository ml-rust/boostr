//! A lane of a batched `qwen35` decode must produce the bits its sequence
//! produces alone.
//!
//! Sequences of different lengths share one cache: each lane is left-padded
//! so every sequence ends at the shared cursor, and `kv_start` names the
//! lane's own first position. The test prefills each sequence on its own,
//! installs it into a lane, then decodes greedily and holds every step's
//! logits to bit equality with the same sequence decoded alone through the
//! same entry point.
//!
//! Run with:
//!   cd boostr && cargo test --test qwen35_batched_decode

mod common;

use boostr::error::Result;
use boostr::inference::kv_cache::layered::LayeredKvCacheConfig;
use boostr::inference::{LayeredGdnState, LayeredKvCache};
use boostr::model::qwen35::Qwen35Model;
use common::cpu_setup;
use common::qwen35_tiny::{H_KV, HD, MAX_POS, VOCAB, tiny_model};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

const STEPS: usize = 4;

fn caches(
    device: &CpuDevice,
    model: &Qwen35Model<CpuRuntime>,
    batch: usize,
) -> (LayeredKvCache<CpuRuntime>, LayeredGdnState<CpuRuntime>) {
    let kv_config = LayeredKvCacheConfig {
        batch_size: batch,
        num_kv_heads: H_KV,
        initial_capacity: MAX_POS,
        max_seq_len: MAX_POS,
        head_dim: HD,
        dtype: DType::F32,
    };
    let kv = LayeredKvCache::<CpuRuntime>::new(model.num_attention_layers(), &kv_config, device)
        .expect("kv cache");
    let gdn = LayeredGdnState::<CpuRuntime>::zeros(
        model.num_gdn_layers(),
        model.gdn_config(),
        batch,
        DType::F32,
        device,
    )
    .expect("gdn state");
    (kv, gdn)
}

fn ids(tokens: &[i64], device: &CpuDevice) -> Tensor<CpuRuntime> {
    Tensor::<CpuRuntime>::from_slice(tokens, &[1, tokens.len()], device).expect("ids")
}

/// The greedy pick of a `[1, 1, VOCAB]` logits row.
fn argmax(logits: &Tensor<CpuRuntime>, row: usize) -> i64 {
    let all = logits.to_vec::<f32>();
    let window = &all[row * VOCAB..(row + 1) * VOCAB];
    let mut best = 0usize;
    for (i, v) in window.iter().enumerate() {
        if *v > window[best] {
            best = i;
        }
    }
    best as i64
}

/// Prefill `prompt` into a cache of its own.
fn prefill(
    client: &CpuClient,
    device: &CpuDevice,
    model: &Qwen35Model<CpuRuntime>,
    prompt: &[i64],
) -> Result<(
    LayeredKvCache<CpuRuntime>,
    LayeredGdnState<CpuRuntime>,
    Tensor<CpuRuntime>,
)> {
    let (mut kv, mut gdn) = caches(device, model, 1);
    let logits = model.forward_qwen35(client, &ids(prompt, device), &mut kv, &mut gdn, 0)?;
    Ok((kv, gdn, logits))
}

/// Greedy decode of one sequence through the batched entry at one lane.
fn decode_solo(
    client: &CpuClient,
    device: &CpuDevice,
    model: &Qwen35Model<CpuRuntime>,
    prompt: &[i64],
) -> Vec<Vec<f32>> {
    let (mut kv, mut gdn, logits) = prefill(client, device, model, prompt).expect("prefill");
    let mut next = argmax(&logits, prompt.len() - 1);
    let kv_start = Tensor::<CpuRuntime>::from_slice(&[0i32], &[1], device).expect("kv_start");
    let mut steps = Vec::new();
    for rope_pos in (prompt.len()..).take(STEPS) {
        let positions = model
            .text_positions_batched(&[rope_pos], device)
            .expect("positions");
        let step = model
            .forward_qwen35_batched_decode(
                client,
                &ids(&[next], device),
                &positions,
                &mut kv,
                &mut gdn,
                &kv_start,
            )
            .expect("solo decode step");
        next = argmax(&step, 0);
        steps.push(step.to_vec::<f32>());
    }
    steps
}

/// Greedy decode of several sequences at once, one lane each.
fn decode_batched(
    client: &CpuClient,
    device: &CpuDevice,
    model: &Qwen35Model<CpuRuntime>,
    prompts: &[&[i64]],
) -> Vec<Vec<Vec<f32>>> {
    let batch = prompts.len();
    let end = prompts.iter().map(|p| p.len()).max().expect("prompts");
    let (mut kv, mut gdn) = caches(device, model, batch);

    let mut next = Vec::with_capacity(batch);
    let mut rope_pos = Vec::with_capacity(batch);
    let mut starts = Vec::with_capacity(batch);
    for (lane, prompt) in prompts.iter().enumerate() {
        let (lane_kv, lane_gdn, logits) = prefill(client, device, model, prompt).expect("prefill");
        kv.install_lane(&lane_kv, lane, end, client)
            .expect("kv lane");
        gdn.install_lane(client, lane, &lane_gdn).expect("gdn lane");
        next.push(argmax(&logits, prompt.len() - 1));
        rope_pos.push(prompt.len());
        starts.push((end - prompt.len()) as i32);
    }
    let kv_start = Tensor::<CpuRuntime>::from_slice(&starts, &[batch], device).expect("kv_start");

    let mut steps = vec![Vec::new(); batch];
    for _ in 0..STEPS {
        let positions = model
            .text_positions_batched(&rope_pos, device)
            .expect("positions");
        let token_ids =
            Tensor::<CpuRuntime>::from_slice(&next, &[batch, 1], device).expect("token ids");
        let step = model
            .forward_qwen35_batched_decode(
                client, &token_ids, &positions, &mut kv, &mut gdn, &kv_start,
            )
            .expect("batched decode step");
        let all = step.to_vec::<f32>();
        for lane in 0..batch {
            steps[lane].push(all[lane * VOCAB..(lane + 1) * VOCAB].to_vec());
            next[lane] = argmax(&step, lane);
            rope_pos[lane] += 1;
        }
    }
    steps
}

fn assert_bits(lane: usize, step: usize, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "lane {lane} step {step}: length");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "lane {lane} step {step}: logit {i} batched {a} solo {b}"
        );
    }
}

/// Prompts of different lengths, so every lane but the longest carries left
/// padding that `kv_start` has to mask off.
#[test]
fn lanes_match_the_same_sequence_alone() {
    let (client, device) = cpu_setup();
    let model = tiny_model::<CpuRuntime>(&device, 0x3535_0100);
    let prompts: [&[i64]; 3] = [&[1, 5, 9, 2, 7], &[3, 11], &[8, 4, 6, 15]];

    let batched = decode_batched(&client, &device, &model, &prompts);
    for (lane, prompt) in prompts.iter().enumerate() {
        let solo = decode_solo(&client, &device, &model, prompt);
        for step in 0..STEPS {
            assert_bits(lane, step, &batched[lane][step], &solo[step]);
        }
    }
}

/// Equal-length prompts: no lane carries padding, so this isolates the
/// batch dimension itself.
#[test]
fn equal_length_lanes_match_solo() {
    let (client, device) = cpu_setup();
    let model = tiny_model::<CpuRuntime>(&device, 0x3535_0101);
    let prompts: [&[i64]; 2] = [&[2, 3, 5], &[13, 1, 4]];

    let batched = decode_batched(&client, &device, &model, &prompts);
    for (lane, prompt) in prompts.iter().enumerate() {
        let solo = decode_solo(&client, &device, &model, prompt);
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
    let solo = decode_solo(&client, &device, &model, prompt);
    for step in 0..STEPS {
        assert_bits(0, step, &batched[0][step], &solo[step]);
    }
}
