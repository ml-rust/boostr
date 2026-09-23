//! One captured `qwen35` decode graph, replayed for several lanes at once.
//!
//! Run with:
//!   cd boostr && cargo test --release --features cuda --test qwen35_decode_graph_lanes
//!
//! Every test captures ONE graph at a fixed lane count and replays it. Each
//! lane holds a prompt of its own length, so a replay that honoured a single
//! shared length would truncate or over-read some lane and the comparison
//! against that sequence decoded alone would fail.
//!
//! - `lanes_of_different_lengths_match_solo_eager`: three occupied lanes.
//! - `a_free_lane_leaves_the_others_alone`: the middle lane released.
//! - `one_capture_serves_every_replay`: capturing once, replaying many.

#![cfg(feature = "cuda")]

mod common;

use boostr::inference::decode_graph::Qwen35DecodeGraph;
use boostr::inference::{LaneAdmission, LaneTable, LayeredGdnState, LayeredKvCache};
use boostr::model::qwen35::Qwen35Model;
use common::qwen35_cuda::{
    argmax, caches, caches_batched, cuda_available, cuda_setup, eager_decode, max_abs_diff, prefill,
};
use common::qwen35_graph::{GraphBufs, capture};
use common::qwen35_tiny::{MAX_POS, VOCAB, tiny_model};
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;
use std::sync::{Mutex, OnceLock};

/// Two captures must never overlap on the device, so every test in this
/// binary holds this lock for its whole run.
static CUDA_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn cuda_lock() -> std::sync::MutexGuard<'static, ()> {
    CUDA_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

/// Largest logit gap the graph and the eager forward may differ by.
const LOGIT_TOL: f32 = 1e-4;
const STEPS: usize = 4;

/// One sequence prefilled into a cache of its own, ready to take a lane.
struct Solo {
    kv: LayeredKvCache<CudaRuntime>,
    gdn: LayeredGdnState<CudaRuntime>,
    first: i64,
    len: usize,
}

/// `len` distinct token ids, seeded by `seed`.
fn prompt(len: usize, seed: usize) -> Vec<i64> {
    (0..len).map(|i| ((i * 5 + seed) % VOCAB) as i64).collect()
}

/// Prefill every prompt on its own.
fn solos(
    client: &CudaClient,
    device: &CudaDevice,
    model: &Qwen35Model<CudaRuntime>,
    prompts: &[Vec<i64>],
) -> Vec<Solo> {
    prompts
        .iter()
        .map(|tokens| {
            let (mut kv, mut gdn) = caches(device, model, MAX_POS);
            let first = prefill(client, device, model, tokens, &mut kv, &mut gdn);
            Solo {
                kv,
                gdn,
                first,
                len: tokens.len(),
            }
        })
        .collect()
}

/// Install every solo sequence into its own lane, lowest lane first.
fn admit_all(
    client: &CudaClient,
    lanes: &mut LaneTable<CudaRuntime>,
    kv: &mut LayeredKvCache<CudaRuntime>,
    gdn: &mut LayeredGdnState<CudaRuntime>,
    solo: &[Solo],
) {
    for (lane, src) in solo.iter().enumerate() {
        let got = lanes
            .admit(
                client,
                kv,
                gdn,
                LaneAdmission {
                    seq: lane as u64,
                    kv: &src.kv,
                    gdn: &src.gdn,
                    rope_pos: src.len,
                },
            )
            .unwrap();
        assert_eq!(got, lane, "lanes are handed out lowest free first");
    }
}

/// One lane's replayed logits rows and the ids the graph picked from them.
struct Lane {
    rows: Vec<Vec<f32>>,
    ids: Vec<i64>,
}

/// A captured batched decode step and the state it reads.
///
/// `kv` and `gdn` are declared after `decode` so the graph drops before the
/// buffers its kernels address.
struct Batched {
    decode: Qwen35DecodeGraph,
    logits_buf: Tensor<CudaRuntime>,
    kv: LayeredKvCache<CudaRuntime>,
    gdn: LayeredGdnState<CudaRuntime>,
    captures: usize,
}

impl Batched {
    /// One replay: each lane's logits row and the id the graph's own argmax
    /// picked for it, in lane order.
    fn step(&mut self, client: &CudaClient) -> (Vec<Vec<f32>>, Vec<i64>) {
        self.decode.pre_replay_and_launch(client).unwrap();
        let all = self.logits_buf.to_vec::<f32>();
        let rows = (0..self.decode.batch())
            .map(|lane| all[lane * VOCAB..(lane + 1) * VOCAB].to_vec())
            .collect();
        (rows, self.decode.next_token_buf.to_vec::<i64>())
    }

    /// Each lane's rows and ids over `steps` replays, in lane order.
    fn decode_steps(&mut self, client: &CudaClient, steps: usize) -> Vec<Lane> {
        let mut per_lane: Vec<Lane> = (0..self.decode.batch())
            .map(|_| Lane {
                rows: Vec::with_capacity(steps),
                ids: Vec::with_capacity(steps),
            })
            .collect();
        for _ in 0..steps {
            let (rows, ids) = self.step(client);
            for (lane, (row, id)) in rows.into_iter().zip(ids).enumerate() {
                per_lane[lane].rows.push(row);
                per_lane[lane].ids.push(id);
            }
        }
        per_lane
    }
}

/// Capture one decode step for `solo.len()` lanes, then release every lane
/// in `free`.
///
/// The warm-up pass runs uncaptured and writes both caches, so every lane is
/// released and admitted again before the capture records the step.
fn build(
    client: &CudaClient,
    device: &CudaDevice,
    model: &Qwen35Model<CudaRuntime>,
    solo: &[Solo],
    free: &[usize],
) -> Batched {
    let batch = solo.len();
    let (mut kv, mut gdn) = caches_batched(device, model, MAX_POS, batch);
    let mut lanes = LaneTable::<CudaRuntime>::new(batch, model.rope_table_len(), device).unwrap();
    admit_all(client, &mut lanes, &mut kv, &mut gdn, solo);

    let mut bufs = GraphBufs::new(device, batch, model.config().vocab_size);
    bufs.warmup(client, model, &kv, &gdn, &lanes, MAX_POS);

    for lane in 0..batch {
        lanes.release(client, &mut gdn, lane).unwrap();
    }
    admit_all(client, &mut lanes, &mut kv, &mut gdn, solo);
    for lane in free {
        lanes.release(client, &mut gdn, *lane).unwrap();
    }

    let mut captures = 0usize;
    let (graph, _) = capture(
        client,
        device,
        model,
        &kv,
        &gdn,
        &lanes,
        &bufs,
        &mut captures,
    );
    let GraphBufs {
        token_buf,
        next_token_buf,
        logits_buf,
        scalars,
    } = bufs;
    let decode =
        Qwen35DecodeGraph::new(graph, scalars, MAX_POS, lanes, token_buf, next_token_buf).unwrap();
    for (lane, src) in solo.iter().enumerate() {
        let token = if free.contains(&lane) { 0 } else { src.first };
        decode.seed_next_token(client, lane, token).unwrap();
    }
    Batched {
        decode,
        logits_buf,
        kv,
        gdn,
        captures,
    }
}

/// `steps` greedy logits rows for one sequence decoded alone.
fn solo_rows(
    client: &CudaClient,
    device: &CudaDevice,
    model: &Qwen35Model<CudaRuntime>,
    tokens: &[i64],
    steps: usize,
) -> (Vec<Vec<f32>>, Vec<i64>) {
    let (mut kv, mut gdn) = caches(device, model, MAX_POS);
    let first = prefill(client, device, model, tokens, &mut kv, &mut gdn);
    eager_decode(client, device, model, first, steps, &mut kv, &mut gdn)
}

/// Every replayed row and id of `lane` against that sequence decoded alone.
fn assert_lane_matches_solo(lane: usize, got: &Lane, want: &[Vec<f32>], want_ids: &[i64]) {
    assert_eq!(got.rows.len(), want.len(), "lane {lane}: step count");
    assert_eq!(got.ids, want_ids, "lane {lane}: argmax ids");
    for (step, (a, b)) in got.rows.iter().zip(want).enumerate() {
        assert!(
            a.iter().all(|v| v.is_finite()),
            "lane {lane} step {step}: non-finite logits"
        );
        let diff = max_abs_diff(a, b);
        assert!(
            diff < LOGIT_TOL,
            "lane {lane} step {step}: logits diff {diff}"
        );
        assert_eq!(
            argmax(a),
            want_ids[step],
            "lane {lane} step {step}: the row's own greedy pick"
        );
    }
}

/// Three lanes of different prompt lengths in one capture, each matching its
/// own solo decode.
#[test]
fn lanes_of_different_lengths_match_solo_eager() {
    if !cuda_available() {
        println!("skip: no CUDA device");
        return;
    }
    let _guard = cuda_lock();
    let (client, device) = cuda_setup();
    let model = tiny_model::<CudaRuntime>(&device, 0x3535_0902);
    let prompts = vec![prompt(4, 3), prompt(6, 7), prompt(9, 11)];

    let solo = solos(&client, &device, &model, &prompts);
    let mut batched = build(&client, &device, &model, &solo, &[]);
    let rows = batched.decode_steps(&client, STEPS);

    for (lane, tokens) in prompts.iter().enumerate() {
        let (want, want_ids) = solo_rows(&client, &device, &model, tokens, STEPS);
        assert_lane_matches_solo(lane, &rows[lane], &want, &want_ids);
    }
    assert_eq!(batched.kv.num_layers(), model.num_attention_layers());
    assert_eq!(batched.gdn.num_layers(), model.num_gdn_layers());
}

/// A lane left free in the same capture reads no key and changes nothing for
/// the lanes around it.
#[test]
fn a_free_lane_leaves_the_others_alone() {
    if !cuda_available() {
        println!("skip: no CUDA device");
        return;
    }
    let _guard = cuda_lock();
    let (client, device) = cuda_setup();
    let model = tiny_model::<CudaRuntime>(&device, 0x3535_0903);
    let prompts = vec![prompt(5, 2), prompt(8, 5), prompt(3, 9)];
    const FREE: usize = 1;

    let solo = solos(&client, &device, &model, &prompts);
    let mut batched = build(&client, &device, &model, &solo, &[FREE]);
    assert!(batched.decode.lanes.is_free(FREE));
    let rows = batched.decode_steps(&client, STEPS);

    for (lane, tokens) in prompts.iter().enumerate() {
        if lane == FREE {
            assert_eq!(batched.decode.lanes.lane_len(lane), 0, "the free lane held");
            continue;
        }
        let (want, want_ids) = solo_rows(&client, &device, &model, tokens, STEPS);
        assert_lane_matches_solo(lane, &rows[lane], &want, &want_ids);
    }
}

/// Capturing once and replaying many times records exactly one capture, and
/// every lane still tracks its own length.
#[test]
fn one_capture_serves_every_replay() {
    if !cuda_available() {
        println!("skip: no CUDA device");
        return;
    }
    let _guard = cuda_lock();
    let (client, device) = cuda_setup();
    let model = tiny_model::<CudaRuntime>(&device, 0x3535_0904);
    let prompts = vec![prompt(4, 1), prompt(7, 6)];
    const REPLAYS: usize = 6;

    let solo = solos(&client, &device, &model, &prompts);
    let mut batched = build(&client, &device, &model, &solo, &[]);
    let rows = batched.decode_steps(&client, REPLAYS);

    assert_eq!(batched.captures, 1, "one capture served every replay");
    assert_eq!(rows[0].rows.len(), REPLAYS);
    assert_eq!(batched.decode.lanes.lane_len(0), 4 + REPLAYS);
    assert_eq!(batched.decode.lanes.lane_len(1), 7 + REPLAYS);
}
