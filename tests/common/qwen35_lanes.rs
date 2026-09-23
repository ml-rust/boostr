//! Driving a batched `qwen35` decode over lanes.
//!
//! One lane per sequence, each holding its own tokens from cache position
//! 0 with its own end. A sequence is prefilled on its own and admitted into
//! a free lane, so a lane's prompt length never depends on what the other
//! lanes hold.

use boostr::error::Result;
use boostr::inference::kv_cache::layered::LayeredKvCacheConfig;
use boostr::inference::{LaneAdmission, LaneTable, LayeredGdnState, LayeredKvCache};
use boostr::model::qwen35::Qwen35Model;
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

use super::qwen35_tiny::{H_KV, HD, MAX_POS, VOCAB};

/// Empty KV cache and GDN state for `batch` lanes.
pub fn caches(
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

/// `[1, len]` token ids.
pub fn ids(tokens: &[i64], device: &CpuDevice) -> Tensor<CpuRuntime> {
    Tensor::<CpuRuntime>::from_slice(tokens, &[1, tokens.len()], device).expect("ids")
}

/// The greedy pick of row `row` of a `[batch, 1, VOCAB]` logits tensor.
pub fn argmax(logits: &Tensor<CpuRuntime>, row: usize) -> i64 {
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
pub fn prefill(
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

/// A batched decode in flight: the lane table, the containers its lanes
/// live in, and the token each lane decodes from next.
pub struct Lanes {
    pub table: LaneTable<CpuRuntime>,
    pub kv: LayeredKvCache<CpuRuntime>,
    pub gdn: LayeredGdnState<CpuRuntime>,
    pub next: Vec<i64>,
}

impl Lanes {
    /// `batch` free lanes over freshly allocated containers.
    pub fn new(device: &CpuDevice, model: &Qwen35Model<CpuRuntime>, batch: usize) -> Self {
        let (kv, gdn) = caches(device, model, batch);
        Self {
            table: LaneTable::<CpuRuntime>::new(batch, model.rope_table_len(), device)
                .expect("lane table"),
            kv,
            gdn,
            next: vec![0; batch],
        }
    }

    /// Prefill `prompt` on its own and install it into a free lane.
    pub fn admit(
        &mut self,
        client: &CpuClient,
        device: &CpuDevice,
        model: &Qwen35Model<CpuRuntime>,
        seq: u64,
        prompt: &[i64],
    ) -> usize {
        let (kv, gdn, logits) = prefill(client, device, model, prompt).expect("prefill");
        let lane = self
            .table
            .admit(
                client,
                &mut self.kv,
                &mut self.gdn,
                LaneAdmission {
                    seq,
                    kv: &kv,
                    gdn: &gdn,
                    rope_pos: prompt.len(),
                },
            )
            .expect("admit");
        self.next[lane] = argmax(&logits, prompt.len() - 1);
        lane
    }

    /// Clear a lane's state and hand it back.
    pub fn release(&mut self, client: &CpuClient, lane: usize) {
        self.table
            .release(client, &mut self.gdn, lane)
            .expect("release");
        self.next[lane] = 0;
    }

    /// One decode step for every lane: the per-lane logits rows, with each
    /// lane's next token and the table already advanced.
    pub fn step(
        &mut self,
        client: &CpuClient,
        device: &CpuDevice,
        model: &Qwen35Model<CpuRuntime>,
    ) -> Vec<Vec<f32>> {
        let batch = self.table.batch();
        let token_ids =
            Tensor::<CpuRuntime>::from_slice(&self.next, &[batch, 1], device).expect("token ids");
        let logits = model
            .forward_qwen35_batched_decode(
                client,
                &token_ids,
                self.table.positions(),
                &mut self.kv,
                &mut self.gdn,
                &self.table.step(),
            )
            .expect("batched decode step");
        let all = logits.to_vec::<f32>();
        let mut rows = Vec::with_capacity(batch);
        for lane in 0..batch {
            rows.push(all[lane * VOCAB..(lane + 1) * VOCAB].to_vec());
            self.next[lane] = argmax(&logits, lane);
        }
        self.table.advance(client).expect("advance");
        rows
    }
}

/// Greedy decode of one sequence alone, `steps` logits rows.
pub fn decode_solo(
    client: &CpuClient,
    device: &CpuDevice,
    model: &Qwen35Model<CpuRuntime>,
    prompt: &[i64],
    steps: usize,
) -> Vec<Vec<f32>> {
    let mut lanes = Lanes::new(device, model, 1);
    lanes.admit(client, device, model, 0, prompt);
    (0..steps)
        .map(|_| lanes.step(client, device, model).remove(0))
        .collect()
}

/// Bit equality of one lane's logits row against the solo run's.
pub fn assert_bits(lane: usize, step: usize, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "lane {lane} step {step}: length");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "lane {lane} step {step}: logit {i} batched {a} solo {b}"
        );
    }
}
