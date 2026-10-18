//! Speculative greedy generation on a tiny CPU Llama, against plain greedy.
//!
//! The target adapter feeds tokens through `forward_with_kv_cache` at the
//! cache end and takes numr's argmax over the logits on the device. Rollback
//! goes through `LayeredKvCache::truncate`. Every speculative run must emit
//! exactly the tokens of one-token-per-forward greedy decoding.
//!
//! A one-token and a multi-token forward can round differently. A prompt
//! whose plain run has a top-2 logit gap below `NEAR_TIE` is skipped with a
//! printed note, so a rounding flip never counts as a generator error.

mod common;

use boostr::error::Result;
use boostr::inference::LayeredKvCache;
use boostr::inference::speculative::{
    Drafter, FinishReason, PromptLookup, SpeculativeConfig, SpeculativeGenerator, SpeculativeModel,
    SpeculativeStats,
};
use boostr::model::LoadedModel;
use common::llama_tiny::{HEAD_DIM, KV_HEADS, LAYERS, VOCAB, tiny_llama_on};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

const SEED: u64 = 0x5bec_11a3;
const INITIAL_CAPACITY: usize = 4;
const MAX_SEQ_LEN: usize = 96;
const MAX_TOKENS: usize = 24;
const NEAR_TIE: f32 = 1e-4;

/// The tiny Llama behind the speculative model contract.
struct TinyTarget {
    model: LoadedModel<CpuRuntime>,
    kv: LayeredKvCache<CpuRuntime>,
    device: CpuDevice,
    /// Top-2 logit gap of every fed row while set. Plain reference runs only.
    gaps: Option<Vec<f32>>,
}

impl TinyTarget {
    fn new() -> Self {
        let device = CpuDevice::new();
        let model = tiny_llama_on::<CpuRuntime>(&device, SEED, DType::F32);
        let kv = LayeredKvCache::new_positional(
            LAYERS,
            1,
            KV_HEADS,
            INITIAL_CAPACITY,
            MAX_SEQ_LEN,
            HEAD_DIM,
            DType::F32,
            &device,
        )
        .expect("cache");
        Self {
            model,
            kv,
            device,
            gaps: None,
        }
    }
}

/// Largest logit minus the second largest, per row of `VOCAB` logits.
fn top2_gaps(logits: &Tensor<CpuRuntime>) -> Vec<f32> {
    let all = logits.contiguous().expect("contiguous").to_vec::<f32>();
    all.chunks(VOCAB)
        .map(|row| {
            let (mut first, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
            for &v in row {
                if v > first {
                    second = first;
                    first = v;
                } else if v > second {
                    second = v;
                }
            }
            first - second
        })
        .collect()
}

impl SpeculativeModel for TinyTarget {
    fn vocab_size(&self) -> usize {
        VOCAB
    }

    fn len(&self) -> usize {
        self.kv.seq_len()
    }

    fn truncate(&mut self, len: usize) -> Result<()> {
        self.kv.truncate(len)
    }

    fn extend_greedy(&mut self, tokens: &[u32]) -> Result<Vec<u32>> {
        let ids: Vec<i64> = tokens.iter().map(|&t| i64::from(t)).collect();
        let input = Tensor::<CpuRuntime>::from_slice(&ids, &[1, ids.len()], &self.device)?;
        let position = self.kv.seq_len();
        let logits = self
            .model
            .forward_with_kv_cache(&input, &mut self.kv, position)?;
        if let Some(gaps) = &mut self.gaps {
            gaps.extend(top2_gaps(&logits));
        }
        // [1, M, VOCAB] -> [1, M] ids on the device; M ids cross to the host.
        let next = logits.argmax(2, false)?;
        Ok(next
            .to_vec::<i64>()
            .into_iter()
            .map(|id| id as u32)
            .collect())
    }
}

/// Plain greedy decoding: one forward per token, EOS before the length limit.
/// Returns the tokens, the finish reason and the smallest top-2 logit gap.
fn plain(
    target: &mut TinyTarget,
    prompt: &[u32],
    config: &SpeculativeConfig,
) -> (Vec<u32>, FinishReason, f32) {
    target.gaps = None;
    target.truncate(0).expect("reset");
    let (head, last) = prompt.split_at(prompt.len() - 1);
    if !head.is_empty() {
        target.extend_greedy(head).expect("prefill");
    }
    target.gaps = Some(Vec::new());
    let mut x = last[0];
    let mut out = Vec::new();
    let finish = loop {
        let ids = target.extend_greedy(&[x]).expect("decode");
        assert_eq!(ids.len(), 1);
        x = ids[0];
        out.push(x);
        if config.is_eos(x) {
            break FinishReason::Eos;
        }
        if out.len() >= config.max_tokens {
            break FinishReason::Length;
        }
    };
    let gaps = target.gaps.take().expect("gaps recorded");
    let min_gap = gaps.into_iter().fold(f32::INFINITY, f32::min);
    (out, finish, min_gap)
}

/// Proposes the precomputed plain continuation, with draft `corrupt_at` of
/// every proposal changed when set.
struct OracleDraft {
    prompt_len: usize,
    continuation: Vec<u32>,
    corrupt_at: Option<usize>,
}

impl Drafter for OracleDraft {
    fn propose(&mut self, context: &[u32], k: usize) -> Vec<u32> {
        let len = self.continuation.len();
        let pos = (context.len() - self.prompt_len).min(len);
        let mut drafts = self.continuation[pos..(pos + k).min(len)].to_vec();
        if let Some(j) = self.corrupt_at
            && let Some(d) = drafts.get_mut(j)
        {
            *d = (*d + 1) % VOCAB as u32;
        }
        drafts
    }
}

/// One speculative run. Returns the target for reuse.
fn speculative<D: Drafter>(
    target: TinyTarget,
    drafter: D,
    config: SpeculativeConfig,
    prompt: &[u32],
) -> (TinyTarget, Vec<u32>, FinishReason, SpeculativeStats) {
    let mut g = SpeculativeGenerator::new(target, drafter, config).expect("valid config");
    let (out, finish, stats) = g.generate(prompt).expect("generate");
    assert_eq!(
        g.model().len(),
        prompt.len() + out.len() - 1,
        "the cache holds every committed token except the last"
    );
    assert!(g.step().is_err(), "a step after the end is refused");
    let (target, _) = g.into_parts();
    (target, out, finish, stats)
}

fn config(k: usize, max_tokens: usize, eos: Vec<u32>) -> SpeculativeConfig {
    SpeculativeConfig {
        num_draft_tokens: k,
        max_tokens,
        eos_token_ids: eos,
    }
}

/// `n` token ids in `0..VOCAB`, fixed by `seed`.
fn tokens(seed: u64, n: usize) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) % VOCAB as u64) as u32
        })
        .collect()
}

/// The last index whose token does not occur earlier in `seq`.
fn last_first_occurrence(seq: &[u32]) -> usize {
    (0..seq.len())
        .rev()
        .find(|&i| !seq[..i].contains(&seq[i]))
        .unwrap_or(0)
}

#[test]
fn speculative_greedy_equals_plain_greedy_on_a_tiny_llama() {
    let prompts = vec![
        tokens(1, 1),
        tokens(2, 4),
        tokens(3, 9),
        vec![5, 9, 5, 9, 5, 9, 5],
        tokens(4, 6),
    ];
    let mut target = TinyTarget::new();
    let mut checked = 0usize;
    let mut rejected_midway = 0usize;
    for prompt in &prompts {
        let free_config = config(1, MAX_TOKENS, Vec::new());
        let (free, free_finish, gap) = plain(&mut target, prompt, &free_config);
        assert_eq!(free_finish, FinishReason::Length);
        if gap < NEAR_TIE {
            println!("skip prompt {prompt:?}: plain run has a top-2 logit gap of {gap:e}");
            continue;
        }
        checked += 1;
        let oracle = |corrupt_at| OracleDraft {
            prompt_len: prompt.len(),
            continuation: free.clone(),
            corrupt_at,
        };

        for k in [1usize, 3, 5] {
            let what = format!("prompt {prompt:?} k {k}");
            let cfg = config(k, MAX_TOKENS, Vec::new());

            let (t, out, finish, _) =
                speculative(target, PromptLookup::default(), cfg.clone(), prompt);
            assert_eq!(
                (&out, finish),
                (&free, FinishReason::Length),
                "{what}: prompt lookup"
            );

            let (t, out, finish, stats) = speculative(t, oracle(None), cfg.clone(), prompt);
            assert_eq!(
                (&out, finish),
                (&free, FinishReason::Length),
                "{what}: oracle"
            );
            assert_eq!(
                stats.accepted, stats.drafted,
                "{what}: oracle accepts all: {stats:?}"
            );
            assert_eq!(
                stats.target_forwards,
                MAX_TOKENS.div_ceil(k + 1),
                "{what}: oracle forwards: {stats:?}"
            );

            let j = k / 2;
            let (t, out, finish, stats) = speculative(t, oracle(Some(j)), cfg.clone(), prompt);
            assert_eq!(
                (&out, finish),
                (&free, FinishReason::Length),
                "{what}: corrupt at {j}"
            );
            assert!(
                stats.accepted < stats.drafted,
                "{what}: drafts rejected: {stats:?}"
            );
            rejected_midway += 1;

            let e = last_first_occurrence(&free);
            let eos_cfg = config(k, MAX_TOKENS, vec![free[e]]);
            let (t, out, finish, _) = speculative(t, oracle(None), eos_cfg.clone(), prompt);
            assert_eq!(
                (&out[..], finish),
                (&free[..=e], FinishReason::Eos),
                "{what}: EOS at {e}"
            );
            let (t, out, finish, _) =
                speculative(t, PromptLookup::default(), eos_cfg.clone(), prompt);
            assert_eq!(
                (&out[..], finish),
                (&free[..=e], FinishReason::Eos),
                "{what}: lookup EOS"
            );

            let mut t = t;
            let (want, want_finish, _) = plain(&mut t, prompt, &eos_cfg);
            assert_eq!((&want[..], want_finish), (&free[..=e], FinishReason::Eos));

            let max = k + 3;
            let len_cfg = config(k, max, Vec::new());
            let (t, out, finish, _) = speculative(t, oracle(None), len_cfg, prompt);
            assert_eq!(
                (&out[..], finish),
                (&free[..max], FinishReason::Length),
                "{what}: max {max}"
            );
            target = t;
        }
    }
    assert!(
        checked >= 3,
        "only {checked} prompts were free of near-ties"
    );
    assert!(rejected_midway > 0);
}
