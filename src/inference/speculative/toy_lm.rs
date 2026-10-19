//! Test support: a toy target model, test drafters and a plain greedy loop.
//!
//! The toy's next token hashes the whole fed prefix. A duplicate, missing or
//! stale cache position changes every later output. Half of its choices
//! depend on the last two tokens only, so contexts repeat and prompt lookup
//! finds matches. Its logits put the next token 1.0 above every other id, so
//! argmax equals the greedy next token.

use super::draft::Drafter;
use super::model::SpeculativeModel;
use super::types::{Draw, FinishReason, SampleRule, SpeculativeConfig, Verdict};
use super::verify::verify_logits;
use crate::error::{Error, Result};
use crate::test_utils::cpu_setup;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// FNV-1a over `prefix`, keyed by `seed`.
fn fnv(seed: u64, prefix: &[u32]) -> u64 {
    let mut h = FNV_OFFSET ^ seed;
    for &t in prefix {
        h ^= u64::from(t) + 1;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// The toy's greedy next token after `prefix`. `prefix` is never empty.
pub(crate) fn next_token(seed: u64, vocab: usize, prefix: &[u32]) -> u32 {
    let h = mix(fnv(seed, prefix));
    let n = prefix.len();
    let last = u64::from(prefix[n - 1]);
    let prev = if n >= 2 {
        u64::from(prefix[n - 2])
    } else {
        0xffff
    };
    let pick = if h.is_multiple_of(2) {
        mix(seed ^ (last << 20) ^ (prev << 40) ^ 0x9e37)
    } else {
        h >> 8
    };
    (pick % vocab as u64) as u32
}

/// The toy's logits after `prefix`: distinct values in `[-3, 3]`.
///
/// Every id except the next token takes a hashed rank in `[-3, 2)`. The
/// next token takes the largest other value plus 1.0.
pub(crate) fn toy_logits(seed: u64, vocab: usize, prefix: &[u32]) -> Vec<f32> {
    let next = next_token(seed, vocab, prefix) as usize;
    let mut out = vec![0.0f32; vocab];
    if vocab == 1 {
        return out;
    }
    let h = mix(fnv(seed, prefix) ^ 0x10c1_75ee_d5a1_7e57);
    let mut others: Vec<(u64, usize)> = (0..vocab)
        .filter(|&j| j != next)
        .map(|j| (mix(h ^ (j as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)), j))
        .collect();
    others.sort_unstable();
    let step = 5.0 / (vocab - 1) as f32;
    let mut top = f32::NEG_INFINITY;
    for (rank, &(key, j)) in others.iter().enumerate() {
        // A jitter below 0.4 of a step keeps the values distinct.
        let jitter = (key >> 40) as f32 / (1u64 << 24) as f32 * 0.4;
        out[j] = -3.0 + step * (rank as f32 + jitter);
        top = top.max(out[j]);
    }
    out[next] = top + 1.0;
    out
}

/// A deterministic causal model whose cache is the list of fed tokens.
pub(crate) struct ToyLm {
    pub vocab: usize,
    pub seed: u64,
    pub fed: Vec<u32>,
}

impl ToyLm {
    pub fn new(vocab: usize, seed: u64) -> Self {
        Self {
            vocab,
            seed,
            fed: Vec::new(),
        }
    }
}

impl SpeculativeModel for ToyLm {
    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn len(&self) -> usize {
        self.fed.len()
    }

    fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.fed.len() {
            return Err(Error::InferenceError {
                reason: format!("truncate to {len} past {}", self.fed.len()),
            });
        }
        self.fed.truncate(len);
        Ok(())
    }

    fn extend_greedy(&mut self, tokens: &[u32]) -> Result<Vec<u32>> {
        let mut ids = Vec::with_capacity(tokens.len());
        for &t in tokens {
            self.fed.push(t);
            ids.push(next_token(self.seed, self.vocab, &self.fed));
        }
        Ok(ids)
    }

    fn extend_sampled(&mut self, tokens: &[u32], rule: &SampleRule, draw: Draw) -> Result<Verdict> {
        if tokens.is_empty() {
            return Err(Error::InferenceError {
                reason: "extend_sampled needs the last committed token".into(),
            });
        }
        let start = self.fed.len();
        let mut rows = Vec::with_capacity(tokens.len() * self.vocab);
        for &t in tokens {
            self.fed.push(t);
            rows.extend(toy_logits(self.seed, self.vocab, &self.fed));
        }
        let (client, device) = cpu_setup();
        let logits =
            Tensor::<CpuRuntime>::from_slice(&rows, &[1, tokens.len(), self.vocab], &device)?;
        let (context, drafts) = self.fed.split_at(start + 1);
        verify_logits(&client, &logits, context, drafts, rule, draw)
    }
}

/// The exact target distribution after `prefix`, in f64.
///
/// It applies the penalties as `apply_sampling_penalties` does, in f32,
/// then the canonical filter of `FilterRule` in f64, ties kept. The second
/// value is the smallest gap between a filter threshold and a value it
/// tests. A tiny gap means f32 and f64 can keep different sets.
pub(crate) fn reference_probs(
    seed: u64,
    vocab: usize,
    prefix: &[u32],
    rule: &SampleRule,
) -> (Vec<f64>, f64) {
    let mut logits = toy_logits(seed, vocab, prefix);
    let mut counts = std::collections::BTreeMap::<u32, i32>::new();
    for &t in prefix {
        *counts.entry(t).or_insert(0) += 1;
    }
    for (&t, &c) in &counts {
        let l = &mut logits[t as usize];
        if rule.repeat_penalty != 1.0 {
            if *l > 0.0 {
                *l /= rule.repeat_penalty;
            } else {
                *l *= rule.repeat_penalty;
            }
        }
        *l -= rule.frequency_penalty * c as f32;
        *l -= rule.presence_penalty;
    }
    let f = rule.filter;
    let scaled: Vec<f64> = logits
        .iter()
        .map(|&l| f64::from(l) / f64::from(f.temperature))
        .collect();
    let top = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = scaled.iter().map(|&l| (l - top).exp()).collect();
    let z: f64 = exp.iter().sum();
    let p: Vec<f64> = exp.iter().map(|&e| e / z).collect();

    let mut margin = f64::INFINITY;
    let mut kept = p.clone();
    let mut sorted = p.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));
    /// Zero every entry below `thr`; record the smallest nonzero gap to it.
    fn cut(row: &mut [f64], thr: f64, margin: &mut f64) {
        for v in row.iter_mut() {
            if *v != thr {
                *margin = margin.min((*v - thr).abs());
            }
            if *v < thr {
                *v = 0.0;
            }
        }
    }
    if f.top_k_active(vocab) {
        let kth = sorted[f.top_k - 1];
        cut(&mut kept, kth, &mut margin);
        let mut unused = f64::INFINITY;
        cut(&mut sorted, kth, &mut unused);
    }
    if f.top_p_active() && vocab > 1 {
        let target = sorted.iter().sum::<f64>() * f64::from(f.top_p);
        let mut acc = 0.0;
        let mut tau = 0.0;
        for &v in &sorted {
            acc += v;
            margin = margin.min((acc - target).abs());
            if acc >= target {
                tau = v;
                break;
            }
        }
        cut(&mut kept, tau, &mut margin);
    }
    if f.min_p_active() {
        let thr = p.iter().copied().fold(0.0, f64::max) * f64::from(f.min_p);
        cut(&mut kept, thr, &mut margin);
    }
    let total: f64 = kept.iter().sum();
    (kept.iter().map(|&v| v / total).collect(), margin)
}

/// Plain greedy decoding: one `extend_greedy([x])` per emitted token.
///
/// Same rules as the generator: EOS first, then `max_tokens`.
pub(crate) fn plain_greedy<M: SpeculativeModel>(
    model: &mut M,
    prompt: &[u32],
    config: &SpeculativeConfig,
) -> Result<(Vec<u32>, FinishReason)> {
    model.truncate(0)?;
    let (head, last) = prompt.split_at(prompt.len() - 1);
    if !head.is_empty() {
        model.extend_greedy(head)?;
    }
    let mut x = last[0];
    let mut out = Vec::new();
    loop {
        let ids = model.extend_greedy(&[x])?;
        x = ids[0];
        out.push(x);
        if config.is_eos(x) {
            return Ok((out, FinishReason::Eos));
        }
        if out.len() >= config.max_tokens {
            return Ok((out, FinishReason::Length));
        }
    }
}

/// The toy's own greedy continuation of `context`, `n` tokens long.
pub(crate) fn toy_continuation(seed: u64, vocab: usize, context: &[u32], n: usize) -> Vec<u32> {
    let mut ctx = context.to_vec();
    for _ in 0..n {
        let t = next_token(seed, vocab, &ctx);
        ctx.push(t);
    }
    ctx.split_off(context.len())
}

/// Proposes exactly the toy's next `k` tokens.
pub(crate) struct Oracle {
    pub seed: u64,
    pub vocab: usize,
}

impl Drafter for Oracle {
    fn propose(&mut self, context: &[u32], k: usize) -> Vec<u32> {
        toy_continuation(self.seed, self.vocab, context, k)
    }
}

/// Proposes the toy's tokens 2..=k+1, skipping the next one.
pub(crate) struct OffByOne {
    pub seed: u64,
    pub vocab: usize,
}

impl Drafter for OffByOne {
    fn propose(&mut self, context: &[u32], k: usize) -> Vec<u32> {
        let mut c = toy_continuation(self.seed, self.vocab, context, k + 1);
        c.remove(0);
        c
    }
}

/// Proposes random ids, sometimes past the vocabulary and past `k`.
pub(crate) struct RandomDrafter {
    pub state: u64,
    pub vocab: usize,
}

impl RandomDrafter {
    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state >> 33
    }
}

impl Drafter for RandomDrafter {
    fn propose(&mut self, _context: &[u32], k: usize) -> Vec<u32> {
        let n = (self.next() % (k as u64 + 3)) as usize;
        (0..n)
            .map(|_| (self.next() % (self.vocab as u64 + 2)) as u32)
            .collect()
    }
}

/// Proposes `k` ids hashed from the whole context.
pub(crate) struct HashDrafter {
    pub salt: u64,
    pub vocab: usize,
}

impl Drafter for HashDrafter {
    fn propose(&mut self, context: &[u32], k: usize) -> Vec<u32> {
        let h = fnv(self.salt, context);
        (0..k as u64)
            .map(|i| (mix(h ^ i.wrapping_mul(0xa076_1d64_78bd_642f)) % self.vocab as u64) as u32)
            .collect()
    }
}

/// Never proposes a token.
pub(crate) struct EmptyDrafter;

impl Drafter for EmptyDrafter {
    fn propose(&mut self, _context: &[u32], _k: usize) -> Vec<u32> {
        Vec::new()
    }
}

/// Returns a fixed proposal on every call.
pub(crate) struct Scripted(pub Vec<u32>);

impl Drafter for Scripted {
    fn propose(&mut self, _context: &[u32], _k: usize) -> Vec<u32> {
        self.0.clone()
    }
}
