//! Test support: a toy target model, test drafters and a plain greedy loop.
//!
//! The toy's next token hashes the whole fed prefix. A duplicate, missing or
//! stale cache position changes every later output. Half of its choices
//! depend on the last two tokens only, so contexts repeat and prompt lookup
//! finds matches.

use super::draft::Drafter;
use super::model::SpeculativeModel;
use super::types::{FinishReason, SpeculativeConfig};
use crate::error::{Error, Result};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The toy's greedy next token after `prefix`. `prefix` is never empty.
pub(crate) fn next_token(seed: u64, vocab: usize, prefix: &[u32]) -> u32 {
    let mut h = FNV_OFFSET ^ seed;
    for &t in prefix {
        h ^= u64::from(t) + 1;
        h = h.wrapping_mul(FNV_PRIME);
    }
    let h = mix(h);
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
