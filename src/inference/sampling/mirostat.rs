//! Mirostat v2: a sampler that steers the surprise of each token toward a target.

use std::borrow::Cow;

use chacha20::ChaCha12Rng;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Mirostat v2 sampler state.
///
/// `mu` is the surprise bound, in bits. Each step keeps the tokens whose
/// surprise `-log2(p)` is at most `mu` and samples among them. Then `mu`
/// moves by `eta * (surprise - tau)`, so the surprise trends toward `tau`.
///
/// The generator is ChaCha12, the algorithm of `rand::rngs::StdRng`. A seed
/// gives the stream of `StdRng::seed_from_u64(seed)`.
#[derive(Debug)]
pub struct MirostatV2 {
    tau: f32,
    eta: f32,
    mu: f32,
    rng: ChaCha12Rng,
}

/// A [`MirostatV2`] at one point of its stream, for a checkpoint or a handoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MirostatV2State {
    /// The target surprise, as `f32` bits.
    pub tau_bits: u32,
    /// The learning rate, as `f32` bits.
    pub eta_bits: u32,
    /// The current surprise bound, as `f32` bits.
    pub mu_bits: u32,
    /// The ChaCha12 key.
    pub seed: [u8; 32],
    /// The next 32-bit word of the stream.
    pub word_pos: u128,
}

impl MirostatV2 {
    /// A sampler with target surprise `tau`, learning rate `eta` and `mu = 2 * tau`.
    ///
    /// `seed: None` takes one seed from `numr::random::next_call_seed`.
    pub fn new(tau: f32, eta: f32, seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(numr::random::next_call_seed);
        Self {
            tau,
            eta,
            mu: 2.0 * tau,
            rng: ChaCha12Rng::seed_from_u64(seed),
        }
    }

    /// The current surprise bound, in bits.
    pub fn mu(&self) -> f32 {
        self.mu
    }

    /// The target surprise, in bits.
    pub fn tau(&self) -> f32 {
        self.tau
    }

    /// The learning rate of `mu`.
    pub fn eta(&self) -> f32 {
        self.eta
    }

    /// The full state. [`Self::from_state`] continues the same stream.
    pub fn state(&self) -> MirostatV2State {
        MirostatV2State {
            tau_bits: self.tau.to_bits(),
            eta_bits: self.eta.to_bits(),
            mu_bits: self.mu.to_bits(),
            seed: self.rng.get_seed(),
            word_pos: self.rng.get_word_pos(),
        }
    }

    /// The sampler `state` describes.
    pub fn from_state(state: &MirostatV2State) -> Self {
        let mut rng = ChaCha12Rng::from_seed(state.seed);
        rng.set_word_pos(state.word_pos);
        Self {
            tau: f32::from_bits(state.tau_bits),
            eta: f32::from_bits(state.eta_bits),
            mu: f32::from_bits(state.mu_bits),
            rng,
        }
    }

    /// Sample one token from one row of logits, and update `mu`.
    ///
    /// A `temperature` above 0 and not 1 scales the logits by its inverse.
    /// The candidates are the tokens with `p > 0` and surprise at most `mu`,
    /// in descending probability. Ties keep id order. With no candidate, the
    /// most likely token is the only one. One uniform draw picks a candidate
    /// by its renormalized mass.
    ///
    /// Returns the token id and `ln(p)` of the token in the full distribution.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `logits` is empty, holds a NaN or `+inf`,
    /// or holds no finite value, or when `temperature` is not finite. A
    /// `-inf` logit is a masked token. A rejected call changes no state.
    pub fn sample(&mut self, logits: &[f32], temperature: f32) -> Result<(u32, f32)> {
        check_logits(logits)?;
        if !temperature.is_finite() {
            return Err(Error::InvalidArgument {
                arg: "temperature",
                reason: format!(
                    "mirostat: got {temperature}; pass a finite value, or 1.0 for no scaling"
                ),
            });
        }
        let scaled: Cow<[f32]> = if temperature != 1.0 && temperature > 0.0 {
            let inv_t = 1.0 / temperature;
            Cow::Owned(logits.iter().map(|&l| l * inv_t).collect())
        } else {
            Cow::Borrowed(logits)
        };

        let max_logit = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<(usize, f32)> = scaled
            .iter()
            .enumerate()
            .map(|(i, &l)| (i, (l - max_logit).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in probs.iter_mut() {
            *p /= sum;
        }
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mu = self.mu;
        let mut candidates: Vec<(usize, f32)> = probs
            .iter()
            .filter(|(_, p)| *p > 0.0 && -p.log2() <= mu)
            .cloned()
            .collect();
        if candidates.is_empty() {
            candidates.push(probs[0]);
        }

        let total: f32 = candidates.iter().map(|(_, p)| p).sum();
        let r: f32 = self.rng.random();
        let mut cumsum = 0.0f32;
        let (mut chosen, mut chosen_prob) = candidates[0];
        for &(id, p) in &candidates {
            cumsum += p / total;
            if cumsum > r {
                chosen = id;
                chosen_prob = p;
                break;
            }
        }

        let surprise = if chosen_prob > 0.0 {
            -chosen_prob.log2()
        } else {
            self.tau
        };
        self.mu -= self.eta * (surprise - self.tau);
        Ok((chosen as u32, chosen_prob.ln()))
    }
}

/// Reject a row the sort and the softmax cannot order.
fn check_logits(logits: &[f32]) -> Result<()> {
    let reason = if logits.is_empty() {
        "mirostat: got an empty logits row; pass one row of the vocab".to_string()
    } else if let Some(i) = logits
        .iter()
        .position(|l| l.is_nan() || *l == f32::INFINITY)
    {
        format!(
            "mirostat: logit {i} is {}; pass finite logits, or -inf for a masked token",
            logits[i]
        )
    } else if logits.iter().all(|l| *l == f32::NEG_INFINITY) {
        "mirostat: every logit is -inf; leave at least one token unmasked".to_string()
    } else {
        return Ok(());
    };
    Err(Error::InvalidArgument {
        arg: "logits",
        reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(salt: u32) -> Vec<f32> {
        (0..64u32)
            .map(|i| ((i.wrapping_mul(2_654_435_761) ^ salt) % 97) as f32 * 0.1 - 4.0)
            .collect()
    }

    fn run(sampler: &mut MirostatV2, steps: u32) -> Result<Vec<(u32, u32)>> {
        (0..steps)
            .map(|s| {
                let (id, lp) = sampler.sample(&row(s), 0.9)?;
                Ok((id, lp.to_bits()))
            })
            .collect()
    }

    #[test]
    fn a_seed_fixes_the_stream() -> Result<()> {
        let a = run(&mut MirostatV2::new(5.0, 0.1, Some(9)), 40)?;
        let b = run(&mut MirostatV2::new(5.0, 0.1, Some(9)), 40)?;
        let c = run(&mut MirostatV2::new(5.0, 0.1, Some(10)), 40)?;
        assert_eq!(a, b);
        assert_ne!(a, c);
        Ok(())
    }

    #[test]
    fn the_stream_equals_std_rng_seeded_the_same() {
        use rand::rngs::StdRng;
        let mut ours = MirostatV2::new(5.0, 0.1, Some(42)).rng;
        let mut std_rng = StdRng::seed_from_u64(42);
        let a: Vec<f32> = (0..64).map(|_| ours.random()).collect();
        let b: Vec<f32> = (0..64).map(|_| std_rng.random()).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn mu_moves_toward_tau() -> Result<()> {
        // A flat row: every token has surprise log2(64) = 6 bits, above tau.
        let mut sampler = MirostatV2::new(3.0, 0.5, Some(1));
        let start = sampler.mu();
        let flat = vec![0.0f32; 64];
        sampler.sample(&flat, 1.0)?;
        assert!(sampler.mu() < start, "{} !< {start}", sampler.mu());

        // A peaked row: the top token has surprise near 0, below tau.
        let mut peaked = vec![-30.0f32; 64];
        peaked[5] = 30.0;
        let mut sampler = MirostatV2::new(3.0, 0.5, Some(1));
        let (id, _) = sampler.sample(&peaked, 1.0)?;
        assert_eq!(id, 5);
        assert!(sampler.mu() > 6.0, "{}", sampler.mu());
        Ok(())
    }

    #[test]
    fn no_candidate_falls_back_to_the_top_token() -> Result<()> {
        // mu = 0: only a token with p = 1 passes, and none has it.
        let mut sampler = MirostatV2::new(0.0, 0.1, Some(3));
        let mut logits = vec![0.0f32; 16];
        logits[11] = 1.0;
        assert_eq!(sampler.sample(&logits, 1.0)?.0, 11);
        Ok(())
    }

    #[test]
    fn masked_tokens_are_never_sampled() -> Result<()> {
        let mut sampler = MirostatV2::new(5.0, 0.1, Some(4));
        let mut logits = vec![f32::NEG_INFINITY; 16];
        logits[2] = 0.0;
        logits[9] = 0.0;
        for _ in 0..20 {
            let (id, _) = sampler.sample(&logits, 0.7)?;
            assert!(id == 2 || id == 9, "{id}");
        }
        Ok(())
    }

    #[test]
    fn unorderable_rows_are_rejected() {
        let mut sampler = MirostatV2::new(5.0, 0.1, Some(5));
        let bad = [
            vec![],
            vec![0.0, f32::NAN, 1.0],
            vec![0.0, f32::INFINITY],
            vec![f32::NEG_INFINITY; 4],
        ];
        for logits in bad {
            match sampler.sample(&logits, 1.0) {
                Err(Error::InvalidArgument { arg: "logits", .. }) => {}
                other => panic!("{logits:?}: expected InvalidArgument, got {other:?}"),
            }
        }
        for t in [f32::NAN, f32::INFINITY] {
            assert!(matches!(
                sampler.sample(&[0.0, 1.0], t),
                Err(Error::InvalidArgument {
                    arg: "temperature",
                    ..
                })
            ));
        }
        assert_eq!(sampler.mu(), 10.0, "a rejected row leaves mu unchanged");
    }

    #[test]
    fn a_restored_state_continues_the_stream() -> std::result::Result<(), Box<dyn std::error::Error>>
    {
        let mut whole = MirostatV2::new(4.0, 0.2, Some(77));
        run(&mut whole, 13)?;
        let state = whole.state();
        let back: MirostatV2State = serde_json::from_str(&serde_json::to_string(&state)?)?;
        assert_eq!(back, state);
        let mut resumed = MirostatV2::from_state(&back);
        assert_eq!(resumed.mu().to_bits(), whole.mu().to_bits());
        assert_eq!(run(&mut resumed, 20)?, run(&mut whole, 20)?);
        Ok(())
    }
}
