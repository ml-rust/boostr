//! Generation params, finish reasons and worker stages carried by the protocol.
//!
//! # `GenerationParams` wire layout
//!
//! ```text
//! [temperature:       f32]
//! [top_k:             u64]
//! [top_p:             f32]
//! [min_p:             f32]
//! [repeat_penalty:    f32]
//! [frequency_penalty: f32]
//! [presence_penalty:  f32]
//! [penalty_last_n:    u64]
//! [has_seed:          u8]   — 0 or 1
//! [seed:              u64]  — 0 when has_seed is 0
//! [max_new_tokens:    u32]
//! [eos_tokens:        u32 count, then u32 ids]
//! [stop_tokens:       u32 count, then u32 ids]
//! ```

use anyhow::{Result, anyhow, bail};

use super::codec::{Reader, Writer};
use crate::inference::speculative::SampleRule;
use crate::ops::traits::inference::FilterRule;

/// The sampling and stop config of one disaggregated request.
///
/// - `rule`: the sampler. `rule.seed == None` takes one key from
///   `numr::random::next_call_seed` at prefill. The prefill worker forwards
///   the resolved key, so prefill and decode draw from one Philox stream.
/// - `max_new_tokens`: the token cap, the first token included. It must be at least 1.
/// - `eos_tokens`: a sampled id here ends the stream with [`FinishReason::Eos`].
/// - `stop_tokens`: a sampled id here ends the stream with [`FinishReason::Stop`].
///
/// The stream carries the ending token itself. An id in both lists ends with `Eos`.
///
/// Mirostat, DRY, typical-p and grammar constraints have no field, as in [`SampleRule`].
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationParams {
    pub rule: SampleRule,
    pub max_new_tokens: u32,
    pub eos_tokens: Vec<u32>,
    pub stop_tokens: Vec<u32>,
}

impl GenerationParams {
    /// Greedy decoding, no penalty, no stop id, `max_new_tokens` tokens.
    pub fn greedy(max_new_tokens: u32) -> Self {
        Self {
            rule: SampleRule::new(FilterRule::temperature_only(0.0)),
            max_new_tokens,
            eos_tokens: Vec::new(),
            stop_tokens: Vec::new(),
        }
    }

    /// Checks the rule and the token cap.
    ///
    /// # Errors
    ///
    /// The first field of `rule` out of range, or `max_new_tokens == 0`.
    pub fn check(&self) -> Result<()> {
        self.rule
            .check()
            .map_err(|e| anyhow!("invalid sampling rule: {e}"))?;
        if self.max_new_tokens == 0 {
            bail!("max_new_tokens is 0: pass at least 1, the first token counts");
        }
        Ok(())
    }

    /// The finish reason of the `generated`-th token `token`, counted from 1.
    pub fn finish_for(&self, token: u32, generated: u32) -> Option<FinishReason> {
        if self.eos_tokens.contains(&token) {
            Some(FinishReason::Eos)
        } else if self.stop_tokens.contains(&token) {
            Some(FinishReason::Stop)
        } else if generated >= self.max_new_tokens {
            Some(FinishReason::Length)
        } else {
            None
        }
    }

    pub(super) fn write(&self, w: &mut Writer) {
        let r = &self.rule;
        w.f32(r.filter.temperature);
        w.u64(r.filter.top_k as u64);
        w.f32(r.filter.top_p);
        w.f32(r.filter.min_p);
        w.f32(r.repeat_penalty);
        w.f32(r.frequency_penalty);
        w.f32(r.presence_penalty);
        w.u64(r.penalty_last_n as u64);
        w.u8(u8::from(r.seed.is_some()));
        w.u64(r.seed.unwrap_or(0));
        w.u32(self.max_new_tokens);
        w.u32s("eos_tokens", &self.eos_tokens);
        w.u32s("stop_tokens", &self.stop_tokens);
    }

    pub(super) fn read(r: &mut Reader<'_>) -> Result<Self> {
        let filter = FilterRule {
            temperature: r.f32("temperature")?,
            top_k: to_usize(r.u64("top_k")?, "top_k")?,
            top_p: r.f32("top_p")?,
            min_p: r.f32("min_p")?,
        };
        let repeat_penalty = r.f32("repeat_penalty")?;
        let frequency_penalty = r.f32("frequency_penalty")?;
        let presence_penalty = r.f32("presence_penalty")?;
        let penalty_last_n = to_usize(r.u64("penalty_last_n")?, "penalty_last_n")?;
        let has_seed = match r.u8("has_seed")? {
            0 => false,
            1 => true,
            other => bail!("field `has_seed` is {other}, expected 0 or 1"),
        };
        let seed = r.u64("seed")?;
        Ok(Self {
            rule: SampleRule {
                filter,
                repeat_penalty,
                frequency_penalty,
                presence_penalty,
                penalty_last_n,
                seed: has_seed.then_some(seed),
            },
            max_new_tokens: r.u32("max_new_tokens")?,
            eos_tokens: r.u32s("eos_tokens")?,
            stop_tokens: r.u32s("stop_tokens")?,
        })
    }
}

fn to_usize(v: u64, field: &str) -> Result<usize> {
    usize::try_from(v).map_err(|_| anyhow!("field `{field}` = {v} does not fit in usize"))
}

/// Why a request stream ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FinishReason {
    /// The last token is in `eos_tokens`.
    Eos = 1,
    /// The last token is in `stop_tokens`.
    Stop = 2,
    /// The stream reached `max_new_tokens`.
    Length = 3,
}

impl FinishReason {
    pub(super) fn from_wire(v: u8) -> Result<Option<Self>> {
        Ok(match v {
            0 => None,
            1 => Some(Self::Eos),
            2 => Some(Self::Stop),
            3 => Some(Self::Length),
            other => bail!("unknown finish reason {other}"),
        })
    }
}

/// The worker role that produced an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Stage {
    Prefill = 1,
    Decode = 2,
}

impl Stage {
    pub(super) fn from_wire(v: u8) -> Result<Self> {
        Ok(match v {
            1 => Self::Prefill,
            2 => Self::Decode,
            other => bail!("unknown worker stage {other}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::codec::FrameKind;
    use super::*;

    fn sampled_params() -> GenerationParams {
        GenerationParams {
            rule: SampleRule {
                filter: FilterRule {
                    temperature: 0.7,
                    top_k: 40,
                    top_p: 0.9,
                    min_p: 0.05,
                },
                repeat_penalty: 1.1,
                frequency_penalty: 0.2,
                presence_penalty: -0.3,
                penalty_last_n: 64,
                seed: Some(u64::MAX - 3),
            },
            max_new_tokens: 128,
            eos_tokens: vec![2, 128_001],
            stop_tokens: vec![13],
        }
    }

    fn round_trip(p: &GenerationParams) -> GenerationParams {
        let mut w = Writer::frame(FrameKind::Cancel, 0);
        p.write(&mut w);
        let buf = w.finish().unwrap();
        let (mut r, _) = Reader::frame(&buf, FrameKind::Cancel).unwrap();
        let out = GenerationParams::read(&mut r).unwrap();
        r.end().unwrap();
        out
    }

    #[test]
    fn every_field_round_trips() {
        let p = sampled_params();
        assert_eq!(round_trip(&p), p);
        let unseeded = GenerationParams {
            rule: SampleRule {
                seed: None,
                ..p.rule
            },
            ..p
        };
        assert_eq!(round_trip(&unseeded).rule.seed, None);
    }

    #[test]
    fn check_refuses_zero_tokens_and_bad_rules() {
        sampled_params().check().unwrap();
        GenerationParams::greedy(1).check().unwrap();
        assert!(GenerationParams::greedy(0).check().is_err());
        let mut bad = sampled_params();
        bad.rule.repeat_penalty = 0.0;
        let err = bad.check().unwrap_err();
        assert!(err.to_string().contains("repeat_penalty"), "{err}");
    }

    #[test]
    fn finish_prefers_eos_then_stop_then_length() {
        let mut p = sampled_params();
        p.stop_tokens.push(2);
        assert_eq!(p.finish_for(2, 1), Some(FinishReason::Eos));
        assert_eq!(p.finish_for(13, 1), Some(FinishReason::Stop));
        assert_eq!(p.finish_for(7, 127), None);
        assert_eq!(p.finish_for(7, 128), Some(FinishReason::Length));
        assert_eq!(p.finish_for(2, 128), Some(FinishReason::Eos));
    }
}
