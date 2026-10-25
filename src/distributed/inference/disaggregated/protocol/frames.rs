//! The four frames of the disaggregated protocol.
//!
//! Each frame starts with the header of [`super::codec`]. The body layouts:
//!
//! ```text
//! PrefillRequest: [decode_rank: u32][params: GenerationParams][prompt: u32 count, u32 ids]
//! KvHandoff:      [params: GenerationParams][generated: u32]
//!                 [context: u32 count, u32 ids][kv_cache: u64 length, bytes]
//! Cancel:         (no body)
//! Event:          [seq: u32][body: u8]
//!                 body 0 = token: [token: u32][finish: u8]   — 0 none, else FinishReason
//!                 body 1 = error: [stage: u8][message: u32 length, UTF-8]
//! ```
//!
//! `seq` orders the events of one request: 0 is the first token from prefill,
//! and decode step `n` sends `n`. An error takes the seq of the token it
//! replaces. nexar does not keep the order of separate sends, so the router
//! reorders by `seq`.

use anyhow::{Result, bail};
use nexar::Rank;

use super::codec::{FrameKind, Reader, Writer};
use super::params::{FinishReason, GenerationParams, Stage};

/// Router → prefill worker: run prefill for `prompt`, then hand off to `decode_rank`.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefillRequest {
    pub request_id: u64,
    pub decode_rank: Rank,
    pub params: GenerationParams,
    pub prompt: Vec<u32>,
}

impl PrefillRequest {
    /// # Errors
    ///
    /// A prompt or list above the `u32` length limit.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut w = Writer::frame(FrameKind::PrefillRequest, self.request_id);
        w.u32(self.decode_rank);
        self.params.write(&mut w);
        w.u32s("prompt", &self.prompt);
        w.finish()
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        let (mut r, request_id) = Reader::frame(buf, FrameKind::PrefillRequest)?;
        let out = Self {
            request_id,
            decode_rank: r.u32("decode_rank")?,
            params: GenerationParams::read(&mut r)?,
            prompt: r.u32s("prompt")?,
        };
        r.end()?;
        Ok(out)
    }
}

/// Prefill worker → decode worker: the state decode resumes from.
///
/// - `params`: the request params, with `rule.seed` resolved to the key prefill drew with.
/// - `context`: the prompt, then every token sampled so far.
/// - `generated`: how many trailing ids of `context` are sampled tokens.
/// - `kv_cache`: the serialized cache after the prompt. Its format is the
///   contract between [`super::super::forward::PrefillFn`] and
///   [`super::super::forward::DecodeOpenFn`].
#[derive(Debug, Clone, PartialEq)]
pub struct KvHandoff {
    pub request_id: u64,
    pub params: GenerationParams,
    pub generated: u32,
    pub context: Vec<u32>,
    pub kv_cache: Vec<u8>,
}

impl KvHandoff {
    /// # Errors
    ///
    /// A context or list above the `u32` length limit.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut w = Writer::frame(FrameKind::KvHandoff, self.request_id);
        self.params.write(&mut w);
        w.u32(self.generated);
        w.u32s("context", &self.context);
        w.bytes(&self.kv_cache);
        w.finish()
    }

    /// # Errors
    ///
    /// A malformed frame, or `generated` of 0 or above the context length.
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        let (mut r, request_id) = Reader::frame(buf, FrameKind::KvHandoff)?;
        let out = Self {
            request_id,
            params: GenerationParams::read(&mut r)?,
            generated: r.u32("generated")?,
            context: r.u32s("context")?,
            kv_cache: r.bytes("kv_cache")?,
        };
        r.end()?;
        if out.generated == 0 || out.generated as usize >= out.context.len() {
            bail!(
                "KV handoff for request {request_id} has {} generated tokens in a context of {}: \
                 expected at least 1 and fewer than the context",
                out.generated,
                out.context.len()
            );
        }
        Ok(out)
    }
}

/// Router → worker: stop request `request_id` and free its state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancel {
    pub request_id: u64,
}

impl Cancel {
    pub fn to_bytes(&self) -> Vec<u8> {
        Writer::header(FrameKind::Cancel, self.request_id)
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        let (r, request_id) = Reader::frame(buf, FrameKind::Cancel)?;
        r.end()?;
        Ok(Self { request_id })
    }
}

/// One sampled token of a request stream.
///
/// - `index`: the position in the stream, 0 for the first token.
/// - `finish`: `Some` on the last token only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenEvent {
    pub index: u32,
    pub token: u32,
    pub finish: Option<FinishReason>,
}

/// The body of an [`Event`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventBody {
    /// A sampled token. Its `index` is the event seq.
    Token(TokenEvent),
    /// The request failed at `stage`. No event follows.
    Error { stage: Stage, message: String },
}

/// Worker → router: one event of request `request_id` at position `seq`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub request_id: u64,
    pub seq: u32,
    pub body: EventBody,
}

impl Event {
    pub fn token(request_id: u64, token: TokenEvent) -> Self {
        Self {
            request_id,
            seq: token.index,
            body: EventBody::Token(token),
        }
    }

    pub fn error(request_id: u64, seq: u32, stage: Stage, message: String) -> Self {
        Self {
            request_id,
            seq,
            body: EventBody::Error { stage, message },
        }
    }

    /// # Errors
    ///
    /// A message above the `u32` length limit.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut w = Writer::frame(FrameKind::Event, self.request_id);
        w.u32(self.seq);
        match &self.body {
            EventBody::Token(t) => {
                w.u8(0);
                w.u32(t.token);
                w.u8(t.finish.map_or(0, |f| f as u8));
            }
            EventBody::Error { stage, message } => {
                w.u8(1);
                w.u8(*stage as u8);
                w.str("message", message);
            }
        }
        w.finish()
    }

    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        let (mut r, request_id) = Reader::frame(buf, FrameKind::Event)?;
        let seq = r.u32("seq")?;
        let body = match r.u8("body")? {
            0 => EventBody::Token(TokenEvent {
                index: seq,
                token: r.u32("token")?,
                finish: FinishReason::from_wire(r.u8("finish")?)?,
            }),
            1 => EventBody::Error {
                stage: Stage::from_wire(r.u8("stage")?)?,
                message: r.str("message")?,
            },
            other => bail!("unknown event body {other}"),
        };
        r.end()?;
        Ok(Self {
            request_id,
            seq,
            body,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::speculative::SampleRule;
    use crate::ops::traits::inference::FilterRule;

    fn params() -> GenerationParams {
        GenerationParams {
            rule: SampleRule {
                repeat_penalty: 1.2,
                penalty_last_n: 16,
                seed: Some(9),
                ..SampleRule::new(FilterRule {
                    temperature: 0.8,
                    top_k: 50,
                    top_p: 0.95,
                    min_p: 0.01,
                })
            },
            max_new_tokens: 64,
            eos_tokens: vec![2],
            stop_tokens: vec![10, 11],
        }
    }

    #[test]
    fn prefill_request_round_trips_with_params() {
        let req = PrefillRequest {
            request_id: 42,
            decode_rank: 3,
            params: params(),
            prompt: vec![1, 500, 128_000],
        };
        assert_eq!(
            PrefillRequest::from_bytes(&req.to_bytes().unwrap()).unwrap(),
            req
        );
    }

    #[test]
    fn kv_handoff_round_trips_and_checks_generated() {
        let h = KvHandoff {
            request_id: 7,
            params: params(),
            generated: 1,
            context: vec![5, 6, 7, 8],
            kv_cache: vec![0xAB; 33],
        };
        assert_eq!(KvHandoff::from_bytes(&h.to_bytes().unwrap()).unwrap(), h);
        for generated in [0, 4] {
            let bad = KvHandoff {
                generated,
                ..h.clone()
            };
            assert!(KvHandoff::from_bytes(&bad.to_bytes().unwrap()).is_err());
        }
    }

    #[test]
    fn events_round_trip_tokens_finishes_and_errors() {
        let cancel = Cancel { request_id: 3 };
        assert_eq!(Cancel::from_bytes(&cancel.to_bytes()).unwrap(), cancel);
        for finish in [
            None,
            Some(FinishReason::Eos),
            Some(FinishReason::Stop),
            Some(FinishReason::Length),
        ] {
            let ev = Event::token(
                9,
                TokenEvent {
                    index: 4,
                    token: 77,
                    finish,
                },
            );
            assert_eq!(Event::from_bytes(&ev.to_bytes().unwrap()).unwrap(), ev);
        }
        let err = Event::error(9, 0, Stage::Prefill, "out of memory: ü".into());
        let back = Event::from_bytes(&err.to_bytes().unwrap()).unwrap();
        assert_eq!(back, err);
        let err = Event::error(9, 5, Stage::Decode, String::new());
        assert_eq!(Event::from_bytes(&err.to_bytes().unwrap()).unwrap(), err);
    }

    #[test]
    fn a_frame_of_another_kind_is_refused() {
        let cancel = Cancel { request_id: 3 }.to_bytes();
        let err = Event::from_bytes(&cancel).unwrap_err();
        assert!(err.to_string().contains("Event"), "{err}");
    }
}
