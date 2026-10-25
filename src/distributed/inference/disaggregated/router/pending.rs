//! One in-flight request on the router: its stream, its workers, and the
//! order of their events.
//!
//! nexar delivers the events of one worker in send order. The events of the
//! two workers of a request have no order, so decode's tokens can overtake
//! prefill's first token. Decode events wait in `held` until the first
//! prefill event arrives. Every later event passes in arrival order.
//!
//! The request depends on its prefill worker until decode holds the KV
//! handoff: a handoff-accepted event from prefill, or any decode event, says so.

use nexar::Rank;
use std::fmt;
use tokio::sync::mpsc::UnboundedSender;

use super::super::protocol::{EventBody, Stage, TokenEvent};

/// The item a request stream yields.
pub(crate) type StreamItem = anyhow::Result<TokenEvent>;

/// A worker reported that it could not finish a request.
///
/// It reaches the caller inside the `anyhow::Error` of a stream item. Use
/// `err.downcast_ref::<WorkerError>()` to read it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerError {
    pub request_id: u64,
    pub stage: Stage,
    pub rank: Rank,
    pub message: String,
}

impl fmt::Display for WorkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stage = match self.stage {
            Stage::Prefill => "prefill",
            Stage::Decode => "decode",
        };
        write!(
            f,
            "{stage} worker rank {} failed request {}: {}",
            self.rank, self.request_id, self.message
        )
    }
}

impl std::error::Error for WorkerError {}

/// What one event did to its request.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Received {
    /// The stream ended: a finish, an error, or a dropped receiver.
    pub(crate) done: bool,
    /// The first prefill event arrived: the request leaves its prefill worker's load.
    pub(crate) left_prefill: bool,
    /// The first token arrived and decode has not confirmed the handoff:
    /// the handoff clock starts.
    pub(crate) awaits_handoff: bool,
}

/// One in-flight request.
pub(crate) struct Pending {
    tx: UnboundedSender<StreamItem>,
    pub(crate) prefill_rank: Rank,
    pub(crate) decode_rank: Rank,
    /// The stream index of the next token.
    next_index: u32,
    /// Decode events that arrived before the first prefill event, in arrival order.
    held: Vec<EventBody>,
    /// No prefill event has arrived. The request counts against its prefill
    /// worker's load.
    pub(crate) in_prefill: bool,
    /// Decode holds the KV handoff, so the prefill worker is no longer needed.
    pub(crate) accepted: bool,
}

impl Pending {
    pub(crate) fn new(
        tx: UnboundedSender<StreamItem>,
        prefill_rank: Rank,
        decode_rank: Rank,
    ) -> Self {
        Self {
            tx,
            prefill_rank,
            decode_rank,
            next_index: 0,
            held: Vec::new(),
            in_prefill: true,
            accepted: false,
        }
    }

    /// Ends the stream with `error`.
    pub(crate) fn end(&self, error: anyhow::Error) {
        let _ = self.tx.send(Err(error));
    }

    /// Applies `body` from `src`. An event from a rank outside the request
    /// changes nothing.
    pub(crate) fn receive(&mut self, request_id: u64, src: Rank, body: EventBody) -> Received {
        let mut out = Received::default();
        if src == self.decode_rank {
            self.accepted = true;
            if self.in_prefill {
                self.held.push(body);
            } else {
                out.done = self.emit(request_id, src, body);
            }
            return out;
        }
        if src != self.prefill_rank {
            tracing::warn!(
                src,
                request_id,
                "dropping an event from a rank outside the request"
            );
            return out;
        }
        if body == EventBody::HandoffAccepted {
            self.accepted = true;
            return out;
        }
        out.left_prefill = std::mem::take(&mut self.in_prefill);
        let continues = matches!(body, EventBody::Token { finish: None, .. });
        out.done = self.emit(request_id, src, body);
        if out.left_prefill {
            for body in std::mem::take(&mut self.held) {
                if out.done {
                    break;
                }
                out.done = self.emit(request_id, self.decode_rank, body);
            }
            out.awaits_handoff = continues && !out.done && !self.accepted;
        }
        out
    }

    /// Sends `body` from `src` to the stream. Returns whether the stream ended.
    fn emit(&mut self, request_id: u64, src: Rank, body: EventBody) -> bool {
        let (item, done) = match body {
            EventBody::Token { token, finish } => {
                let index = self.next_index;
                self.next_index += 1;
                let t = TokenEvent {
                    index,
                    token,
                    finish,
                };
                (Ok(t), finish.is_some())
            }
            EventBody::Error { stage, message } => {
                let e = WorkerError {
                    request_id,
                    stage,
                    rank: src,
                    message,
                };
                (Err(e.into()), true)
            }
            EventBody::HandoffAccepted => return false,
        };
        self.tx.send(item).is_err() || done
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::protocol::FinishReason;
    use super::*;
    use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

    const PREFILL: Rank = 1;
    const DECODE: Rank = 2;

    fn tok(token: u32, finish: Option<FinishReason>) -> EventBody {
        EventBody::Token { token, finish }
    }

    fn pending() -> (Pending, UnboundedReceiver<StreamItem>) {
        let (tx, rx) = unbounded_channel();
        (Pending::new(tx, PREFILL, DECODE), rx)
    }

    fn drain(rx: &mut UnboundedReceiver<StreamItem>) -> Vec<(u32, u32)> {
        std::iter::from_fn(|| rx.try_recv().ok())
            .map(|item| item.unwrap())
            .map(|t| (t.index, t.token))
            .collect()
    }

    #[test]
    fn decode_tokens_wait_for_the_first_prefill_token() {
        let (mut p, mut rx) = pending();
        assert_eq!(p.receive(7, DECODE, tok(20, None)), Received::default());
        p.receive(7, DECODE, tok(30, None));
        assert!(rx.try_recv().is_err(), "decode waits for prefill");
        let r = p.receive(7, PREFILL, tok(10, None));
        assert!(r.left_prefill && !r.done);
        assert!(!r.awaits_handoff, "decode events confirm the handoff");
        assert_eq!(drain(&mut rx), [(0, 10), (1, 20), (2, 30)]);
        let r = p.receive(7, DECODE, tok(40, Some(FinishReason::Eos)));
        assert!(r.done);
        let last = rx.try_recv().unwrap().unwrap();
        assert_eq!((last.index, last.finish), (3, Some(FinishReason::Eos)));
    }

    #[test]
    fn a_held_finish_ends_the_stream_after_the_prefill_token() {
        let (mut p, mut rx) = pending();
        p.receive(7, DECODE, tok(20, Some(FinishReason::Length)));
        assert!(p.receive(7, PREFILL, tok(10, None)).done);
        assert_eq!(drain(&mut rx), [(0, 10), (1, 20)]);
    }

    #[test]
    fn a_prefill_error_drops_held_decode_events() {
        let (mut p, mut rx) = pending();
        p.receive(7, DECODE, tok(20, None));
        let err = EventBody::Error {
            stage: Stage::Prefill,
            message: "oom".into(),
        };
        let r = p.receive(7, PREFILL, err);
        assert!(r.done && r.left_prefill && !r.awaits_handoff);
        let err = rx.try_recv().unwrap().unwrap_err();
        let w = err.downcast_ref::<WorkerError>().expect("a worker error");
        assert_eq!((w.stage, w.rank), (Stage::Prefill, PREFILL));
        assert!(rx.try_recv().is_err(), "the held token is dropped");
    }

    #[test]
    fn the_handoff_clock_starts_at_a_continuing_first_token_only() {
        let (mut p, _rx) = pending();
        assert!(p.receive(7, PREFILL, tok(10, None)).awaits_handoff);
        assert!(!p.accepted);
        let (mut p, _rx) = pending();
        let r = p.receive(7, PREFILL, tok(10, Some(FinishReason::Eos)));
        assert!(r.done && !r.awaits_handoff, "no decode phase, no clock");
    }

    #[test]
    fn handoff_accepted_changes_nothing_in_the_stream() {
        let (mut p, mut rx) = pending();
        p.receive(7, PREFILL, tok(10, None));
        let r = p.receive(7, PREFILL, EventBody::HandoffAccepted);
        assert_eq!(r, Received::default());
        assert!(p.accepted);
        p.receive(7, DECODE, tok(20, None));
        p.receive(7, DECODE, tok(30, Some(FinishReason::Stop)));
        assert_eq!(drain(&mut rx), [(0, 10), (1, 20), (2, 30)]);
    }

    #[test]
    fn a_handoff_error_after_the_first_token_ends_the_stream() {
        let (mut p, mut rx) = pending();
        p.receive(7, PREFILL, tok(10, None));
        let err = EventBody::Error {
            stage: Stage::Prefill,
            message: "handoff".into(),
        };
        assert!(p.receive(7, PREFILL, err).done);
        assert_eq!(rx.try_recv().unwrap().unwrap().token, 10);
        assert!(rx.try_recv().unwrap().is_err());
    }

    #[test]
    fn an_event_from_another_rank_is_dropped() {
        let (mut p, mut rx) = pending();
        let r = p.receive(7, 3, tok(99, Some(FinishReason::Eos)));
        assert_eq!(r, Received::default());
        assert!(rx.try_recv().is_err());
        assert!(p.in_prefill && !p.accepted);
    }
}
