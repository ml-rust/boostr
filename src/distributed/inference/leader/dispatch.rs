//! The leader's dispatcher: routes each logits row and stage error to the
//! request that waits for it, by request id.
//!
//! One listener task per source turns frames into events: logits from the
//! last stage, stage errors from every worker, and nexar's dead-peer
//! reports. One dispatcher task owns the table of waiting requests, so no
//! lock is held across a send or a receive. A lost stage fails every
//! waiting request, and every later one.
//!
//! A waiting request waits for the reply to one frame id. A reply to
//! another frame id of the same request id answers a frame whose caller
//! stopped waiting, and the dispatcher drops it.

use anyhow::{Result, anyhow, bail};
use nexar::{NexarClient, Rank};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

use super::super::transport::reply::peek_reply_key;
use super::super::transport::tasks::TaskGuard;
use super::super::transport::{LogitsFrame, StageError, tags};

/// What a waiting request gets: one logits row, or the error that ended it.
pub(super) type Reply = Result<Vec<f32>>;

enum Event {
    /// A request waits for the reply to frame `frame_id`.
    Await {
        request_id: u64,
        frame_id: u64,
        reply: oneshot::Sender<Reply>,
    },
    /// A request no longer waits: for frame `frame_id` only, or for any frame when `None`.
    Forget {
        request_id: u64,
        frame_id: Option<u64>,
    },
    Logits(LogitsFrame),
    Failed(StageError),
    /// A frame that did not parse, with the request id and frame id it holds.
    Malformed {
        rank: Rank,
        request_id: Option<u64>,
        frame_id: Option<u64>,
        error: anyhow::Error,
    },
    /// A stage rank disconnected or died.
    Lost {
        rank: Rank,
        reason: String,
    },
}

/// The dispatcher task and its listeners. Dropping it stops them, and every
/// waiting request gets an error.
pub(super) struct Dispatch {
    events: mpsc::UnboundedSender<Event>,
    _tasks: Vec<TaskGuard>,
}

impl Dispatch {
    /// Starts the dispatcher on the current tokio runtime.
    ///
    /// `last` sends the logits. Every rank in `workers` can send a stage
    /// error. A logits row of another length than `vocab_size` fails its request.
    ///
    /// # Errors
    ///
    /// No tokio runtime on this thread.
    pub(super) fn start(
        client: &Arc<NexarClient>,
        last: Rank,
        workers: &[Rank],
        vocab_size: usize,
    ) -> Result<Self> {
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|_| anyhow!("the swarm leader must start inside a tokio runtime"))?;
        let (events, rx) = mpsc::unbounded_channel();
        let mut tasks = vec![TaskGuard::new(runtime.spawn(route(rx, vocab_size)))];
        tasks.push(listen(
            &runtime,
            client,
            last,
            tags::LOGITS,
            &events,
            |buf| LogitsFrame::from_frame(buf).map(Event::Logits),
        ));
        for &rank in workers {
            tasks.push(listen(
                &runtime,
                client,
                rank,
                tags::STAGE_ERROR,
                &events,
                |buf| StageError::from_frame(buf).map(Event::Failed),
            ));
        }
        tasks.push(watch_failures(&runtime, client, workers.to_vec(), &events));
        Ok(Self {
            events,
            _tasks: tasks,
        })
    }

    /// Registers `request_id` as waiting for the reply to frame `frame_id`.
    /// Call it before the frame goes out, so its reply never arrives first.
    ///
    /// # Errors
    ///
    /// The dispatcher stopped. A duplicate id or a lost stage fails the
    /// returned receiver instead.
    pub(super) fn expect(
        &self,
        request_id: u64,
        frame_id: u64,
    ) -> Result<oneshot::Receiver<Reply>> {
        let (reply, rx) = oneshot::channel();
        let event = Event::Await {
            request_id,
            frame_id,
            reply,
        };
        if self.events.send(event).is_err() {
            bail!("the swarm leader's dispatcher stopped");
        }
        Ok(rx)
    }

    /// Stops waiting for `request_id`: for frame `frame_id` only, so a
    /// later frame of the id keeps its waiter, or for any frame when `None`.
    /// A reply that arrives later drops.
    pub(super) fn forget(&self, request_id: u64, frame_id: Option<u64>) {
        let _ = self.events.send(Event::Forget {
            request_id,
            frame_id,
        });
    }
}

/// Turns every frame from `src` on `tag` into an event, until `src` disconnects.
fn listen(
    runtime: &tokio::runtime::Handle,
    client: &Arc<NexarClient>,
    src: Rank,
    tag: u64,
    events: &mpsc::UnboundedSender<Event>,
    parse: fn(&[u8]) -> Result<Event>,
) -> TaskGuard {
    let (client, events) = (Arc::clone(client), events.clone());
    TaskGuard::new(runtime.spawn(async move {
        loop {
            let event = match client.recv_bytes_tagged(src, tag).await {
                Ok(buf) => parse(&buf).unwrap_or_else(|error| {
                    let (request_id, frame_id) = peek_reply_key(&buf);
                    Event::Malformed {
                        rank: src,
                        request_id,
                        frame_id,
                        error,
                    }
                }),
                Err(e) => Event::Lost {
                    rank: src,
                    reason: e.to_string(),
                },
            };
            let lost = matches!(event, Event::Lost { .. });
            if events.send(event).is_err() || lost {
                break;
            }
        }
    }))
}

/// Reports every stage rank nexar declares dead.
fn watch_failures(
    runtime: &tokio::runtime::Handle,
    client: &NexarClient,
    workers: Vec<Rank>,
    events: &mpsc::UnboundedSender<Event>,
) -> TaskGuard {
    let mut watch = client.failure_watch();
    let events = events.clone();
    TaskGuard::new(runtime.spawn(async move {
        while watch.changed().await.is_ok() {
            let dead = watch.borrow_and_update().clone();
            for rank in dead.into_iter().filter(|r| workers.contains(r)) {
                let reason = "nexar reports the peer dead".to_string();
                if events.send(Event::Lost { rank, reason }).is_err() {
                    return;
                }
            }
        }
    }))
}

/// One waiting request: the frame it waits for, and where its reply goes.
struct Waiter {
    frame_id: u64,
    reply: oneshot::Sender<Reply>,
}

/// Removes and returns the waiter of `request_id` when it waits for
/// `frame_id`, or for any frame when `frame_id` is `None`.
fn take(
    waiting: &mut HashMap<u64, Waiter>,
    request_id: u64,
    frame_id: Option<u64>,
) -> Option<oneshot::Sender<Reply>> {
    match waiting.entry(request_id) {
        Entry::Occupied(slot) if frame_id.is_none_or(|f| f == slot.get().frame_id) => {
            Some(slot.remove().reply)
        }
        _ => None,
    }
}

/// The dispatcher: owns the table of waiting requests.
async fn route(mut rx: mpsc::UnboundedReceiver<Event>, vocab_size: usize) {
    let mut waiting: HashMap<u64, Waiter> = HashMap::new();
    let mut lost: Option<String> = None;
    while let Some(event) = rx.recv().await {
        match event {
            Event::Await {
                request_id,
                frame_id,
                reply,
            } => {
                if let Some(reason) = &lost {
                    let _ = reply.send(Err(anyhow!("request {request_id}: {reason}")));
                    continue;
                }
                match waiting.entry(request_id) {
                    Entry::Occupied(_) => {
                        let _ = reply.send(Err(anyhow!(
                            "request {request_id} already has a frame in the pipeline. Wait \
                             for its reply before the next frame, and use a new id per request"
                        )));
                    }
                    Entry::Vacant(slot) => {
                        slot.insert(Waiter { frame_id, reply });
                    }
                }
            }
            Event::Forget {
                request_id,
                frame_id,
            } => {
                take(&mut waiting, request_id, frame_id);
            }
            Event::Logits(LogitsFrame {
                request_id,
                frame_id,
                logits,
            }) => {
                let Some(reply) = take(&mut waiting, request_id, Some(frame_id)) else {
                    tracing::debug!(
                        request_id,
                        frame_id,
                        "dropping logits of a frame nobody waits for"
                    );
                    continue;
                };
                let result = if logits.len() == vocab_size {
                    Ok(logits)
                } else {
                    Err(anyhow!(
                        "request {request_id}: the last stage sent {} logits, and the \
                         vocabulary holds {vocab_size}",
                        logits.len()
                    ))
                };
                let _ = reply.send(result);
            }
            Event::Failed(error) => {
                match take(&mut waiting, error.request_id, Some(error.frame_id)) {
                    Some(reply) => {
                        let _ = reply.send(Err(anyhow::Error::new(error)));
                    }
                    None => tracing::debug!("dropping a stage error nobody waits for: {error}"),
                }
            }
            Event::Malformed {
                rank,
                request_id,
                frame_id,
                error,
            } => {
                // A reply with no readable frame id fails its request: it can
                // be the reply the request waits for.
                let reply = request_id.and_then(|id| take(&mut waiting, id, frame_id));
                match reply {
                    Some(reply) => {
                        let _ = reply.send(Err(
                            error.context(format!("rank {rank} sent a malformed reply"))
                        ));
                    }
                    None => tracing::warn!(rank, "dropping a malformed reply: {error:#}"),
                }
            }
            Event::Lost { rank, reason } => {
                let reason = format!("the pipeline lost stage rank {rank}: {reason}");
                for (request_id, waiter) in waiting.drain() {
                    let _ = waiter
                        .reply
                        .send(Err(anyhow!("request {request_id}: {reason}")));
                }
                lost.get_or_insert(reason);
            }
        }
    }
}
