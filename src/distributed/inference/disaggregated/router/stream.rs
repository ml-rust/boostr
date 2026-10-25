//! The token stream of one routed request.

use anyhow::{Result, anyhow};
use nexar::Rank;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedReceiver;

use super::super::protocol::{FinishReason, TokenEvent};
use super::super::wire::spawn_cancel;
use super::dispatch::Shared;
use super::pending::StreamItem;

/// Every token of a finished request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generation {
    /// Sampled tokens in order. The last one is the token that finished the stream.
    pub tokens: Vec<u32>,
    pub finish: FinishReason,
}

/// The events of one request, in order, as the workers produce them.
///
/// [`Self::next`] yields `Ok(token)` per token. The last token carries
/// `finish: Some(_)`. An error item also ends the stream: a
/// [`super::WorkerError`] when a worker failed the request, else a
/// transport error. After the last item, `next` returns `None`.
///
/// Dropping the stream before its end cancels the request: the router stops
/// tracking it and tells both workers to stop and free its state.
pub struct TokenStream {
    pub(crate) request_id: u64,
    pub(crate) rx: UnboundedReceiver<StreamItem>,
    pub(crate) shared: Arc<Shared>,
    pub(crate) prefill_rank: Rank,
    pub(crate) decode_rank: Rank,
    pub(crate) ended: bool,
}

impl TokenStream {
    /// The router's id for this request.
    pub fn request_id(&self) -> u64 {
        self.request_id
    }

    /// The decode worker the router chose.
    pub fn decode_rank(&self) -> Rank {
        self.decode_rank
    }

    /// The next event, or `None` after the stream ended.
    pub async fn next(&mut self) -> Option<Result<TokenEvent>> {
        if self.ended {
            return None;
        }
        let item = match self.rx.recv().await {
            Some(item) => item,
            None => Err(anyhow!(
                "request {} stream closed before a finish reason: the router stopped",
                self.request_id
            )),
        };
        self.ended = item.as_ref().map_or(true, |t| t.finish.is_some());
        Some(item)
    }

    /// Reads the stream to its end.
    ///
    /// # Errors
    ///
    /// The stream's error item.
    pub async fn collect(mut self) -> Result<Generation> {
        let mut tokens = Vec::new();
        while let Some(item) = self.next().await {
            let token = item?;
            tokens.push(token.token);
            if let Some(finish) = token.finish {
                return Ok(Generation { tokens, finish });
            }
        }
        Err(anyhow!(
            "request {} stream ended without a finish reason",
            self.request_id
        ))
    }
}

impl Drop for TokenStream {
    fn drop(&mut self) {
        if self.ended {
            return;
        }
        let tracked = self.shared.forget(self.request_id);
        if !tracked {
            return;
        }
        spawn_cancel(
            &self.shared.client,
            vec![self.prefill_rank, self.decode_rank],
            self.request_id,
        );
    }
}
