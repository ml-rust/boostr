//! Prefill worker: prompt forward, first-token sampling and the KV handoff.

use anyhow::{Result, anyhow, bail};
use nexar::{NexarClient, Rank};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

use super::cancel::CancelBook;
use super::forward::PrefillFn;
use super::protocol::{Event, KvHandoff, PrefillRequest, Stage, TokenEvent, peek_request_id, tags};
use super::sampler::RequestSampler;
use super::wire::{
    TaskGuard, run_blocking, send_error, send_event, send_frame, spawn_cancel_listener,
    spawn_listener,
};
use crate::ops::traits::inference::SamplingOps;

/// What prefill produced for one request.
#[derive(Debug)]
pub(crate) struct PrefillResult {
    /// The first token, with its finish reason.
    pub first: TokenEvent,
    /// The decode state. `None` when the first token ends the stream.
    pub handoff: Option<KvHandoff>,
}

/// Runs prefill for `req` and samples its first token.
///
/// # Errors
///
/// Invalid params, an empty prompt, a [`PrefillFn`] error, a sampler error,
/// or a KV cache above `max_kv_bytes`.
pub(crate) fn prefill_request<R>(
    prefill_fn: &PrefillFn<R>,
    client: &R::Client,
    req: PrefillRequest,
    max_kv_bytes: usize,
) -> Result<PrefillResult>
where
    R: Runtime<DType = DType>,
    R::Client: SamplingOps<R>,
{
    req.params.check()?;
    if req.prompt.is_empty() {
        bail!("the prompt has no tokens");
    }
    let out = prefill_fn(&req.prompt).map_err(|e| e.context("prefill forward failed"))?;
    let mut sampler = RequestSampler::start(req.params, &req.prompt);
    let first = sampler.sample(client, &out.logits)?;
    let handoff = if first.finish.is_some() {
        None
    } else {
        if out.kv_cache.len() > max_kv_bytes {
            bail!(
                "the KV cache is {} bytes, above the {max_kv_bytes}-byte transfer limit: \
                 raise max_kv_transfer_bytes or shorten the prompt",
                out.kv_cache.len()
            );
        }
        Some(KvHandoff {
            request_id: req.request_id,
            params: sampler.params().clone(),
            generated: sampler.generated(),
            context: sampler.context().to_vec(),
            kv_cache: out.kv_cache,
        })
    };
    Ok(PrefillResult { first, handoff })
}

/// Prefill worker: runs the prompt forward, samples the first token with the
/// request's sampler, streams that token to the router and hands the KV cache
/// to the decode worker the router chose.
///
/// A request whose first token ends it (EOS, stop id, or `max_new_tokens == 1`)
/// never reaches a decode worker. Every error becomes a
/// [`Stage::Prefill`] error event for its request, and the loop goes on.
pub struct PrefillWorker<R: Runtime> {
    client: Arc<NexarClient>,
    rank: Rank,
    router_rank: Rank,
    max_kv_transfer_bytes: usize,
    sampler_client: R::Client,
    prefill_fn: Arc<PrefillFn<R>>,
}

impl<R> PrefillWorker<R>
where
    R: Runtime<DType = DType>,
    R::Client: SamplingOps<R> + 'static,
{
    /// A prefill worker.
    ///
    /// - `sampler_client`: the runtime client the first token samples on. It
    ///   must run on the device `prefill_fn` returns logits on.
    /// - `prefill_fn`: the model forward over a prompt. See [`PrefillFn`].
    pub fn new(
        client: Arc<NexarClient>,
        rank: Rank,
        router_rank: Rank,
        max_kv_transfer_bytes: usize,
        sampler_client: R::Client,
        prefill_fn: PrefillFn<R>,
    ) -> Self {
        Self {
            client,
            rank,
            router_rank,
            max_kv_transfer_bytes,
            sampler_client,
            prefill_fn: Arc::new(prefill_fn),
        }
    }

    pub fn rank(&self) -> Rank {
        self.rank
    }

    /// Serves prefill requests one at a time until the router disconnects.
    ///
    /// # Errors
    ///
    /// A failed send to the router. A failed handoff to a decode worker is an
    /// error event for that request, never a loop error.
    pub async fn run_loop(&self) -> Result<()> {
        tracing::info!(rank = self.rank, "prefill worker loop starting");
        let book = Arc::new(CancelBook::default());
        let _cancels = spawn_cancel_listener(
            Arc::clone(&self.client),
            self.router_rank,
            Arc::clone(&book),
        );
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let _requests: TaskGuard = spawn_listener(
            Arc::clone(&self.client),
            self.router_rank,
            tags::PREFILL_REQUEST,
            tx,
            |buf: &[u8]| PrefillRequest::from_bytes(buf).map_err(|e| (peek_request_id(buf), e)),
        );
        while let Some(frame) = rx.recv().await {
            match frame {
                Ok(req) => self.serve(req, &book).await?,
                Err((Some(id), e)) => self.fail(id, 0, e).await?,
                Err((None, e)) => tracing::warn!(
                    rank = self.rank,
                    "dropping a malformed prefill frame: {e:#}"
                ),
            }
        }
        tracing::info!(rank = self.rank, "prefill worker loop ended");
        Ok(())
    }

    async fn serve(&self, req: PrefillRequest, book: &CancelBook) -> Result<()> {
        let id = req.request_id;
        let decode_rank = req.decode_rank;
        if book.is_cancelled(id) {
            book.finish(id);
            return Ok(());
        }
        let prefill_fn = Arc::clone(&self.prefill_fn);
        let client = self.sampler_client.clone();
        let limit = self.max_kv_transfer_bytes;
        let result = run_blocking(move || prefill_request(&prefill_fn, &client, req, limit))
            .await
            .and_then(|r| r);
        let cancelled = book.is_cancelled(id);
        book.finish(id);
        let out = match result {
            Ok(_) if cancelled => return Ok(()),
            Ok(out) => out,
            Err(e) => return self.fail(id, 0, e).await,
        };
        send_event(&self.client, self.router_rank, &Event::token(id, out.first)).await?;
        if let Some(handoff) = out.handoff {
            let sent = match handoff.to_bytes() {
                Ok(frame) => send_frame(&self.client, decode_rank, tags::KV_HANDOFF, &frame).await,
                Err(e) => Err(e),
            };
            if let Err(e) = sent {
                let e = anyhow!("KV handoff to decode rank {decode_rank} failed: {e:#}");
                return self.fail(id, out.first.index + 1, e).await;
            }
        }
        Ok(())
    }

    async fn fail(&self, id: u64, seq: u32, e: anyhow::Error) -> Result<()> {
        send_error(
            &self.client,
            self.router_rank,
            self.rank,
            Stage::Prefill,
            id,
            seq,
            e,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::super::protocol::{FinishReason, GenerationParams};
    use super::super::test_harness::mock_prefill;
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn request(params: GenerationParams, prompt: Vec<u32>) -> PrefillRequest {
        PrefillRequest {
            request_id: 11,
            decode_rank: 2,
            params,
            prompt,
        }
    }

    fn client() -> CpuClient {
        CpuClient::new(CpuDevice::new())
    }

    #[test]
    fn prefill_returns_the_sampled_first_token_and_the_handoff() {
        let f = mock_prefill(None);
        let out = prefill_request::<CpuRuntime>(
            &f,
            &client(),
            request(GenerationParams::greedy(8), vec![3, 4, 5]),
            usize::MAX,
        )
        .unwrap();
        assert_eq!(
            out.first,
            TokenEvent {
                index: 0,
                token: 6,
                finish: None
            }
        );
        let h = out.handoff.expect("decode continues");
        assert_eq!(h.request_id, 11);
        assert_eq!(h.context, vec![3, 4, 5, 6]);
        assert_eq!(h.generated, 1);
        assert!(
            h.params.rule.seed.is_some(),
            "the handoff carries the resolved key"
        );
        assert_eq!(h.kv_cache.len(), 12, "the mock cache holds the prompt");
    }

    #[test]
    fn a_finishing_first_token_skips_the_handoff() {
        let f = mock_prefill(None);
        let mut eos = GenerationParams::greedy(8);
        eos.eos_tokens = vec![6];
        let out = prefill_request::<CpuRuntime>(&f, &client(), request(eos, vec![5]), usize::MAX)
            .unwrap();
        assert_eq!(out.first.finish, Some(FinishReason::Eos));
        assert!(out.handoff.is_none());
        let out = prefill_request::<CpuRuntime>(
            &f,
            &client(),
            request(GenerationParams::greedy(1), vec![5]),
            usize::MAX,
        )
        .unwrap();
        assert_eq!(out.first.finish, Some(FinishReason::Length));
        assert!(out.handoff.is_none());
    }

    #[test]
    fn prefill_errors_are_returned() {
        let failing = mock_prefill(Some("out of device memory"));
        let err = prefill_request::<CpuRuntime>(
            &failing,
            &client(),
            request(GenerationParams::greedy(4), vec![1]),
            usize::MAX,
        )
        .unwrap_err();
        assert!(
            format!("{err:#}").contains("out of device memory"),
            "{err:#}"
        );
        let ok = mock_prefill(None);
        let err = prefill_request::<CpuRuntime>(
            &ok,
            &client(),
            request(GenerationParams::greedy(4), vec![1, 2]),
            4,
        )
        .unwrap_err();
        assert!(err.to_string().contains("transfer limit"), "{err}");
        let err = prefill_request::<CpuRuntime>(
            &ok,
            &client(),
            request(GenerationParams::greedy(4), vec![]),
            usize::MAX,
        )
        .unwrap_err();
        assert!(err.to_string().contains("no tokens"), "{err}");
        let err = prefill_request::<CpuRuntime>(
            &ok,
            &client(),
            request(GenerationParams::greedy(0), vec![1]),
            usize::MAX,
        )
        .unwrap_err();
        assert!(err.to_string().contains("max_new_tokens"), "{err}");
    }
}
