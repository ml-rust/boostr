//! Decode worker: resumes a request from its KV handoff and streams tokens.

use anyhow::{Result, anyhow};
use nexar::{NexarClient, Rank};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

use super::cancel::CancelBook;
use super::forward::{DecodeOpenFn, DecodeStepFn};
use super::protocol::{Event, KvHandoff, Stage, TokenEvent, peek_request_id, tags};
use super::sampler::RequestSampler;
use super::wire::{
    TaskGuard, run_blocking, send_error, send_event, spawn_cancel_listener, spawn_listener,
};
use crate::ops::traits::inference::SamplingOps;

/// One request in decode: its step closure and its sampler.
pub(crate) struct ActiveDecode<R: Runtime> {
    step: DecodeStepFn<R>,
    sampler: RequestSampler,
}

impl<R> ActiveDecode<R>
where
    R: Runtime<DType = DType>,
    R::Client: SamplingOps<R>,
{
    /// Opens `handoff` through `open`.
    ///
    /// # Errors
    ///
    /// The handoff has no resolved seed, or `open` fails.
    pub(crate) fn open(open: &DecodeOpenFn<R>, handoff: KvHandoff) -> Result<Self> {
        let sampler = RequestSampler::resume(handoff.params, handoff.context, handoff.generated)?;
        let step = open(&handoff.kv_cache).map_err(|e| e.context("opening the KV cache failed"))?;
        Ok(Self { step, sampler })
    }

    /// The index of the token the next [`Self::advance`] samples.
    pub(crate) fn next_index(&self) -> u32 {
        self.sampler.generated()
    }

    /// Feeds the last token and samples the next one.
    pub(crate) fn advance(&mut self, client: &R::Client) -> Result<TokenEvent> {
        let token = self
            .sampler
            .last_token()
            .ok_or_else(|| anyhow!("the decode context is empty"))?;
        let position = self.sampler.last_position();
        let logits = (self.step)(token, position)
            .map_err(|e| e.context(format!("decode step at position {position} failed")))?;
        self.sampler.sample(client, &logits)
    }
}

/// Decode worker: takes KV handoffs from any prefill worker, then runs one
/// request at a time to its finish.
///
/// Each step feeds the last sampled token, samples the next with the
/// request's seeded sampler and streams it to the router. The stream ends on
/// an EOS id, a stop id or `max_new_tokens`. A cancel from the router ends
/// the request between steps and frees its state without another event.
/// Every error becomes a [`Stage::Decode`] error event, and the loop goes on.
pub struct DecodeWorker<R: Runtime> {
    client: Arc<NexarClient>,
    rank: Rank,
    router_rank: Rank,
    prefill_workers: Vec<Rank>,
    max_kv_transfer_bytes: usize,
    sampler_client: R::Client,
    open_fn: Arc<DecodeOpenFn<R>>,
}

impl<R> DecodeWorker<R>
where
    R: Runtime<DType = DType>,
    R::Client: SamplingOps<R> + 'static,
{
    /// A decode worker.
    ///
    /// - `prefill_workers`: every rank that can send this worker a handoff.
    /// - `sampler_client`: the runtime client each token samples on. It must
    ///   run on the device the step closures return logits on.
    /// - `open_fn`: opens one request from its serialized cache. See [`DecodeOpenFn`].
    pub fn new(
        client: Arc<NexarClient>,
        rank: Rank,
        router_rank: Rank,
        prefill_workers: Vec<Rank>,
        max_kv_transfer_bytes: usize,
        sampler_client: R::Client,
        open_fn: DecodeOpenFn<R>,
    ) -> Self {
        Self {
            client,
            rank,
            router_rank,
            prefill_workers,
            max_kv_transfer_bytes,
            sampler_client,
            open_fn: Arc::new(open_fn),
        }
    }

    pub fn rank(&self) -> Rank {
        self.rank
    }

    /// Serves handoffs until every prefill worker disconnects.
    ///
    /// # Errors
    ///
    /// A failed send to the router.
    pub async fn run_loop(&self) -> Result<()> {
        tracing::info!(rank = self.rank, "decode worker loop starting");
        let book = Arc::new(CancelBook::default());
        let _cancels = spawn_cancel_listener(
            Arc::clone(&self.client),
            self.router_rank,
            Arc::clone(&book),
        );
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let _listeners: Vec<TaskGuard> = self
            .prefill_workers
            .iter()
            .map(|&src| {
                spawn_listener(
                    Arc::clone(&self.client),
                    src,
                    tags::KV_HANDOFF,
                    tx.clone(),
                    |buf: &[u8]| KvHandoff::from_bytes(buf).map_err(|e| (peek_request_id(buf), e)),
                )
            })
            .collect();
        drop(tx);
        while let Some(frame) = rx.recv().await {
            match frame {
                Ok(handoff) => self.serve(handoff, &book).await?,
                // Prefill always hands off after its one token, so decode's first seq is 1.
                Err((Some(id), e)) => self.fail(id, 1, e).await?,
                Err((None, e)) => {
                    tracing::warn!(rank = self.rank, "dropping a malformed KV handoff: {e:#}")
                }
            }
        }
        tracing::info!(rank = self.rank, "decode worker loop ended");
        Ok(())
    }

    async fn serve(&self, handoff: KvHandoff, book: &CancelBook) -> Result<()> {
        let id = handoff.request_id;
        let result = self.decode(handoff, book).await;
        book.finish(id);
        result
    }

    async fn decode(&self, handoff: KvHandoff, book: &CancelBook) -> Result<()> {
        let id = handoff.request_id;
        let first_seq = handoff.generated;
        if book.is_cancelled(id) {
            return Ok(());
        }
        if handoff.kv_cache.len() > self.max_kv_transfer_bytes {
            let e = anyhow!(
                "the KV cache is {} bytes, above this decode worker's {}-byte limit",
                handoff.kv_cache.len(),
                self.max_kv_transfer_bytes
            );
            return self.fail(id, first_seq, e).await;
        }
        let open_fn = Arc::clone(&self.open_fn);
        let opened = run_blocking(move || ActiveDecode::open(&open_fn, handoff)).await;
        let mut active = match opened.and_then(|r| r) {
            Ok(a) => a,
            Err(e) => return self.fail(id, first_seq, e).await,
        };
        loop {
            if book.is_cancelled(id) {
                tracing::debug!(rank = self.rank, request_id = id, "decode cancelled");
                return Ok(());
            }
            let seq = active.next_index();
            let client = self.sampler_client.clone();
            let stepped = run_blocking(move || {
                let r = active.advance(&client);
                (active, r)
            })
            .await;
            let (back, result) = match stepped {
                Ok(pair) => pair,
                Err(e) => return self.fail(id, seq, e).await,
            };
            active = back;
            match result {
                Ok(token) => {
                    send_event(&self.client, self.router_rank, &Event::token(id, token)).await?;
                    if token.finish.is_some() {
                        return Ok(());
                    }
                }
                Err(e) => return self.fail(id, seq, e).await,
            }
        }
    }

    async fn fail(&self, id: u64, seq: u32, e: anyhow::Error) -> Result<()> {
        send_error(
            &self.client,
            self.router_rank,
            self.rank,
            Stage::Decode,
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
    use super::super::test_harness::{MockDecode, mock_kv};
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn handoff(params: GenerationParams) -> KvHandoff {
        let mut params = params;
        params.rule.seed = Some(1);
        KvHandoff {
            request_id: 4,
            params,
            generated: 1,
            context: vec![3, 4, 5, 6],
            kv_cache: mock_kv(&[3, 4, 5]),
        }
    }

    fn run(active: &mut ActiveDecode<CpuRuntime>) -> (Vec<u32>, Option<FinishReason>) {
        let client = CpuClient::new(CpuDevice::new());
        let mut tokens = Vec::new();
        loop {
            let ev = active.advance(&client).unwrap();
            assert_eq!(ev.index as usize, tokens.len() + 1);
            tokens.push(ev.token);
            if ev.finish.is_some() {
                return (tokens, ev.finish);
            }
        }
    }

    #[test]
    fn decode_stops_at_eos_with_its_reason() {
        let mut p = GenerationParams::greedy(50);
        p.eos_tokens = vec![9];
        let open = MockDecode::default().open_fn();
        let mut a = ActiveDecode::open(&open, handoff(p)).unwrap();
        assert_eq!(a.next_index(), 1);
        assert_eq!(run(&mut a), (vec![7, 8, 9], Some(FinishReason::Eos)));
    }

    #[test]
    fn decode_stops_at_a_stop_id_and_at_length() {
        let mut p = GenerationParams::greedy(50);
        p.stop_tokens = vec![8];
        let open = MockDecode::default().open_fn();
        let mut a = ActiveDecode::open(&open, handoff(p)).unwrap();
        assert_eq!(run(&mut a), (vec![7, 8], Some(FinishReason::Stop)));
        let mut a = ActiveDecode::open(&open, handoff(GenerationParams::greedy(4))).unwrap();
        assert_eq!(run(&mut a), (vec![7, 8, 9], Some(FinishReason::Length)));
    }

    #[test]
    fn seeded_sampled_decode_is_deterministic() {
        let mut p = GenerationParams::greedy(12);
        p.rule.filter.temperature = 1.0;
        let open = MockDecode {
            flat: true,
            ..MockDecode::default()
        }
        .open_fn();
        let a = run(&mut ActiveDecode::open(&open, handoff(p.clone())).unwrap());
        let b = run(&mut ActiveDecode::open(&open, handoff(p)).unwrap());
        assert_eq!(a, b);
        assert_eq!(a.0.len(), 11);
    }

    #[test]
    fn step_and_open_errors_surface() {
        let open = MockDecode {
            fail_at_step: Some(2),
            ..MockDecode::default()
        }
        .open_fn();
        let mut a = ActiveDecode::open(&open, handoff(GenerationParams::greedy(9))).unwrap();
        let client = CpuClient::new(CpuDevice::new());
        a.advance(&client).unwrap();
        let err = a.advance(&client).unwrap_err();
        assert!(format!("{err:#}").contains("mock step failure"), "{err:#}");
        let mut bad = handoff(GenerationParams::greedy(9));
        bad.kv_cache = vec![1, 2, 3];
        let err = ActiveDecode::open(&open, bad).err().expect("bad cache");
        assert!(
            format!("{err:#}").contains("opening the KV cache"),
            "{err:#}"
        );
    }
}
