//! Mock forward closures and an in-process cluster for the disaggregated tests.
//!
//! The mock model has a vocab of [`VOCAB`]. Its logits peak at `(token + 1) % VOCAB`
//! for the last fed token. Its "KV cache" is the fed token ids as `u32` LE,
//! and each step checks that `position` equals the cache length.

use anyhow::{Result, anyhow, bail};
use nexar::NexarClient;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use super::config::{DEFAULT_HANDOFF_TIMEOUT, DisaggConfig};
use super::decode_worker::DecodeWorker;
use super::forward::{DecodeOpenFn, PrefillFn, PrefillOutput};
use super::prefill_worker::PrefillWorker;
use super::router::DisaggRouter;
use crate::distributed::inference::transport::cpu_adapter;

pub(crate) const VOCAB: usize = 16;

/// `[1, seq_len, VOCAB]` logits whose last row peaks at `peak`, or is flat.
fn logits(seq_len: usize, peak: Option<u32>) -> Result<Tensor<CpuRuntime>> {
    let mut data = vec![0.0f32; seq_len * VOCAB];
    if let Some(p) = peak {
        data[(seq_len - 1) * VOCAB + p as usize] = 10.0;
    }
    Ok(Tensor::<CpuRuntime>::from_slice(
        &data,
        &[1, seq_len, VOCAB],
        &CpuDevice::new(),
    )?)
}

pub(crate) fn mock_kv(tokens: &[u32]) -> Vec<u8> {
    tokens.iter().flat_map(|t| t.to_le_bytes()).collect()
}

fn next_of(token: u32) -> u32 {
    (token + 1) % VOCAB as u32
}

/// A prefill closure. `fail` makes every call return that error.
pub(crate) fn mock_prefill(fail: Option<&'static str>) -> PrefillFn<CpuRuntime> {
    Box::new(move |prompt: &[u32]| {
        if let Some(msg) = fail {
            bail!("{msg}");
        }
        let last = *prompt.last().ok_or_else(|| anyhow!("empty prompt"))?;
        Ok(PrefillOutput {
            logits: logits(prompt.len(), Some(next_of(last)))?,
            kv_cache: mock_kv(prompt),
        })
    })
}

/// Decode closure config.
#[derive(Default, Clone)]
pub(crate) struct MockDecode {
    /// Flat logits, so a sampled rule draws from the whole vocab.
    pub flat: bool,
    /// The 1-based step of each request that returns an error.
    pub fail_at_step: Option<u32>,
    /// A sleep before each step.
    pub delay: Option<Duration>,
    /// Each step waits for one message on this gate.
    pub gate: Option<Arc<Mutex<mpsc::Receiver<()>>>>,
    /// Steps run across all requests.
    pub steps: Arc<AtomicU32>,
}

impl MockDecode {
    pub(crate) fn gated() -> (Self, mpsc::Sender<()>) {
        let (tx, rx) = mpsc::channel();
        let mock = Self {
            gate: Some(Arc::new(Mutex::new(rx))),
            ..Self::default()
        };
        (mock, tx)
    }

    pub(crate) fn open_fn(&self) -> DecodeOpenFn<CpuRuntime> {
        let opts = self.clone();
        Box::new(move |kv: &[u8]| {
            if !kv.len().is_multiple_of(4) {
                bail!("mock cache of {} bytes is not whole u32 ids", kv.len());
            }
            let mut cache: Vec<u32> = kv
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| u32::from_le_bytes(c))
                .collect();
            let opts = opts.clone();
            let mut step_no = 0u32;
            Ok(Box::new(move |token: u32, position: u32| {
                step_no += 1;
                opts.steps.fetch_add(1, Ordering::Relaxed);
                if let Some(gate) = &opts.gate {
                    let rx = gate.lock().map_err(|_| anyhow!("gate poisoned"))?;
                    rx.recv().map_err(|_| anyhow!("gate closed"))?;
                }
                if let Some(d) = opts.delay {
                    std::thread::sleep(d);
                }
                if opts.fail_at_step == Some(step_no) {
                    bail!("mock step failure at step {step_no}");
                }
                if position as usize != cache.len() {
                    bail!("position {position} but the cache holds {}", cache.len());
                }
                cache.push(token);
                logits(1, (!opts.flat).then(|| next_of(token)))
            }) as _)
        })
    }
}

/// `n` connected clients in this process, sorted by rank.
pub(crate) async fn local_clients(n: u32) -> Vec<NexarClient> {
    // nexar's QUIC needs a process-wide rustls provider. A second install is a no-op error.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let mut clients = NexarClient::bootstrap_local(n, cpu_adapter())
        .await
        .expect("bootstrap a local cluster");
    clients.sort_by_key(|c| c.rank());
    clients
}

/// The layout of every test cluster: router 0, prefill 1, decode 2.
pub(crate) fn config(handoff_timeout: Duration) -> DisaggConfig {
    DisaggConfig {
        prefill_workers: vec![1],
        decode_workers: vec![2],
        router_rank: 0,
        max_kv_transfer_bytes: 1 << 20,
        handoff_timeout,
    }
}

/// A prefill worker on rank 1 over the mock model.
pub(crate) fn prefill_worker(
    client: Arc<NexarClient>,
    fail: Option<&'static str>,
    handoff_timeout: Duration,
) -> PrefillWorker<CpuRuntime> {
    let sampler = CpuClient::new(CpuDevice::new());
    PrefillWorker::new(client, 1, 0, 1 << 20, sampler, mock_prefill(fail))
        .with_handoff_timeout(handoff_timeout)
}

/// A decode worker on rank 2 over the mock model.
pub(crate) fn decode_worker(
    client: Arc<NexarClient>,
    decode: &MockDecode,
) -> DecodeWorker<CpuRuntime> {
    let sampler = CpuClient::new(CpuDevice::new());
    DecodeWorker::new(client, 2, 0, vec![1], 1 << 20, sampler, decode.open_fn())
}

/// A router on rank 0 and the bare clients of ranks 1 and 2. The test runs
/// a worker, or a fake one, on each.
pub(crate) struct Parts {
    pub router: DisaggRouter,
    pub prefill: Arc<NexarClient>,
    pub decode: Arc<NexarClient>,
}

impl Parts {
    pub(crate) async fn start(handoff_timeout: Duration) -> Self {
        let mut clients = local_clients(3).await.into_iter().map(Arc::new);
        let (router, prefill, decode) = (
            clients.next().expect("rank 0"),
            clients.next().expect("rank 1"),
            clients.next().expect("rank 2"),
        );
        Self {
            router: DisaggRouter::new(router, config(handoff_timeout)).expect("router"),
            prefill,
            decode,
        }
    }
}

/// Router on rank 0, one prefill worker on rank 1, one decode worker on rank 2.
pub(crate) struct Cluster {
    pub router: DisaggRouter,
    _workers: Vec<tokio::task::JoinHandle<Result<()>>>,
}

impl Cluster {
    pub(crate) async fn start(prefill_fail: Option<&'static str>, decode: MockDecode) -> Self {
        let parts = Parts::start(DEFAULT_HANDOFF_TIMEOUT).await;
        let p = prefill_worker(parts.prefill, prefill_fail, DEFAULT_HANDOFF_TIMEOUT);
        let d = decode_worker(parts.decode, &decode);
        let workers = vec![
            tokio::spawn(async move { p.run_loop().await }),
            tokio::spawn(async move { d.run_loop().await }),
        ];
        Self {
            router: parts.router,
            _workers: workers,
        }
    }
}

impl Cluster {
    /// The worker run loops. The cluster no longer aborts them on drop.
    pub(crate) fn take_workers(&mut self) -> Vec<tokio::task::JoinHandle<Result<()>>> {
        std::mem::take(&mut self._workers)
    }
}

impl Drop for Cluster {
    fn drop(&mut self) {
        for w in &self._workers {
            w.abort();
        }
    }
}
