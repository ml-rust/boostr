//! Swarm pipeline serving over in-process nexar ranks: the leader on rank 0
//! runs stage 0, and each worker rank serves a mock stage.
//!
//! The mock checks the request protocol: a prefill opens a request at
//! position 0, and a decode continues at the position the request reached.
//! The last stage returns `[request_id, position, sum(hidden), rank]`.

use anyhow::{Result, bail};
use boostr::distributed::inference::leader::SwarmLeader;
use boostr::distributed::inference::stage::{Stage, StageOutput};
use boostr::distributed::inference::transport::{
    Activation, GenRequestHeader, LayerAssignment, StageError, StageOp, StagePayload, cpu_adapter,
};
use boostr::distributed::inference::worker::SwarmWorker;
use nexar::{NexarClient, Rank};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::task::{JoinHandle, JoinSet};

const WAIT: Duration = Duration::from_secs(10);
const VOCAB: usize = 4;
const LAYERS: usize = 6;
/// The request id the failing rank refuses.
const FAIL_ID: u64 = 666;

/// Rank → request ids its stage holds.
type Held = Arc<Mutex<BTreeMap<Rank, BTreeSet<u64>>>>;

struct Mock {
    rank: Rank,
    last: bool,
    fails: bool,
    delay: Duration,
    open: HashMap<u64, u32>,
    held: Held,
}

impl Mock {
    fn publish(&self) {
        let ids = self.open.keys().copied().collect();
        self.held.lock().unwrap().insert(self.rank, ids);
    }
}

impl Stage for Mock {
    fn forward(&mut self, h: &GenRequestHeader, payload: StagePayload) -> Result<StageOutput> {
        if self.fails && h.request_id == FAIL_ID {
            bail!("boom");
        }
        let reached = self.open.get(&h.request_id).copied();
        match (h.op, reached) {
            (StageOp::Prefill, None) if h.position == 0 => {}
            (StageOp::Decode, Some(p)) if p == h.position => {}
            _ => bail!("{:?} at {} on a request at {reached:?}", h.op, h.position),
        }
        let StagePayload::Activation(a) = payload else {
            bail!("a worker stage takes an activation");
        };
        self.open.insert(h.request_id, h.position + h.seq_len);
        self.publish();
        std::thread::sleep(self.delay);
        let hidden: Vec<f32> = a.hidden.iter().map(|v| v + 1.0).collect();
        Ok(if self.last {
            let sum = hidden.iter().sum();
            StageOutput::Logits(vec![
                h.request_id as f32,
                h.position as f32,
                sum,
                self.rank as f32,
            ])
        } else {
            StageOutput::Activation(Activation { hidden, mlp: None })
        })
    }

    fn release(&mut self, request_id: u64) {
        self.open.remove(&request_id);
        self.publish();
    }

    fn release_all(&mut self) {
        self.open.clear();
        self.publish();
    }
}

struct Swarm {
    leader: SwarmLeader,
    workers: Vec<JoinHandle<Result<()>>>,
    held: Held,
}

struct Options {
    world: u32,
    fail_rank: Option<Rank>,
    delay: Duration,
}

impl Swarm {
    async fn start(o: Options) -> Self {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let mut clients = NexarClient::bootstrap_local(o.world, cpu_adapter())
            .await
            .unwrap();
        clients.sort_by_key(|c| c.rank());
        let mut clients = clients.into_iter().map(Arc::new);
        let leader = SwarmLeader::by_rank(clients.next().unwrap(), LAYERS, VOCAB).unwrap();
        let held = Held::default();
        let workers = clients
            .map(|client| {
                let rank = client.rank();
                let s = &leader.schedule().stages()[rank as usize];
                let assignment = LayerAssignment {
                    start_layer: s.start_layer as u32,
                    end_layer: s.end_layer as u32,
                    has_embedding: s.has_embedding,
                    has_lm_head: s.has_lm_head,
                };
                let next = (rank + 1 < o.world).then_some(rank + 1);
                let stage = Box::new(Mock {
                    rank,
                    last: next.is_none(),
                    fails: o.fail_rank == Some(rank),
                    delay: o.delay,
                    open: HashMap::new(),
                    held: Arc::clone(&held),
                });
                let mut worker =
                    SwarmWorker::new(client, rank, assignment, Some(rank - 1), next, 0, stage);
                tokio::spawn(async move {
                    worker.run_compute_loop().await?;
                    worker.shutdown(WAIT).await
                })
            })
            .collect();
        Self {
            leader,
            workers,
            held,
        }
    }

    fn held(&self, rank: Rank) -> BTreeSet<u64> {
        self.held
            .lock()
            .unwrap()
            .get(&rank)
            .cloned()
            .unwrap_or_default()
    }

    async fn stop(self) {
        self.leader.shutdown(WAIT).await.unwrap();
        for w in self.workers {
            tokio::time::timeout(WAIT, w)
                .await
                .expect("the worker loop ends after the shutdown notice")
                .unwrap()
                .unwrap();
        }
    }
}

fn frame(
    request_id: u64,
    op: StageOp,
    position: u32,
    hidden: &[f32],
) -> (GenRequestHeader, StagePayload) {
    let header = GenRequestHeader {
        request_id,
        op,
        position,
        seq_len: hidden.len() as u32,
    };
    let payload = StagePayload::Activation(Activation {
        hidden: hidden.to_vec(),
        mlp: None,
    });
    (header, payload)
}

async fn forward(leader: &SwarmLeader, f: (GenRequestHeader, StagePayload)) -> Result<Vec<f32>> {
    leader.pipeline_forward(f.0, &f.1).await
}

fn opts(world: u32) -> Options {
    Options {
        world,
        fail_rank: None,
        delay: Duration::ZERO,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_three_stage_pipeline_advances_positions_and_releases() {
    let s = Swarm::start(opts(3)).await;
    let l = &s.leader;
    // Two worker stages add 1.0 each: [1, 2] becomes [3, 4].
    let row = forward(l, frame(1, StageOp::Prefill, 0, &[1.0, 2.0]))
        .await
        .unwrap();
    assert_eq!(row, [1.0, 0.0, 7.0, 2.0]);
    let row = forward(l, frame(1, StageOp::Decode, 2, &[5.0]))
        .await
        .unwrap();
    assert_eq!(row, [1.0, 2.0, 7.0, 2.0]);
    let row = forward(l, frame(1, StageOp::Decode, 3, &[0.0]))
        .await
        .unwrap();
    assert_eq!(row, [1.0, 3.0, 2.0, 2.0]);
    assert_eq!((s.held(1), s.held(2)), ([1].into(), [1].into()));

    // A decode at the wrong position fails on the first worker stage.
    let err = forward(l, frame(1, StageOp::Decode, 9, &[0.0]))
        .await
        .unwrap_err();
    let stage = err.downcast_ref::<StageError>().expect("a stage error");
    assert_eq!((stage.rank, stage.request_id), (1, 1));

    let tokens = StagePayload::Tokens(vec![1]);
    let header = frame(2, StageOp::Prefill, 0, &[0.0]).0;
    let err = l.pipeline_forward(header, &tokens).await.unwrap_err();
    assert!(err.to_string().contains("takes the activation"), "{err}");

    l.release(1).await.unwrap();
    // A release travels ahead of every later frame, so after this reply
    // each stage has freed request 1.
    forward(l, frame(3, StageOp::Prefill, 0, &[0.0]))
        .await
        .unwrap();
    assert_eq!((s.held(1), s.held(2)), ([3].into(), [3].into()));
    s.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stage_error_fails_one_request_and_the_worker_keeps_serving() {
    let s = Swarm::start(Options {
        fail_rank: Some(2),
        ..opts(3)
    })
    .await;
    let l = &s.leader;
    let err = forward(l, frame(FAIL_ID, StageOp::Prefill, 0, &[1.0]))
        .await
        .unwrap_err();
    let stage = err.downcast_ref::<StageError>().expect("a stage error");
    assert_eq!((stage.rank, stage.request_id), (2, FAIL_ID));
    assert!(stage.message.contains("boom"), "{stage}");
    // Rank 2 freed the failed request. Rank 1 holds it until the release.
    assert!(!s.held(2).contains(&FAIL_ID));
    assert!(s.held(1).contains(&FAIL_ID));

    let row = forward(l, frame(7, StageOp::Prefill, 0, &[1.0]))
        .await
        .unwrap();
    assert_eq!(row, [7.0, 0.0, 3.0, 2.0]);
    l.release(FAIL_ID).await.unwrap();
    forward(l, frame(7, StageOp::Decode, 1, &[1.0]))
        .await
        .unwrap();
    assert_eq!((s.held(1), s.held(2)), ([7].into(), [7].into()));
    s.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_requests_fill_the_pipeline_and_never_cross() {
    let s = Swarm::start(Options {
        delay: Duration::from_millis(20),
        ..opts(3)
    })
    .await;
    let leader = Arc::new(s.leader);
    let mut set = JoinSet::new();
    for id in 1..=8u64 {
        let leader = Arc::clone(&leader);
        set.spawn(async move {
            let hidden = [id as f32, 10.0 * id as f32];
            let prefill = forward(&leader, frame(id, StageOp::Prefill, 0, &hidden)).await?;
            let decode = forward(&leader, frame(id, StageOp::Decode, 2, &[id as f32])).await?;
            leader.release(id).await?;
            Ok::<_, anyhow::Error>((id, prefill, decode))
        });
    }
    while let Some(done) = set.join_next().await {
        let (id, prefill, decode) = done.unwrap().unwrap();
        let x = id as f32;
        assert_eq!(prefill, [x, 0.0, 11.0 * x + 4.0, 2.0], "request {id}");
        assert_eq!(decode, [x, 2.0, x + 2.0, 2.0], "request {id}");
    }
    let Ok(leader) = Arc::try_unwrap(leader) else {
        panic!("every request task dropped its leader handle");
    };
    Swarm {
        leader,
        workers: s.workers,
        held: s.held,
    }
    .stop()
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lost_stage_fails_the_request_in_flight_and_every_later_one() {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let mut clients = NexarClient::bootstrap_local(2, cpu_adapter())
        .await
        .unwrap();
    clients.sort_by_key(|c| c.rank());
    let worker = clients.pop().unwrap();
    let leader =
        Arc::new(SwarmLeader::by_rank(Arc::new(clients.pop().unwrap()), LAYERS, VOCAB).unwrap());
    // Rank 1 runs no compute loop: the request waits until rank 1 goes away.
    let waiting = {
        let leader = Arc::clone(&leader);
        tokio::spawn(async move { forward(&leader, frame(1, StageOp::Prefill, 0, &[1.0])).await })
    };
    tokio::time::sleep(Duration::from_millis(100)).await;
    // A dropped client closes its connections at once, as a crashed process does.
    drop(worker);
    let err = tokio::time::timeout(Duration::from_secs(30), waiting)
        .await
        .expect("a lost stage ends the waiting request")
        .unwrap()
        .unwrap_err();
    assert!(err.to_string().contains("lost stage rank 1"), "{err}");
    // The dispatcher refuses it, or the send to the gone rank fails first.
    let err = forward(&leader, frame(2, StageOp::Prefill, 0, &[1.0]))
        .await
        .unwrap_err();
    assert!(format!("{err:#}").contains("rank 1"), "{err:#}");
    let Ok(leader) = Arc::try_unwrap(leader) else {
        panic!("the request task dropped its leader handle");
    };
    let _ = tokio::time::timeout(WAIT, leader.shutdown(WAIT)).await;
}
