//! Lost workers and the router's fault state, for a server's health check.

use nexar::Rank;

use super::routing::DisaggRouter;

impl DisaggRouter {
    /// The workers the router lost, in rank order.
    ///
    /// A worker is lost when its event channel closes or nexar reports it
    /// dead. A lost worker stays lost, and the router routes no request to it.
    pub fn lost_workers(&self) -> Vec<Rank> {
        self.shared.lost_ranks().into_iter().collect()
    }

    /// Whether the router lost at least one worker.
    ///
    /// A degraded router still serves while [`Self::fault`] is `None`. It
    /// takes one short lock and allocates nothing.
    pub fn is_degraded(&self) -> bool {
        self.shared.any_lost()
    }

    /// Why the router can serve no request: it lost every prefill worker or
    /// every decode worker. `None` while one of each is live.
    ///
    /// A request routed during a fault fails at once with the same reason.
    pub fn fault(&self) -> Option<String> {
        let lost = self.shared.lost_ranks();
        let gone = |ranks: &[Rank]| ranks.iter().all(|r| lost.contains(r));
        let mut roles = Vec::new();
        if gone(&self.config.prefill_workers) {
            roles.push(format!("prefill ranks {:?}", self.config.prefill_workers));
        }
        if gone(&self.config.decode_workers) {
            roles.push(format!("decode ranks {:?}", self.config.decode_workers));
        }
        if roles.is_empty() {
            return None;
        }
        Some(format!(
            "the disaggregated router lost every worker of a role: {}. Restart the cluster",
            roles.join(" and ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::config::{DEFAULT_HANDOFF_TIMEOUT, DisaggConfig};
    use super::super::super::protocol::GenerationParams;
    use super::super::super::test_harness::{
        MockDecode, decode_worker_on, local_clients, prefill_worker,
    };
    use super::super::super::wire::TaskGuard;
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    const WAIT: Duration = Duration::from_secs(10);

    fn layout(prefill: Vec<Rank>, decode: Vec<Rank>) -> DisaggConfig {
        DisaggConfig {
            prefill_workers: prefill,
            decode_workers: decode,
            ..super::super::super::test_harness::config(DEFAULT_HANDOFF_TIMEOUT)
        }
    }

    async fn router(config: DisaggConfig) -> (DisaggRouter, Vec<Arc<nexar::NexarClient>>) {
        let mut clients: Vec<_> = local_clients(config.world_size())
            .await
            .into_iter()
            .map(Arc::new)
            .collect();
        let rest = clients.split_off(1);
        let router = DisaggRouter::new(clients.remove(0), config).unwrap();
        (router, rest)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_lost_prefill_rank_is_never_chosen() {
        let (r, _workers) = router(layout(vec![1, 2], vec![3])).await;
        assert!(!r.is_degraded() && r.fault().is_none());
        r.shared.fail_rank(1, "test loss");
        assert!(r.is_degraded());
        assert_eq!(r.lost_workers(), [1]);
        // Rank 1 has no load and rank 2 has some: the lost rank still loses.
        r.shared.prefill_loads[1]
            .in_flight
            .store(5, std::sync::atomic::Ordering::Relaxed);
        for _ in 0..8 {
            assert_eq!(r.choose_prefill_worker().unwrap(), 2);
        }
        assert!(r.fault().is_none(), "rank 2 still serves prefill");
        r.shared.fail_rank(2, "test loss");
        let fault = r.fault().expect("no prefill worker is live");
        assert!(fault.contains("prefill ranks [1, 2]"), "{fault}");
        let err = tokio::time::timeout(
            WAIT,
            r.route_request_stream(&[1], GenerationParams::greedy(2), None),
        )
        .await
        .expect("the request fails at once")
        .err()
        .expect("no live prefill worker");
        assert!(err.to_string().contains("no live prefill worker"), "{err}");
        assert_eq!(r.shared.tracked(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn round_robin_and_affinity_skip_a_lost_decode_rank() {
        let (r, _workers) = router(layout(vec![1], vec![2, 3])).await;
        r.record_kv_affinity("s".into(), 2);
        r.shared.fail_rank(2, "test loss");
        for _ in 0..6 {
            assert_eq!(r.choose_decode_worker(None).unwrap(), 3);
        }
        assert_eq!(r.choose_decode_worker(Some("s")).unwrap(), 3);
        assert_eq!(
            r.kv_affinity_of("s"),
            None,
            "the pin to the lost rank is gone"
        );
        r.shared.fail_rank(3, "test loss");
        assert!(r.fault().unwrap().contains("decode ranks [2, 3]"));
        let err = tokio::time::timeout(
            WAIT,
            r.route_request_stream(&[1], GenerationParams::greedy(2), Some("s")),
        )
        .await
        .expect("the request fails at once")
        .err()
        .expect("no live decode worker");
        assert!(err.to_string().contains("no live decode worker"), "{err}");
    }

    /// Router 0, prefill 1, decode 2 and 3. The session first pins to decode
    /// rank 2. Rank 2 then closes, and the next request of the session
    /// finishes on rank 3.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_session_pinned_to_a_lost_decode_rank_moves_to_a_live_one() {
        let (r, workers) = router(layout(vec![1], vec![2, 3])).await;
        let p = prefill_worker(Arc::clone(&workers[0]), None, DEFAULT_HANDOFF_TIMEOUT);
        let mock = MockDecode::default();
        let d2 = decode_worker_on(Arc::clone(&workers[1]), 2, &mock);
        let d3 = decode_worker_on(Arc::clone(&workers[2]), 3, &mock);
        let _tasks = [
            TaskGuard::new(tokio::spawn(async move { p.run_loop().await })),
            TaskGuard::new(tokio::spawn(async move { d2.run_loop().await })),
            TaskGuard::new(tokio::spawn(async move { d3.run_loop().await })),
        ];

        let s = r
            .route_request_stream(&[1], GenerationParams::greedy(3), Some("s"))
            .await
            .unwrap();
        assert_eq!(s.decode_rank(), 2);
        assert_eq!(s.collect().await.unwrap().tokens, [2, 3, 4]);

        workers[1].close();
        let deadline = tokio::time::Instant::now() + WAIT;
        while r.lost_workers() != [2] {
            assert!(
                tokio::time::Instant::now() < deadline,
                "rank 2 never counted as lost"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let s = r
            .route_request_stream(&[1], GenerationParams::greedy(3), Some("s"))
            .await
            .unwrap();
        assert_eq!(s.decode_rank(), 3, "the session moved off the lost rank");
        let g = tokio::time::timeout(WAIT, s.collect())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(g.tokens, [2, 3, 4]);
        assert_eq!(r.kv_affinity_of("s"), Some(3));
        assert!(r.is_degraded() && r.fault().is_none());
    }
}
