//! Stopping the workers of a disaggregated cluster.

use super::super::protocol::tags;
use super::super::wire::send_frame;
use super::routing::DisaggRouter;

impl DisaggRouter {
    /// Sends every prefill and decode worker the [`tags::SHUTDOWN`] notice.
    ///
    /// Each worker ends its run loop after the request in progress. Call it
    /// once the router admits no more requests, then shut down every rank's
    /// nexar client. A failed send to one worker is logged, and the others
    /// still get theirs.
    pub async fn stop_workers(&self) {
        let workers = self
            .config
            .prefill_workers
            .iter()
            .chain(&self.config.decode_workers);
        for &rank in workers {
            if let Err(e) = send_frame(&self.shared.client, rank, tags::SHUTDOWN, &[1]).await {
                tracing::warn!(rank, "shutdown notice not sent: {e:#}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::protocol::GenerationParams;
    use super::super::super::test_harness::{Cluster, MockDecode};
    use std::time::Duration;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stop_workers_ends_both_run_loops() {
        let mut c = Cluster::start(None, MockDecode::default()).await;
        let g = c
            .router
            .route_request(&[1], GenerationParams::greedy(3), None)
            .await
            .unwrap();
        assert_eq!(g.tokens, vec![2, 3, 4]);
        c.router.stop_workers().await;
        for worker in c.take_workers() {
            let ended = tokio::time::timeout(Duration::from_secs(10), worker)
                .await
                .expect("the worker loop ends after the shutdown notice");
            ended.unwrap().unwrap();
        }
    }
}
