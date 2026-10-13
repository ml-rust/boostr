//! [`CancelToken`]: a cooperative stop flag a caller sets from any thread
//! and a decision run reads between its stages.

use crate::error::{Error, Result};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A shared stop flag. Clones share one flag.
///
/// A run reads the flag before the prefix, before each pass and before
/// scoring. A pass already on the device runs to its end, so a cancel
/// takes effect at the next check.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// A token that is not cancelled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the flag. Every clone sees it.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// `true` once any clone called [`CancelToken::cancel`].
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    /// `Ok` while the flag is clear.
    ///
    /// # Errors
    ///
    /// [`Error::Cancelled`] carrying `at` once the flag is set.
    pub fn check(&self, at: impl FnOnce() -> String) -> Result<()> {
        if self.is_cancelled() {
            Err(Error::Cancelled { at: at() })
        } else {
            Ok(())
        }
    }
}

/// Two tokens are equal when they share one flag, that is, when one is a
/// clone of the other. The flag's value is not compared: two separate
/// tokens are unequal whether or not either is cancelled.
impl PartialEq for CancelToken {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for CancelToken {}

#[cfg(test)]
mod tests {
    use super::super::backbone::{DecisionBackbone, PrefixState};
    use super::super::packing::RunnerConfig;
    use super::super::plan::DecisionPlan;
    use super::super::runner::tests::{model, plan, run};
    use super::*;
    use crate::inference::{LayeredGdnState, LayeredKvCache};
    use crate::model::qwen35::Qwen35Model;
    use crate::model::traits::ModelClient;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;
    use std::cell::Cell;

    /// The test model, which sets `token` after its `after`-th pass.
    struct CancelAfter<'a> {
        inner: &'a Qwen35Model<CpuRuntime>,
        token: CancelToken,
        after: usize,
        passes: Cell<usize>,
    }

    impl DecisionBackbone<CpuRuntime> for CancelAfter<'_> {
        fn max_positions(&self) -> usize {
            DecisionBackbone::max_positions(self.inner)
        }

        fn vocab_size(&self) -> usize {
            DecisionBackbone::vocab_size(self.inner)
        }

        fn lane_bytes(&self, width: usize) -> usize {
            DecisionBackbone::lane_bytes(self.inner, width)
        }

        fn prefill_prefix<C: ModelClient<CpuRuntime>>(
            &self,
            client: &C,
            tokens: &[u32],
            rows: &[usize],
        ) -> Result<PrefixState<CpuRuntime>> {
            DecisionBackbone::prefill_prefix(self.inner, client, tokens, rows)
        }

        fn lane_caches(
            &self,
            lanes: usize,
            width: usize,
            device: &CpuDevice,
        ) -> Result<(LayeredKvCache<CpuRuntime>, LayeredGdnState<CpuRuntime>)> {
            DecisionBackbone::lane_caches(self.inner, lanes, width, device)
        }

        fn forward_rows<C: ModelClient<CpuRuntime>>(
            &self,
            client: &C,
            ids: &Tensor<CpuRuntime>,
            prefix: &PrefixState<CpuRuntime>,
            gdn: &mut LayeredGdnState<CpuRuntime>,
            kv: &mut LayeredKvCache<CpuRuntime>,
            gather: &Tensor<CpuRuntime>,
        ) -> Result<Tensor<CpuRuntime>> {
            let out =
                DecisionBackbone::forward_rows(self.inner, client, ids, prefix, gdn, kv, gather)?;
            self.passes.set(self.passes.get() + 1);
            if self.passes.get() == self.after {
                self.token.cancel();
            }
            Ok(out)
        }

        fn candidate_logits<C: ModelClient<CpuRuntime>>(
            &self,
            client: &C,
            hidden: &Tensor<CpuRuntime>,
            ids: &Tensor<CpuRuntime>,
        ) -> Result<Tensor<CpuRuntime>> {
            DecisionBackbone::candidate_logits(self.inner, client, hidden, ids)
        }
    }

    fn one_lane(token: &CancelToken) -> RunnerConfig {
        RunnerConfig {
            max_lanes: 1,
            cancel: Some(token.clone()),
            ..RunnerConfig::default()
        }
    }

    fn cancelled_at(plan: &DecisionPlan, after: usize, token: &CancelToken) -> (String, usize) {
        let (client, device) = cpu_setup();
        let inner = model(&device);
        let backbone = CancelAfter {
            inner: &inner,
            token: token.clone(),
            after,
            passes: Cell::new(0),
        };
        let err = run(&client, &backbone, plan, None, &one_lane(token))
            .expect_err("a cancelled run must not return answers");
        assert!(matches!(err, Error::Cancelled { .. }), "{err}");
        (err.to_string(), backbone.passes.get())
    }

    #[test]
    fn clones_share_one_flag() {
        let a = CancelToken::new();
        let b = a.clone();
        assert!(!b.is_cancelled());
        assert!(a.check(|| "here".into()).is_ok());
        a.cancel();
        assert!(b.is_cancelled());
        let e = b.check(|| "here".into()).unwrap_err().to_string();
        assert_eq!(e, "cancelled here");
        assert_eq!(a, b);
        assert_ne!(a, CancelToken::new());
    }

    /// A token set before the run stops it before any device work.
    #[test]
    fn cancel_before_start() {
        let token = CancelToken::new();
        token.cancel();
        let (err, passes) = cancelled_at(&plan(6), usize::MAX, &token);
        assert!(err.contains("before the prefix"), "{err}");
        assert_eq!(passes, 0);
    }

    /// One lane per pass gives five passes; a cancel during the second
    /// stops the run before the third.
    #[test]
    fn cancel_between_passes() {
        let token = CancelToken::new();
        let (err, passes) = cancelled_at(&plan(6), 2, &token);
        assert!(err.contains("before pass 3 of 5"), "{err}");
        assert_eq!(passes, 2);
    }

    /// A cancel during the last pass stops the run before scoring.
    #[test]
    fn cancel_before_scoring() {
        let token = CancelToken::new();
        let (err, passes) = cancelled_at(&plan(6), 5, &token);
        assert!(err.contains("before scoring"), "{err}");
        assert_eq!(passes, 5);
    }
}
