//! Micro-batch scheduling logic for pipeline parallelism.
//!
//! [`PipelineClock`] computes the ordered sequence of [`PipelineAction`]s for
//! each pipeline stage given the schedule type (1F1B, interleaved, zero-bubble).

/// An action a pipeline stage must perform at a given clock tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineAction {
    /// Run forward pass for micro-batch `id`.
    Forward(usize),
    /// Run backward pass for micro-batch `id`.
    Backward(usize),
    /// Zero-bubble: compute only input gradient for micro-batch `id`.
    BackwardInput(usize),
    /// Zero-bubble: compute only weight gradient for micro-batch `id`.
    BackwardWeights(usize),
    /// No work this tick (pipeline bubble).
    Idle,
}

/// Computes the schedule of actions for a single pipeline stage.
pub struct PipelineClock {
    num_stages: usize,
    num_micro_batches: usize,
    stage_id: usize,
}

impl PipelineClock {
    pub fn new(num_stages: usize, num_micro_batches: usize, stage_id: usize) -> Self {
        Self {
            num_stages,
            num_micro_batches,
            stage_id,
        }
    }

    pub fn num_stages(&self) -> usize {
        self.num_stages
    }

    pub fn num_micro_batches(&self) -> usize {
        self.num_micro_batches
    }

    pub fn stage_id(&self) -> usize {
        self.stage_id
    }

    /// Standard 1F1B schedule.
    ///
    /// ```text
    /// Stage 0: F0 F1 F2 F3 B3 B2 B1 B0
    /// Stage 1:    F0 F1 F2 F3 B3 B2 B1 B0
    /// Stage 2:       F0 F1 F2 F3 B3 B2 B1 B0
    /// Stage 3:          F0 F1 F2 F3 B3 B2 B1 B0
    /// ```
    ///
    /// - Warmup: stage `k` does `num_stages - k - 1` extra forwards before steady state.
    /// - Steady state: alternate Forward + Backward.
    /// - Cooldown: remaining backwards.
    pub fn schedule_1f1b(&self) -> Vec<PipelineAction> {
        let s = self.num_stages;
        let m = self.num_micro_batches;
        let k = self.stage_id;

        // Number of warmup forwards = min(s - k - 1, m)
        let warmup = (s - k - 1).min(m);
        let mut actions = Vec::with_capacity(2 * m);
        let mut fwd_id = 0usize;
        let mut bwd_id = 0usize;

        // Warmup: extra forwards
        for _ in 0..warmup {
            actions.push(PipelineAction::Forward(fwd_id));
            fwd_id += 1;
        }

        // Steady state: 1 forward + 1 backward until all forwards done
        let steady = m - warmup;
        for _ in 0..steady {
            actions.push(PipelineAction::Forward(fwd_id));
            fwd_id += 1;
            actions.push(PipelineAction::Backward(bwd_id));
            bwd_id += 1;
        }

        // Cooldown: remaining backwards
        while bwd_id < m {
            actions.push(PipelineAction::Backward(bwd_id));
            bwd_id += 1;
        }

        actions
    }

    /// Interleaved 1F1B schedule for `num_virtual` virtual stages per rank.
    ///
    /// Each rank owns `num_virtual` non-contiguous stage chunks. Bubble ratio
    /// reduces from `(S-1)/M` to `(S-1)/(M*V)`.
    ///
    /// Returns a sequence of `(virtual_stage_idx, PipelineAction)`.
    ///
    /// Key ordering constraint: forwards go v=0→V-1 (low→high logical stage),
    /// backwards go v=V-1→0 (high→low, since gradients flow backward through
    /// the pipeline).
    pub fn schedule_interleaved(&self, num_virtual: usize) -> Vec<(usize, PipelineAction)> {
        let m = self.num_micro_batches;

        let mut actions = Vec::with_capacity(2 * m * num_virtual);

        // All forwards first, round-robin across virtual stages (v=0 first).
        // Each virtual stage processes micro-batches 0..m in order.
        for mb in 0..m {
            for v in 0..num_virtual {
                actions.push((v, PipelineAction::Forward(mb)));
            }
        }

        // All backwards, round-robin across virtual stages in REVERSE order
        // (v=V-1 first, since gradients flow backward through the pipeline).
        for mb in 0..m {
            for v in (0..num_virtual).rev() {
                actions.push((v, PipelineAction::Backward(mb)));
            }
        }

        actions
    }

    /// Zero-bubble schedule that splits backward into B (input grad) and W (weight grad).
    ///
    /// W passes are scheduled into pipeline bubbles. This achieves near-zero bubble
    /// ratio when `M >= S`.
    pub fn schedule_zero_bubble(&self) -> Vec<PipelineAction> {
        let s = self.num_stages;
        let m = self.num_micro_batches;
        let k = self.stage_id;

        let warmup = (s - k - 1).min(m);
        let mut actions = Vec::with_capacity(3 * m);
        let mut fwd_id = 0usize;
        let mut b_id = 0usize; // BackwardInput id
        let mut w_id = 0usize; // BackwardWeights id

        // Warmup: forwards, interleaved with W passes to fill bubbles
        for i in 0..warmup {
            actions.push(PipelineAction::Forward(fwd_id));
            fwd_id += 1;
            // After the first couple of warmup forwards, we can start scheduling
            // W passes for completed B passes from the previous iteration
            if i > 0 && w_id < b_id {
                actions.push(PipelineAction::BackwardWeights(w_id));
                w_id += 1;
            }
        }

        // Steady state: F, B, W interleaved
        let steady = m - warmup;
        for _ in 0..steady {
            actions.push(PipelineAction::Forward(fwd_id));
            fwd_id += 1;
            actions.push(PipelineAction::BackwardInput(b_id));
            b_id += 1;
            // Schedule W for a completed B pass
            if w_id < b_id {
                actions.push(PipelineAction::BackwardWeights(w_id));
                w_id += 1;
            }
        }

        // Cooldown: remaining B and W passes
        while b_id < m {
            actions.push(PipelineAction::BackwardInput(b_id));
            b_id += 1;
            if w_id < b_id {
                actions.push(PipelineAction::BackwardWeights(w_id));
                w_id += 1;
            }
        }

        // Any remaining W passes
        while w_id < m {
            actions.push(PipelineAction::BackwardWeights(w_id));
            w_id += 1;
        }

        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1f1b_single_stage() {
        let clock = PipelineClock::new(1, 4, 0);
        let actions = clock.schedule_1f1b();
        // Single stage: no warmup, steady = all 4, immediate F+B pairs
        assert_eq!(
            actions,
            vec![
                PipelineAction::Forward(0),
                PipelineAction::Backward(0),
                PipelineAction::Forward(1),
                PipelineAction::Backward(1),
                PipelineAction::Forward(2),
                PipelineAction::Backward(2),
                PipelineAction::Forward(3),
                PipelineAction::Backward(3),
            ]
        );
    }

    #[test]
    fn test_1f1b_first_stage_4stages_4mb() {
        let clock = PipelineClock::new(4, 4, 0);
        let actions = clock.schedule_1f1b();
        // Stage 0 warmup = 3, steady = 1, cooldown = 3
        // Warmup: F0, F1, F2
        // Steady: F3, B0
        // Cooldown: B1, B2, B3
        assert_eq!(
            actions,
            vec![
                PipelineAction::Forward(0),
                PipelineAction::Forward(1),
                PipelineAction::Forward(2),
                PipelineAction::Forward(3),
                PipelineAction::Backward(0),
                PipelineAction::Backward(1),
                PipelineAction::Backward(2),
                PipelineAction::Backward(3),
            ]
        );
    }

    #[test]
    fn test_1f1b_last_stage_4stages_4mb() {
        let clock = PipelineClock::new(4, 4, 3);
        let actions = clock.schedule_1f1b();
        // Stage 3 (last) warmup = 0, all steady: F+B pairs
        assert_eq!(
            actions,
            vec![
                PipelineAction::Forward(0),
                PipelineAction::Backward(0),
                PipelineAction::Forward(1),
                PipelineAction::Backward(1),
                PipelineAction::Forward(2),
                PipelineAction::Backward(2),
                PipelineAction::Forward(3),
                PipelineAction::Backward(3),
            ]
        );
    }

    #[test]
    fn test_1f1b_middle_stage() {
        let clock = PipelineClock::new(4, 6, 1);
        let actions = clock.schedule_1f1b();
        // Stage 1 warmup = 2
        // Warmup: F0, F1
        // Steady (4): F2 B0, F3 B1, F4 B2, F5 B3
        // Cooldown: B4, B5
        assert_eq!(
            actions,
            vec![
                PipelineAction::Forward(0),
                PipelineAction::Forward(1),
                PipelineAction::Forward(2),
                PipelineAction::Backward(0),
                PipelineAction::Forward(3),
                PipelineAction::Backward(1),
                PipelineAction::Forward(4),
                PipelineAction::Backward(2),
                PipelineAction::Forward(5),
                PipelineAction::Backward(3),
                PipelineAction::Backward(4),
                PipelineAction::Backward(5),
            ]
        );
    }

    #[test]
    fn test_1f1b_action_count() {
        // Every 1F1B schedule produces exactly 2*M actions (M forwards + M backwards)
        for stages in 1..=6 {
            for mb in 1..=8 {
                for stage_id in 0..stages {
                    let clock = PipelineClock::new(stages, mb, stage_id);
                    let actions = clock.schedule_1f1b();
                    let fwd_count = actions
                        .iter()
                        .filter(|a| matches!(a, PipelineAction::Forward(_)))
                        .count();
                    let bwd_count = actions
                        .iter()
                        .filter(|a| matches!(a, PipelineAction::Backward(_)))
                        .count();
                    assert_eq!(fwd_count, mb, "stages={stages} mb={mb} stage={stage_id}");
                    assert_eq!(bwd_count, mb, "stages={stages} mb={mb} stage={stage_id}");
                }
            }
        }
    }

    #[test]
    fn test_zero_bubble_covers_all_mb() {
        for stages in 1..=4 {
            for mb in stages..=8 {
                for stage_id in 0..stages {
                    let clock = PipelineClock::new(stages, mb, stage_id);
                    let actions = clock.schedule_zero_bubble();
                    let fwd = actions
                        .iter()
                        .filter(|a| matches!(a, PipelineAction::Forward(_)))
                        .count();
                    let bi = actions
                        .iter()
                        .filter(|a| matches!(a, PipelineAction::BackwardInput(_)))
                        .count();
                    let bw = actions
                        .iter()
                        .filter(|a| matches!(a, PipelineAction::BackwardWeights(_)))
                        .count();
                    assert_eq!(fwd, mb, "fwd: stages={stages} mb={mb} stage={stage_id}");
                    assert_eq!(bi, mb, "bi: stages={stages} mb={mb} stage={stage_id}");
                    assert_eq!(bw, mb, "bw: stages={stages} mb={mb} stage={stage_id}");
                }
            }
        }
    }

    #[test]
    fn test_interleaved_covers_all_mb() {
        let clock = PipelineClock::new(2, 4, 0);
        let actions = clock.schedule_interleaved(2);
        // Each of 2 virtual stages should see 4 forwards and 4 backwards
        for v in 0..2 {
            let fwd = actions
                .iter()
                .filter(|&&(vs, ref a)| vs == v && matches!(a, PipelineAction::Forward(_)))
                .count();
            let bwd = actions
                .iter()
                .filter(|&&(vs, ref a)| vs == v && matches!(a, PipelineAction::Backward(_)))
                .count();
            assert_eq!(fwd, 4, "virtual stage {v} forward count");
            assert_eq!(bwd, 4, "virtual stage {v} backward count");
        }
    }
}
