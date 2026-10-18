//! Sequence state accounting. Every state counter changes here and
//! nowhere else. A transition leaves the state a sequence is in and
//! enters the next one.

use crate::error::{Error, Result};
use crate::inference::memory::BlockAllocator;

use super::core::SequenceScheduler;
use super::types::{SequenceId, SequenceState};

impl<A: BlockAllocator> SequenceScheduler<A> {
    /// The counter that tracks `state`.
    fn counter_mut(&mut self, state: SequenceState) -> &mut usize {
        match state {
            SequenceState::Waiting => &mut self.stats.waiting_count,
            SequenceState::Running => &mut self.stats.running_count,
            SequenceState::Preempted => &mut self.stats.preempted_count,
            SequenceState::Finished => &mut self.stats.finished_count,
        }
    }

    /// Count one sequence out of `state`.
    ///
    /// The caller has checked the state with [`require_state`](Self::require_state),
    /// which also checks the counter is above 0. A counter at 0 stays at 0.
    pub(super) fn leave_state(&mut self, state: SequenceState) {
        let counter = self.counter_mut(state);
        if let Some(next) = counter.checked_sub(1) {
            *counter = next;
        }
    }

    /// Count one sequence into `state`.
    pub(super) fn enter_state(&mut self, state: SequenceState) {
        let counter = self.counter_mut(state);
        *counter = counter.saturating_add(1);
    }

    /// Read the state of `seq_id` and check it is one of `expected`.
    ///
    /// Returns the current state. Changes nothing. `op` names the requested
    /// operation in the error. An unknown id is an error, because a finished
    /// sequence leaves the scheduler at `cleanup_finished`.
    pub(super) fn require_state(
        &self,
        seq_id: SequenceId,
        op: &str,
        expected: &[SequenceState],
    ) -> Result<SequenceState> {
        let Some(seq) = self.sequences.get(&seq_id) else {
            return Err(Self::not_found(seq_id, op));
        };
        let state = seq.state;
        if !expected.contains(&state) {
            return Err(Error::SchedulerError {
                reason: format!(
                    "Cannot {op} sequence {seq_id}: its state is {state:?}, \
                     and {op} needs {expected:?}. Read the state with \
                     get_sequence_state before {op}"
                ),
            });
        }
        let counted = match state {
            SequenceState::Waiting => self.stats.waiting_count,
            SequenceState::Running => self.stats.running_count,
            SequenceState::Preempted => self.stats.preempted_count,
            SequenceState::Finished => self.stats.finished_count,
        };
        if counted == 0 {
            return Err(Error::SchedulerError {
                reason: format!(
                    "Cannot {op} sequence {seq_id}: the {state:?} counter is 0 \
                     while the sequence is {state:?}. The scheduler state is \
                     corrupt. Drop this scheduler and create a new one"
                ),
            });
        }
        Ok(state)
    }

    /// Move `seq_id` from state `from` to state `to`.
    ///
    /// Checks the current state first. On an error nothing changes. The
    /// queues and the blocks of the sequence stay with the caller.
    pub(super) fn move_state(
        &mut self,
        seq_id: SequenceId,
        op: &str,
        from: SequenceState,
        to: SequenceState,
    ) -> Result<()> {
        self.require_state(seq_id, op, &[from])?;
        let Some(seq) = self.sequences.get_mut(&seq_id) else {
            return Err(Self::not_found(seq_id, op));
        };
        seq.state = to;
        self.leave_state(from);
        self.enter_state(to);
        Ok(())
    }

    /// The error for an id the scheduler does not track.
    pub(super) fn not_found(seq_id: SequenceId, op: &str) -> Error {
        Error::SchedulerError {
            reason: format!(
                "Cannot {op} sequence {seq_id}: the scheduler does not track it. \
                 It was never added or cleanup_finished removed it. \
                 Check has_sequence before {op}"
            ),
        }
    }
}
